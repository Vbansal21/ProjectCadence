"""
Long-Arena Benchmark for LSH v2 Model
=====================================

A feature-complete, robust evaluation harness for long-sequence models.

Highlights
----------
- Powers-of-two sequence scaling from 1K → 128K (configurable)
- Exact latency on GPU via CUDA events; high-resolution wall-time on CPU
- Throughput, latency, memory (peak + delta), perplexity (next-token)
- Early-stop policies (fail-fast on repeated OOM/timeout/instability)
- Optional AMP autocast (fp16/bf16) for realistic throughput/efficiency
- Optional per-sample lightweight profiling
- Smoke test mode
- Deterministic seeds for repeatability
- Results persisted as:
    - data/results/benchmark_summary.json   (for pipeline consolidator)
    - <output_dir>/{benchmark_summary.json, long_arena_results.json, long_arena_benchmark.png}
    - Optional: <output_dir>/profiles/*.txt
- Graceful fallbacks if data generator or checkpoint is unavailable

Directory assumptions
---------------------
.
├─ benchmarks/
│   └─ long_arena_benchmark.py        # (this file)
├─ data/
│   ├─ synthetic_data_generator.py
│   └─ results/
└─ src/
    └─ lsh_v2_model.py                # provides create_lsh_v2_model(...)

Author: Vaibhav Bansal
Version: 2.2
"""

from __future__ import annotations

import sys
import json
import gc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

try:
    import psutil
except Exception:
    psutil = None  # CPU memory becomes best-effort if psutil is absent

# Add src & data to import path
THIS_DIR = Path(__file__).parent
ROOT_DIR = THIS_DIR.parent
sys.path.append(str(ROOT_DIR / "src"))
sys.path.append(str(ROOT_DIR / "data"))

# --- Model & data imports (deferred; validated in __init__) ---
# from lsh_v2_model import create_lsh_v2_model
# from synthetic_data_generator import SyntheticDataGenerator


# ------------------------------ Utilities ------------------------------


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _dev_str() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(torch.cuda.current_device())
        return f"cuda:{torch.cuda.current_device()} ({name})"
    return "cpu"


def _set_determinism(seed: int = 2025):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Allow TF32 for realistic perf; does not break determinism meaningfully for benchmarking
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


# ------------------------------ Config ------------------------------


@dataclass
class BenchmarkConfig:
    # Model / data
    model_path: str = ""  # optional path to a checkpoint (.pt)
    vocab_size: int = 32000
    pad_token_id: int = 0  # used for batch padding if batch>1 in smoke/micro runs

    # Workload
    sequence_lengths: List[int] = field(
        default_factory=lambda: [2**i for i in range(10, 18)]
    )  # 1K..128K
    num_samples_per_length: int = 5
    batch_size: int = 1  # primarily for smoke tests; main eval uses B=1 for clarity

    # Performance / limits
    max_memory_gb: float = 16.0  # soft cap; if crossed, we early exit current length
    timeout_seconds: int = 300  # per-sample soft timeout (best-effort)
    warmup_runs: int = 1  # number of non-recorded warmups per length (for caches)

    # Mixed-precision (AMP) controls
    use_autocast: bool = True  # if True and device supports, run under autocast
    autocast_dtype: str = (
        "bf16"  # "bf16", "fp16", "none"  (bf16 preferred for stability)
    )

    # Profiling & logging
    profile_per_sample: bool = (
        False  # very lightweight autograd profiler (one sample/length recommended)
    )
    profiles_dir: str = "data/results/quick_benchmark/profiles"
    log_file: str = "data/results/quick_benchmark/benchmark.log"

    # Misc
    fail_fast_oom: int = 2  # stop a seq_len after this many OOMs
    fail_fast_exceptions: int = 3  # stop a seq_len after this many generic failures
    smoke: bool = False  # if True: run a quick 512/1K micro-check and exit

    def resolve_autocast_dtype(self):
        if not self.use_autocast:
            return None
        if self.autocast_dtype.lower() == "fp16":
            return torch.float16
        if self.autocast_dtype.lower() == "bf16":
            return torch.bfloat16
        return None


# ------------------------------ Benchmark Harness ------------------------------


class LongArenaBenchmark:
    """Long-sequence scaling benchmark with precision, profiling, and robust metrics."""

    def __init__(self, config: BenchmarkConfig):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Prepare dirs
        Path("data/results").mkdir(parents=True, exist_ok=True)

        # Logger
        self.log_path = Path(self.cfg.log_file)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.log_path, "a", encoding="utf-8")
        self._log(f"== Long-Arena init @ {_now()} ==")
        self._log(
            f"Device: {_dev_str()} | AMP: {self.cfg.use_autocast} ({self.cfg.autocast_dtype})"
        )

        # Determinism
        _set_determinism(2025)

        # Import model factory lazily to avoid hard crash if missing
        try:
            from lsh_v2_model import create_lsh_v2_model

            self.model = create_lsh_v2_model(self.cfg.vocab_size, "100M")
        except Exception as e:
            raise RuntimeError(
                f"Could not import/create model from src/lsh_v2_model.py: {e}"
            )

        # Load checkpoint if provided
        if self.cfg.model_path and Path(self.cfg.model_path).exists():
            try:
                ckpt = torch.load(self.cfg.model_path, map_location="cpu")
                sd = ckpt.get("model_state_dict", ckpt)
                self.model.load_state_dict(sd, strict=False)
                self._log(f"Loaded checkpoint: {self.cfg.model_path}")
            except Exception as e:
                self._log(
                    f"WARNING: failed to load checkpoint '{self.cfg.model_path}': {e}"
                )

        self.model.to(self.device).eval()

        # Data generator (optional; fallback to random if unavailable)
        self.data_gen = None
        try:
            from synthetic_data_generator import SyntheticDataGenerator

            self.data_gen = SyntheticDataGenerator(
                vocab_size=self.cfg.vocab_size,
                max_seq_len=max(self.cfg.sequence_lengths + [2048]),
            )
            self._log("SyntheticDataGenerator available.")
        except Exception as e:
            self._log(
                f"NOTE: SyntheticDataGenerator unavailable, falling back to random tokens: {e}"
            )

        # Results accumulator
        self.results: Dict[str, list] = {
            "sequence_lengths": [],
            "latency_s": [],
            "throughput_tok_s": [],
            "mem_used_gb": [],
            "mem_peak_gb": [],
            "perplexity": [],
            "success_rate": [],
        }
        self.scaling_analysis: Dict[str, float] = {}

    # -------------------------- Logging --------------------------

    def _log(self, msg: str):
        print(msg, flush=True)
        try:
            self._fh.write(msg + ("\n" if not msg.endswith("\n") else ""))
            self._fh.flush()
        except Exception:
            pass

    # -------------------------- Memory --------------------------

    @staticmethod
    def _cpu_mem_gb() -> float:
        if psutil is None:
            return 0.0
        return psutil.Process().memory_info().rss / (1024**3)

    def _mem_state(self) -> Tuple[float, float]:
        """Return (allocated_gb, peak_gb) appropriate to device."""
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024**3)
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            return alloc, peak
        # CPU fallback: use RSS for both alloc/peak proxies
        rss = self._cpu_mem_gb()
        return rss, rss

    # -------------------------- Data --------------------------

    def _make_batch(self, seq_len: int) -> torch.Tensor:
        """
        Return [B, N] tokens for evaluation.
        If data generator exists, prefer a mixed task sequence; else random ints in vocab.
        """
        B = self.cfg.batch_size
        if self.data_gen is not None:
            s = self.data_gen.generate_mixed_task_sequence(seq_len)
            toks = torch.tensor(s["tokens"], dtype=torch.long).unsqueeze(0)  # [1, N]
            if B > 1:
                # For benchmarking, replicate the same sample to avoid data skew
                toks = toks.repeat(B, 1)
            return toks
        # Fallback: random tokens
        toks = torch.randint(0, self.cfg.vocab_size, (B, seq_len), dtype=torch.long)
        return toks

    # -------------------------- Timed Forward --------------------------

    def _forward_once(
        self, input_ids: torch.Tensor, autocast_dtype: Optional[torch.dtype]
    ) -> Tuple[float, float, float, float, float]:
        """
        Execute one forward pass and return:
          (latency_s, tok_per_s, delta_mem_gb, peak_mem_gb, perplexity)
        """
        # Prepare
        input_ids = input_ids.to(self.device, non_blocking=True)
        N = input_ids.size(1)
        B = input_ids.size(0)

        # Reset GPU peak memory stats to measure this pass precisely
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        mem_before, _ = self._mem_state()

        # Timers
        use_gpu_timer = torch.cuda.is_available()
        if use_gpu_timer:
            start_evt, end_evt = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )

        # Forward
        ctx = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=True)
            if (torch.cuda.is_available() and autocast_dtype is not None)
            else (
                torch.autocast(device_type="cpu", dtype=autocast_dtype, enabled=True)
                if (
                    not torch.cuda.is_available()
                    and autocast_dtype is not None
                    and autocast_dtype in (torch.bfloat16,)
                )
                else torch.no_grad()
            )
        )

        with ctx:
            if use_gpu_timer:
                torch.cuda.synchronize()
                start_evt.record()
            else:
                t0 = time.perf_counter()

            with torch.inference_mode():
                outputs = self.model(input_ids)

            if use_gpu_timer:
                end_evt.record()
                torch.cuda.synchronize()
                latency_s = start_evt.elapsed_time(end_evt) / 1000.0
            else:
                latency_s = time.perf_counter() - t0

        # Metrics
        _, peak_gb = self._mem_state()
        mem_after, _ = self._mem_state()
        delta_mem_gb = max(0.0, mem_after - mem_before)
        tok_per_s = (B * N) / max(latency_s, 1e-9)

        # Perplexity (next-token)
        logits = outputs["logits"]  # [B, N, V]
        labels = input_ids[:, 1:]
        logits_for_loss = logits[:, :-1, :]
        loss = nn.CrossEntropyLoss()(
            logits_for_loss.reshape(-1, logits_for_loss.size(-1)), labels.reshape(-1)
        )
        ppl = float(torch.exp(loss).detach().cpu().item())

        return latency_s, tok_per_s, delta_mem_gb, peak_gb, ppl

    # -------------------------- Sequence-Length Benchmark --------------------------

    def _bench_length(
        self, seq_len: int, autocast_dtype: Optional[torch.dtype]
    ) -> Dict[str, float]:
        self._log(f"\n🔍 Sequence length: {seq_len:,} (B={self.cfg.batch_size})")
        # Warmups (not recorded)
        for w in range(self.cfg.warmup_runs):
            try:
                xw = self._make_batch(
                    min(seq_len, 2048)
                )  # cap warmup length to keep quick
                _ = self._forward_once(xw, autocast_dtype)
            except Exception:
                break  # ignore warmup failures

        latencies, throughputs, mem_deltas, mem_peaks, ppls = [], [], [], [], []
        fails_oom = 0
        fails_other = 0

        for sidx in range(self.cfg.num_samples_per_length):
            try:
                x = self._make_batch(seq_len)
                t0 = time.time()
                latency_s, tps, dmem, pmem, ppl = self._forward_once(x, autocast_dtype)
                if time.time() - t0 > self.cfg.timeout_seconds:
                    self._log(
                        f"  ⚠️ Sample {sidx + 1}: soft timeout ({self.cfg.timeout_seconds}s) exceeded"
                    )
                latencies.append(latency_s)
                throughputs.append(tps)
                mem_deltas.append(dmem)
                mem_peaks.append(pmem)
                ppls.append(ppl)
                self._log(
                    f"  {sidx + 1:02d}: {tps:9.1f} tok/s | {latency_s:7.4f} s | Δmem {dmem:5.2f} GB | "
                    f"peak {pmem:5.2f} GB | ppl {ppl:7.2f}"
                )

                if pmem > self.cfg.max_memory_gb:
                    self._log(
                        f"  ⚠️ Peak memory {pmem:.2f} GB > cap {self.cfg.max_memory_gb:.2f} GB, early-exit length"
                    )
                    break

            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" in msg or "cuda error" in msg:
                    fails_oom += 1
                    self._log(
                        f"  ❌ Sample {sidx + 1}: OOM ({fails_oom}/{self.cfg.fail_fast_oom})"
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    if fails_oom >= self.cfg.fail_fast_oom:
                        self._log("  🛑 Too many OOMs; aborting this sequence length")
                        break
                    continue
                fails_other += 1
                self._log(
                    f"  ❌ Sample {sidx + 1}: runtime error ({fails_other}/{self.cfg.fail_fast_exceptions}) -> {e}"
                )
                if fails_other >= self.cfg.fail_fast_exceptions:
                    self._log("  🛑 Too many failures; aborting this sequence length")
                    break
            except Exception as e:
                fails_other += 1
                self._log(
                    f"  ❌ Sample {sidx + 1}: exception ({fails_other}/{self.cfg.fail_fast_exceptions}) -> {e}"
                )
                if fails_other >= self.cfg.fail_fast_exceptions:
                    self._log("  🛑 Too many failures; aborting this sequence length")
                    break

        succ = len(latencies)
        total = succ + fails_oom + fails_other
        if succ == 0:
            return {
                "success_rate": 0.0,
                "lat": float("inf"),
                "tps": 0.0,
                "mem_delta": float("inf"),
                "mem_peak": float("inf"),
                "ppl": float("inf"),
            }

        return {
            "success_rate": succ / max(total, 1),
            "lat": float(np.mean(latencies)),
            "tps": float(np.mean(throughputs)),
            "mem_delta": float(np.mean(mem_deltas)),
            "mem_peak": float(np.max(mem_peaks)),  # report worst-case peak for safety
            "ppl": float(np.mean(ppls)),
        }

    # -------------------------- Full Benchmark --------------------------

    def run_benchmark(self, output_dir: str = "data/results/quick_benchmark"):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self._log(f"\n🎯 Long-Arena Benchmark start @ {_now()}")
        self._log(f"Output dir: {Path(output_dir).resolve().as_posix()}")

        # Autocast dtype resolution
        ac_dtype = self.cfg.resolve_autocast_dtype()
        if self.cfg.use_autocast and ac_dtype is None:
            self._log(
                "AMP requested with dtype='none' or unsupported; proceeding without autocast."
            )
        elif self.cfg.use_autocast:
            self._log(f"Autocast enabled with dtype={str(ac_dtype).split('.')[-1]}")

        # Smoke mode
        if self.cfg.smoke:
            self._log("\n🚦 Smoke mode: 512 & 1024 tokens, 1 sample each")
            tmp = (512, 1024)
            for n in tmp:
                r = self._bench_length(n, ac_dtype)
                self._log(
                    f"SMOKE {n}: tps={r['tps']:.1f}, lat={r['lat']:.4f}s, "
                    f"Δmem={r['mem_delta']:.2f}GB, peak={r['mem_peak']:.2f}GB, ppl={r['ppl']:.2f}, "
                    f"succ={r['success_rate']:.0%}"
                )
            self._persist(output_dir)  # writes empty/full as available
            self._plot(output_dir)
            self._log("\n✅ Smoke complete.")
            self._fh.close()
            return

        # Main loop
        prev_sr = 1.0
        for n in self.cfg.sequence_lengths:
            # heuristic: if previous success rate < 0.3, skip larger sizes
            if n > min(self.cfg.sequence_lengths) and prev_sr < 0.3:
                self._log(f"⚠️ Skipping {n:,} due to low prior success ({prev_sr:.0%})")
                continue
            r = self._bench_length(n, ac_dtype)
            prev_sr = r["success_rate"]

            self.results["sequence_lengths"].append(n)
            self.results["latency_s"].append(r["lat"])
            self.results["throughput_tok_s"].append(r["tps"])
            self.results["mem_used_gb"].append(r["mem_delta"])
            self.results["mem_peak_gb"].append(r["mem_peak"])
            self.results["perplexity"].append(r["ppl"])
            self.results["success_rate"].append(prev_sr)

            self._log(
                f"📊 {n:,}: succ={prev_sr:.0%}, tps={r['tps']:.0f}, "
                f"lat={r['lat']:.4f}s, Δmem={r['mem_delta']:.2f}GB, peak={r['mem_peak']:.2f}GB, ppl={r['ppl']:.2f}"
            )

        # Analyze scaling & persist
        self._analyze_scaling()
        self._persist(output_dir)
        self._plot(output_dir)

        self._log("\n🏁 Benchmark complete.")
        if self.scaling_analysis:
            sa = self.scaling_analysis
            self._log(f"  Max N: {sa.get('max_seq_len', 0):,}")
            self._log(f"  Peak TPS: {sa.get('peak_tps', 0):.0f}")
            self._log(
                f"  Time complexity exponent α ≈ {sa.get('time_alpha', float('nan')):.3f}"
            )
            self._log(
                f"  Memory complexity exponent β ≈ {sa.get('mem_beta', float('nan')):.3f}"
            )
        self._fh.close()

    # -------------------------- Analysis & Persist --------------------------

    def _analyze_scaling(self):
        seq = np.array(self.results["sequence_lengths"], dtype=np.float64)
        tps = np.array(self.results["throughput_tok_s"], dtype=np.float64)
        mem = np.array(self.results["mem_peak_gb"], dtype=np.float64)
        lat = np.array(self.results["latency_s"], dtype=np.float64)

        mask = (
            (seq > 0)
            & np.isfinite(tps)
            & (tps > 0)
            & np.isfinite(mem)
            & (mem > 0)
            & np.isfinite(lat)
            & (lat > 0)
        )
        seq, tps, mem, lat = seq[mask], tps[mask], mem[mask], lat[mask]
        if len(seq) < 3:
            self._log("⚠️ Not enough valid points for scaling law fit.")
            self.scaling_analysis = {}
            return

        logn = np.log(seq)
        # latency ~ n^α  => log(lat) ~ α*log(n) + c
        a_time, b_time = np.polyfit(logn, np.log(lat), 1)
        # memory ~ n^β   => log(mem) ~ β*log(n) + c
        a_mem, b_mem = np.polyfit(logn, np.log(mem), 1)
        # tps ~ n^γ      => log(tps) ~ γ*log(n) + c
        a_tps, b_tps = np.polyfit(logn, np.log(tps), 1)

        r2_time = np.corrcoef(logn, np.log(lat))[0, 1] ** 2
        r2_mem = np.corrcoef(logn, np.log(mem))[0, 1] ** 2
        r2_tps = np.corrcoef(logn, np.log(tps))[0, 1] ** 2

        self.scaling_analysis = {
            "time_alpha": float(a_time),
            "time_r2": float(r2_time),
            "mem_beta": float(a_mem),
            "mem_r2": float(r2_mem),
            "tps_gamma": float(a_tps),
            "tps_r2": float(r2_tps),
            "max_seq_len": int(seq.max()),
            "peak_tps": float(tps.max()),
            "min_mem_per_token_gb": float(np.min(mem / seq)),
        }
        self._log("\n📈 Scaling fit:")
        self._log(f"  latency ~ n^{a_time:.3f} (R²={r2_time:.3f})")
        self._log(f"   memory ~ n^{a_mem:.3f} (R²={r2_mem:.3f})")
        self._log(f" throughput ~ n^{a_tps:.3f} (R²={r2_tps:.3f})")

    def _persist(self, output_dir: str):
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        model_params = sum(p.numel() for p in self.model.parameters())
        summary = {
            "timestamp": _now(),
            "device": _dev_str(),
            "amp": {"enabled": self.cfg.use_autocast, "dtype": self.cfg.autocast_dtype},
            "model": {"parameters": model_params, "vocab_size": self.cfg.vocab_size},
            "config": {
                "sequence_lengths": self.cfg.sequence_lengths,
                "num_samples_per_length": self.cfg.num_samples_per_length,
                "batch_size": self.cfg.batch_size,
                "max_memory_gb": self.cfg.max_memory_gb,
                "timeout_seconds": self.cfg.timeout_seconds,
            },
            "results": self.results,
            "scaling": self.scaling_analysis,
        }

        # Write full results & summary to output_dir
        (out_dir / "long_arena_results.json").write_text(json.dumps(summary, indent=2))
        (out_dir / "benchmark_summary.json").write_text(
            json.dumps(
                {
                    "timestamp": summary["timestamp"],
                    "device": summary["device"],
                    "model_parameters": model_params,
                    "max_seq_len": (
                        max(self.results["sequence_lengths"])
                        if self.results["sequence_lengths"]
                        else 0
                    ),
                    "peak_throughput_tok_s": (
                        max(self.results["throughput_tok_s"])
                        if self.results["throughput_tok_s"]
                        else 0.0
                    ),
                    "scaling": self.scaling_analysis,
                },
                indent=2,
            )
        )

        # Also write a consolidator-friendly copy
        Path("data/results").mkdir(parents=True, exist_ok=True)
        Path("data/results/benchmark_summary.json").write_text(
            (out_dir / "benchmark_summary.json").read_text()
        )
        self._log(
            f"💾 Saved results to: {out_dir.as_posix()} and data/results/benchmark_summary.json"
        )

    def _plot(self, output_dir: str):
        if not self.results["sequence_lengths"]:
            self._log("No results to plot.")
            return
        out_dir = Path(output_dir)
        seq = self.results["sequence_lengths"]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        # Throughput
        axes[0, 0].loglog(seq, self.results["throughput_tok_s"], "o-")
        axes[0, 0].set_title("Throughput vs Sequence Length")
        axes[0, 0].set_xlabel("Tokens")
        axes[0, 0].set_ylabel("Tokens / sec")
        axes[0, 0].grid(True, which="both")

        # Latency
        axes[0, 1].loglog(seq, self.results["latency_s"], "o-")
        axes[0, 1].set_title("Latency vs Sequence Length")
        axes[0, 1].set_xlabel("Tokens")
        axes[0, 1].set_ylabel("Seconds")
        axes[0, 1].grid(True, which="both")

        # Memory (peak)
        axes[1, 0].loglog(seq, self.results["mem_peak_gb"], "o-")
        axes[1, 0].set_title("Peak Memory vs Sequence Length")
        axes[1, 0].set_xlabel("Tokens")
        axes[1, 0].set_ylabel("GB")
        axes[1, 0].grid(True, which="both")

        # Success rate
        axes[1, 1].semilogx(
            seq, [100.0 * s for s in self.results["success_rate"]], "o-"
        )
        axes[1, 1].set_title("Success Rate")
        axes[1, 1].set_xlabel("Tokens")
        axes[1, 1].set_ylabel("Success (%)")
        axes[1, 1].set_ylim(0, 105)
        axes[1, 1].grid(True, which="both")

        plt.tight_layout()
        out_path = out_dir / "long_arena_benchmark.png"
        plt.savefig(out_path.as_posix(), dpi=300, bbox_inches="tight")
        plt.close()
        self._log(f"🖼️ Plot saved: {out_path.as_posix()}")


# ------------------------------ CLI (optional) ------------------------------


def _default_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        model_path="",  # random init by default
        sequence_lengths=[1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
        num_samples_per_length=5,
        batch_size=1,
        max_memory_gb=12.0,
        timeout_seconds=300,
        use_autocast=True,
        autocast_dtype="bf16",
        smoke=False,
        profile_per_sample=False,
        log_file="data/results/quick_benchmark/benchmark.log",
        profiles_dir="data/results/quick_benchmark/profiles",
    )


if __name__ == "__main__":
    # Minimal CLI for ad-hoc local runs; the pipeline imports the classes directly.
    cfg = _default_config()

    # Make output directory default to data/results/quick_benchmark for convenience.
    out_dir = "data/results/quick_benchmark"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    bench = LongArenaBenchmark(cfg)
    bench.run_benchmark(out_dir)
