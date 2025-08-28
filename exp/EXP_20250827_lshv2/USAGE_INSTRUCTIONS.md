# LSH v2 – Usage & Operations Guide

*Project root `/ProjectCadence/exp/current`*

---

## 🚀 Quick-Start

### 1 · Full Orchestrator _(preferred)_

```bash
# Always start in the project root
cd /ProjectCadence/exp/current

# Sanity-check, tiny dataset, short seq-len
python run_complete_pipeline.py --quick

# End-to-end pipeline (hours on a single GPU)
python run_complete_pipeline.py

# Smoke-test only (model + data)
python run_complete_pipeline.py --test-only
```

> **Tip — resume a failed run**
> Each phase writes an “OK” marker; rerunning will skip completed phases unless you delete the marker or pass `--force`.

### 2 · Component-level Invocation


| Task                  | Command                                                                                                                                                                                                                     | Notes                                                   |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| **Model sanity test** | `python - <<'PY'\nimport sys, torch; sys.path.append('src'); from lsh_v2_model import create_lsh_v2_model\nm=create_lsh_v2_model(vocab_size=32000,target_params='100M')\nprint(sum(p.numel() for p in m.parameters()))\nPY` | Creates a 100 M-param model and prints total parameters |
| **Synthetic data**    | `python data/synthetic_data_generator.py`                                                                                                                                                                                   | Generates/parses`.jsonl` into `data/results/`           |
| **Training**          | `python scripts/train_lsh_v2.py`                                                                                                                                                                                            | Uses`TrainingConfig` inside the script                  |
| **Benchmark**         | `python benchmarks/long_arena_benchmark.py`                                                                                                                                                                                 | Evaluates latency & memory scaling                      |

---

## ⚙️ CLI Reference — `run_complete_pipeline.py`

```text
--quick           Short sequences + tiny datasets
--no-training     Skip Phase 2 (training)
--no-benchmark    Skip Phase 3 (Long-Arena)
--test-only       Run Phases 1 (Model + Data) only
-h, --help        Show full help
```

**Examples**

```bash
# Benchmark a pre-trained checkpoint, no further training
python run_complete_pipeline.py --no-training

# Debug GPU OOM quickly
CUDA_VISIBLE_DEVICES="" python run_complete_pipeline.py --quick
```

---

## 🛠️ Configuration Keys

### Model (`src/lsh_v2_model.py`)

```python
config = dict(
    vocab_size      = 32_000,   # BPE vocab
    d_model         = 768,
    n_layers        = 24,
    n_heads         = 12,
    n_kv_heads      = 4,        # grouped-query attn
    max_seq_len     = 131_072,
    bucket_size     = 128,      # LSH bucket granularity
    use_ponder      = True,     # PonderNet adaptive depth
    weight_tied_layers = 6      # ↓ memory via layer tying
)
```

### Training (`scripts/train_lsh_v2.py`)

```python
config = TrainingConfig(
    model_size          = "100M",
    seq_length          = 131_072,
    batch_size          = 1,
    num_epochs          = 5,
    learning_rate       = 1e-4,
    num_train_sequences = 1_000,
    num_eval_sequences  = 100,
    gradient_checkpoint = True,    # RAM-saving
    profile_components  = True
)
```

### Benchmark (`benchmarks/long_arena_benchmark.py`)

```python
config = BenchmarkConfig(
    sequence_lengths       = [1_024, 2_048, 4_096, 8_192,
                              16_384, 32_768, 65_536, 131_072],
    num_samples_per_length = 10,
    max_memory_gb          = 16.0,
    timeout_seconds        = 300
)
```

---

## 📁 Artifact Layout

```
data/
├─ results/
│  ├─ quick_training/           # --quick output
│  ├─ quick_benchmark/
│  ├─ training_metrics.json
│  ├─ component_profiles.json
│  └─ benchmark_summary.json
final_results/
└─ pipeline_results.json        # consolidated manifest
lsh_v2_complete_package.zip
```

*Trained checkpoints* are saved in `data/results/checkpoints/`.
*Plots* (`*.png`) live alongside each metrics file.

---

## 📊 Reading the Numbers


| Metric                 | File                                      | Interpretation                              |
| ------------------------ | ------------------------------------------- | --------------------------------------------- |
| **Loss curves**        | `training_metrics.json` → `"loss_train"` | Expect monotone ↓ until ~epoch 3           |
| **Per-component time** | `component_profiles.json`                 | `hashing_ms`, `bucket_attn_ms`, `global_ms` |
| **Latency scaling**    | `benchmark_summary.json`                  | Columns:`seq_len`, `latency_ms`, `rss_mb`   |
| **Throughput**         | `tokens_per_sec`                          | Should approach**10 k t/s** @ 4 k tokens    |

### Complexity Targets

$$
\text{Time}(n) \sim n^{\alpha},\; \alpha \le 1.2
\quad\;\;
\text{Memory}(n) \sim n^{\beta},\; \beta \approx 1.0

$$

Success flags:

- ✅ `seq_len = 65 536` feasible without OOM
- ✅ `α ≤ 1.2`, `β ≈ 1.0`
- ✅ Throughput > 10 k t/s on A100 80 GB

---

## 🐛 Troubleshooting Cheatsheet


| Symptom                          | Likely Cause       | Quick Fix                                                       |
| ---------------------------------- | -------------------- | ----------------------------------------------------------------- |
| **CUDA OOM** @ >32 k tokens      | insufficient VRAM  | `seq_length=32_768`, `gradient_checkpoint=True`, or move to CPU |
| ImportError`create_lsh_v2_model` | renamed entrypoint | verify function name in`src/lsh_v2_model.py`                    |
| `TimeoutExpired` in benchmark    | ––               | raise`timeout_seconds` in `BenchmarkConfig`                     |
| Sluggish training                | I/O bound data gen | pre-generate dataset, set`num_workers>4`                        |

---

## ⏱️ Wall-Clock Expectations*


| Mode                   | Model + Data | Train      | Benchmark | Total     |
| ------------------------ | -------------- | ------------ | ----------- | ----------- |
| **Quick**              | 2 min        | 15–30 min | 10 min    | ≈ 45 min |
| **Full (single A100)** | 2 min        | 2–6 h     | 40 min    | 3–7 h    |
| **Test-only**          | 2 min        | –         | –        | 2 min     |

<sup>\* Measured on AMD 7950X + A100 80 GB; scale accordingly.</sup>

---

## 🛰️ Advanced Ops

- **Multi-GPU**: wrap `scripts/train_lsh_v2.py` with `torchrun --nproc_per_node=N …`.
- **W&B / MLflow**: call `trainer.enable_wandb(project="lsh_v2")`.
- **Distributed data**: switch generator to `torch.utils.data.IterableDataset` for streaming.

---

## 🎯 Follow-ups

1. Inspect `pipeline_results.json` for high-level health.
2. Tune hyper-parameters → rerun `--quick` before committing hours.
3. Extend `benchmarks/` with custom long-context tasks (e.g., `pg19`, `arxiv-math`).
4. Tag released checkpoints and zip with `package.py` for reproducibility.

---

## 🛠️ Support Flow

1. **Reproduce** with `--quick` and capture full console log.
2. **Validate env** `python -m pip list | grep torch`.
3. **File an issue** in the project tracker; attach `pipeline_results.json`.

> *The pipeline prints traceback excerpts and keeps raw logs in `data/results/*.log` – include them for faster triage.*

---
