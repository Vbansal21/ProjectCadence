"""
LSH v2 Training Script with Granular Performance Analysis
========================================================

Training script for LSH v2 model with:
- Synthetic data generation
- Granular component profiling during training
- Memory and compute optimization
- Long sequence training (up to 131K tokens)
- Adaptive learning rate scheduling

Author: Research Implementation
Version: 2.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import json
import os
import sys
from typing import Dict, List
import matplotlib.pyplot as plt
from dataclasses import dataclass
import psutil
import gc
import matplotlib

# stdlib
from pathlib import Path
import importlib.util

matplotlib.use("Agg")

import torch.utils.checkpoint as ckp

ckp.set_checkpoint_debug_enabled(True)

torch.set_float32_matmul_precision("high")  # stable on CPU
torch.use_deterministic_algorithms(True)  # optional; may restrict some kernels

# Add src to path
# --- replace these lines ---
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))

# --- with this: ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

sys.path.insert(0, SRC_DIR)
sys.path.insert(0, DATA_DIR)
# --------------------------------

# Resolve project roots
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA = ROOT / "data"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load project modules without late import statements (keeps Ruff E402 happy)
_lsh = _load_module("lsh_v2_model", SRC / "lsh_v2_model.py")
create_lsh_v2_model = _lsh.create_lsh_v2_model

_synth = _load_module("synthetic_data_generator", DATA / "synthetic_data_generator.py")
SyntheticDataGenerator = _synth.SyntheticDataGenerator


@dataclass
class TrainingConfig:
    """Training configuration"""

    model_size: str = "100M"
    vocab_size: int = 32000
    seq_length: int = 131072
    batch_size: int = 1  # Large sequences require small batch size
    num_epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    save_every: int = 1000
    eval_every: int = 500
    log_every: int = 100

    # Data generation
    num_train_sequences: int = 1000
    num_eval_sequences: int = 100

    # Profiling
    profile_components: bool = True
    profile_memory: bool = True
    save_profiles: bool = True

    # Mixed tasks
    use_amp: bool = True
    amp_dtype: str = "bf16"  # "bf16" if supported, else "fp16"


class SyntheticDataset(Dataset):
    """Dataset for synthetic training data"""

    def __init__(self, sequences: List[Dict]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        tokens = torch.tensor(sequence["tokens"], dtype=torch.long)

        # Create input and target (shifted by 1)
        input_ids = tokens[:-1]
        labels = tokens[1:]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "task_info": sequence.get("task_info", []),
        }


class ComponentProfiler:
    """Profiler for LSH v2 components during training"""

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.profiles = []
        self.hooks = []
        self.current_profile = {}
        self.start_time = None

        # Component timing
        self.component_times = {
            "global_context": [],
            "mla": [],
            "hashing": [],
            "bucket_attention": [],
            "hierarchical_compression": [],
            "long_term_memory": [],
            "output_processing": [],
            "total": [],
        }

        # Memory tracking
        self.memory_usage = []

        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks for coarse per-module timing."""
        wanted = {
            "global_enc": "global_context",
            "gqa": "gqa",
            "hasher": "hashing",
            "spectral": "spectral",
            "mem": "memory_tokens",
        }

        def make_hook(public_name):
            def hook(module, input, output):
                if self.start_time is not None:
                    now = time.perf_counter()
                    elapsed = now - self.start_time
                    self.current_profile[public_name] = elapsed
                    self.start_time = now

            return hook

        for name, module in self.model.named_modules():
            for subname, public in wanted.items():
                if name.endswith(subname):
                    self.hooks.append(module.register_forward_hook(make_hook(public)))

    def start_profiling(self):
        """Start profiling a forward pass"""
        self.current_profile = {}
        self.start_time = time.time()

        # Memory before
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated()
        else:
            memory_before = psutil.Process().memory_info().rss

        self.current_profile["memory_before"] = memory_before

    def end_profiling(self, seq_len: int):
        """End profiling and save results"""
        total_time = time.time() - self.start_time if self.start_time else 0
        self.current_profile["total"] = total_time
        self.current_profile["seq_len"] = seq_len

        # Memory after
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_after = torch.cuda.memory_allocated()
        else:
            memory_after = psutil.Process().memory_info().rss

        self.current_profile["memory_after"] = memory_after
        self.current_profile["memory_used"] = (
            memory_after - self.current_profile["memory_before"]
        )

        # Store profile
        self.profiles.append(self.current_profile.copy())

        # Update component times
        for component in self.component_times:
            if component in self.current_profile:
                self.component_times[component].append(self.current_profile[component])

        self.memory_usage.append(self.current_profile["memory_used"])
        self.start_time = None

    def get_average_profile(self) -> Dict:
        """Get average profiling results"""
        if not self.profiles:
            return {}

        avg_profile = {}
        for key in self.profiles[0]:
            if isinstance(self.profiles[0][key], (int, float)):
                avg_profile[key] = np.mean([p[key] for p in self.profiles])

        return avg_profile

    def save_profiles(self, filepath: str):
        """Save profiling results"""
        profile_data = {
            "profiles": self.profiles,
            "component_times": self.component_times,
            "memory_usage": self.memory_usage,
            "average_profile": self.get_average_profile(),
        }

        with open(filepath, "w") as f:
            json.dump(profile_data, f, indent=2)

    def cleanup(self):
        """Remove hooks"""
        for hook in self.hooks:
            hook.remove()


class WarmupCosine:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0, max_lr=None):
        self.optimizer = optimizer
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.min_lr = float(min_lr)
        if max_lr is None:
            # initial LR of the first param group as max
            self.max_lr = float(optimizer.param_groups[0]["lr"])
        else:
            self.max_lr = float(max_lr)
        self.t = 0  # current step

    def _lr_at(self, t: int) -> float:
        if t < self.warmup_steps:
            # linear warmup to max_lr
            return self.min_lr + (self.max_lr - self.min_lr) * (
                t / max(1, self.warmup_steps)
            )
        # cosine decay from max_lr to min_lr
        progress = (t - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(max(progress, 0.0), 1.0)
        import math

        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )

    def get_last_lr(self) -> List[float]:
        return [self._lr_at(self.t) for _ in self.optimizer.param_groups]

    def step(self):
        lr = self._lr_at(self.t)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        self.t += 1

    # >>> add these two <<<
    def state_dict(self):
        return {
            "t": self.t,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
            "max_lr": self.max_lr,
        }

    def load_state_dict(self, state):
        self.t = int(state.get("t", 0))
        self.warmup_steps = int(state.get("warmup_steps", self.warmup_steps))
        self.total_steps = int(state.get("total_steps", self.total_steps))
        self.min_lr = float(state.get("min_lr", self.min_lr))
        self.max_lr = float(state.get("max_lr", self.max_lr))
        # re-apply current LR consistent with restored step
        lr = self._lr_at(self.t)
        for g in self.optimizer.param_groups:
            g["lr"] = lr


class LSHv2Trainer:
    """Trainer for LSH v2 model"""

    def __init__(self, config: TrainingConfig, output_dir: str):
        self.config = config
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Mixed precision
        self.use_amp = self.config.use_amp and (self.device.type == "cuda")
        self.amp_dtype = (
            torch.bfloat16
            if (
                self.config.amp_dtype.lower() == "bf16"
                and torch.cuda.is_bf16_supported()
            )
            else torch.float16
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_amp and self.amp_dtype == torch.float16
        )

        # Optional perf flags
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        # Save config
        with open(os.path.join(output_dir, "training_config.json"), "w") as f:
            json.dump(config.__dict__, f, indent=4)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize model
        print(f"Creating LSH v2 model ({config.model_size})...")
        self.model = create_lsh_v2_model(config.vocab_size, config.model_size)
        self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        total_steps = self.config.num_epochs * (
            self.config.num_train_sequences // self.config.batch_size
        )
        self.scheduler = WarmupCosine(
            self.optimizer,
            warmup_steps=self.config.warmup_steps,
            total_steps=total_steps,
            min_lr=0.1 * self.config.learning_rate,
        )

        # Profiler
        if config.profile_components:
            self.profiler = ComponentProfiler(self.model, self.device)
        else:
            self.profiler = None

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float("inf")

        # Metrics
        self.train_losses = []
        self.eval_losses = []
        self.learning_rates = []

        print(f"✅ Trainer initialized on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def generate_data(self):
        """Generate training and evaluation data"""
        print("Generating synthetic training data...")

        generator = SyntheticDataGenerator(
            vocab_size=self.config.vocab_size, max_seq_len=self.config.seq_length
        )

        # Generate training data
        train_sequences = []
        for i in range(self.config.num_train_sequences):
            if i % 100 == 0:
                print(
                    f"Generated {i}/{self.config.num_train_sequences} training sequences"
                )

            sequence = generator.generate_mixed_task_sequence(self.config.seq_length)
            train_sequences.append(sequence)

        # Generate evaluation data
        eval_sequences = []
        for i in range(self.config.num_eval_sequences):
            if i % 50 == 0:
                print(f"Generated {i}/{self.config.num_eval_sequences} eval sequences")

            sequence = generator.generate_mixed_task_sequence(self.config.seq_length)
            eval_sequences.append(sequence)

        # Create datasets
        self.train_dataset = SyntheticDataset(train_sequences)
        self.eval_dataset = SyntheticDataset(eval_sequences)

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=False,
        )
        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=False,
        )

        print("✅ Data generation complete:")
        print(f"  Training sequences: {len(train_sequences)}")
        print(f"  Evaluation sequences: {len(eval_sequences)}")
        print(f"  Sequence length: {self.config.seq_length}")

    def train_step(self, batch: Dict) -> Dict:
        self.model.train()
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)

        if self.profiler:
            self.profiler.start_profiling()

        # Forward (AMP)
        with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
            step_seed = int(torch.empty((), dtype=torch.int64).random_().item())
            outputs = self.model(input_ids, rng_seed=step_seed)  # dict with 'logits'
            logits = outputs["logits"]  # [B,N,V]
            loss = nn.CrossEntropyLoss()(
                logits.reshape(-1, logits.size(-1)),  # safer than view
                labels.reshape(-1),
            )

        if self.profiler:
            self.profiler.end_profiling(input_ids.size(1))

        self.optimizer.zero_grad(set_to_none=True)

        # Backward (scaled if fp16)
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip
            )
            self.optimizer.step()

        self.scheduler.step()

        self.step += 1
        self.train_losses.append(loss.item())
        self.learning_rates.append(self.scheduler.get_last_lr()[0])

        return {
            "loss": loss.item(),
            "lr": self.scheduler.get_last_lr()[0],
            "step": self.step,
        }

    def evaluate(self) -> Dict:
        """Evaluate model"""
        self.model.eval()
        eval_losses = []

        with torch.no_grad():
            for batch in self.eval_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                step_seed = int(torch.empty((), dtype=torch.int64).random_().item())
                outputs = self.model(input_ids, rng_seed=step_seed)
                logits = outputs["logits"]

                loss = nn.CrossEntropyLoss()(
                    logits.view(-1, logits.size(-1)), labels.view(-1)
                )

                eval_losses.append(loss.item())

        avg_eval_loss = np.mean(eval_losses)
        self.eval_losses.append(avg_eval_loss)

        return {"eval_loss": avg_eval_loss}

    def _maybe_state_dict(obj):
        fn = getattr(obj, "state_dict", None)
        return fn() if callable(fn) else None

    def save_checkpoint(self, filepath: str):
        """Save training checkpoint"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "config": self.config,
            "train_losses": self.train_losses,
            "eval_losses": self.eval_losses,
            "learning_rates": self.learning_rates,
        }

        torch.save(checkpoint, filepath)

    def load_checkpoint(self, path: str, strict: bool = True):
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model_state_dict"], strict=strict)
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # Restore scheduler only if both sides support it
        sched_state = ckpt.get("scheduler_state_dict", None)
        if sched_state is not None:
            ld = getattr(self.scheduler, "load_state_dict", None)
            if callable(ld):
                self.scheduler.load_state_dict(sched_state)
        if ckpt.get("scaler_state_dict") and hasattr(self, "scaler"):
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        self.epoch = ckpt.get("epoch", 0)
        self.best_loss = ckpt.get("best_loss", float("inf"))
        self.train_losses = ckpt.get("train_losses", [])

    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config.num_epochs} epochs...")

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            epoch_losses = []

            for batch_idx, batch in enumerate(self.train_loader):
                # Training step
                step_results = self.train_step(batch)
                epoch_losses.append(step_results["loss"])

                # Logging
                if self.step % self.config.log_every == 0:
                    print(
                        f"Step {self.step}: Loss = {step_results['loss']:.4f}, LR = {step_results['lr']:.2e}"
                    )

                # Evaluation
                if self.step % self.config.eval_every == 0:
                    eval_results = self.evaluate()
                    print(f"Evaluation: Loss = {eval_results['eval_loss']:.4f}")

                    # Save best model
                    if eval_results["eval_loss"] < self.best_loss:
                        self.best_loss = eval_results["eval_loss"]
                        self.save_checkpoint(
                            os.path.join(self.output_dir, "best_model.pt")
                        )

                # Save checkpoint
                if self.step % self.config.save_every == 0:
                    self.save_checkpoint(
                        os.path.join(self.output_dir, f"checkpoint_step_{self.step}.pt")
                    )

                # Memory cleanup
                if self.step % 100 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # End of epoch
            avg_epoch_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch + 1} complete: Average Loss = {avg_epoch_loss:.4f}")

        print("Training complete!")

        # Save final results
        self.save_final_results()

    def save_final_results(self):
        """Save final training results and analysis"""
        # Save final checkpoint
        self.save_checkpoint(os.path.join(self.output_dir, "final_model.pt"))

        # Save profiling results
        if self.profiler:
            self.profiler.save_profiles(
                os.path.join(self.output_dir, "component_profiles.json")
            )
            self.profiler.cleanup()

        # Save training metrics
        metrics = {
            "train_losses": self.train_losses,
            "eval_losses": self.eval_losses,
            "learning_rates": self.learning_rates,
            "final_step": self.step,
            "best_loss": self.best_loss,
        }

        with open(os.path.join(self.output_dir, "training_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # Create training plots
        self.create_training_plots()

        print(f"✅ Results saved to {self.output_dir}")

    def create_training_plots(self):
        """Create training visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Training loss
        axes[0, 0].plot(self.train_losses)
        axes[0, 0].set_title("Training Loss")
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].grid(True)

        # Evaluation loss
        if self.eval_losses:
            eval_steps = np.arange(0, len(self.eval_losses)) * self.config.eval_every
            axes[0, 1].plot(eval_steps, self.eval_losses)
            axes[0, 1].set_title("Evaluation Loss")
            axes[0, 1].set_xlabel("Step")
            axes[0, 1].set_ylabel("Loss")
            axes[0, 1].grid(True)

        # Learning rate
        axes[1, 0].plot(self.learning_rates)
        axes[1, 0].set_title("Learning Rate")
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("Learning Rate")
        axes[1, 0].grid(True)

        # Component profiling (if available)
        if self.profiler and self.profiler.profiles:
            component_names = list(self.profiler.component_times.keys())
            avg_times = [
                np.mean(times) if times else 0
                for times in self.profiler.component_times.values()
            ]

            axes[1, 1].bar(component_names, avg_times)
            axes[1, 1].set_title("Average Component Times")
            axes[1, 1].set_xlabel("Component")
            axes[1, 1].set_ylabel("Time (seconds)")
            axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "training_analysis.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def main():
    """Main training function"""
    # Configuration
    config = TrainingConfig(
        model_size="100M",
        seq_length=131072,
        batch_size=1,
        num_epochs=3,
        num_train_sequences=500,  # Reduced for testing
        num_eval_sequences=50,
        learning_rate=1e-4,
        profile_components=True,
    )

    # Output directory
    output_dir = RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Create trainer
    trainer = LSHv2Trainer(config, output_dir)

    # Generate data
    trainer.generate_data()

    # Train model
    trainer.train()

    print("🎉 Training complete! Check results in:", output_dir)


if __name__ == "__main__":
    main()
