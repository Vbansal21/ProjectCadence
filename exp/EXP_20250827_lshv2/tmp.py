import sys
import os
import json

sys.path.append("src")
sys.path.append("data")
sys.path.append("scripts")
from scripts.train_lsh_v2 import TrainingConfig, LSHv2Trainer
import torch
import torch.utils.checkpoint as ckp

ckp.set_checkpoint_debug_enabled(True)
torch.set_float32_matmul_precision("high")  # stable on CPU
torch.use_deterministic_algorithms(True)  # optional; may restrict some kernels

config = TrainingConfig(
    model_size="100M",
    seq_length=4096,
    batch_size=1,
    num_epochs=1,
    num_train_sequences=10,
    num_eval_sequences=5,
    learning_rate=1e-4,
    profile_components=True,
    log_every=1,
    eval_every=10,
    save_every=20,
)

output_dir = "data/results/quick_training"
os.makedirs(output_dir, exist_ok=True)

# stash config for reproducibility
try:
    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump(
            config.__dict__ if hasattr(config, "__dict__") else dict(config),
            f,
            indent=4,
        )
except Exception:
    pass

trainer = LSHv2Trainer(config, output_dir)
trainer.generate_data()
trainer.train()
print("🎉 Quick training complete!")
