"""
Training Monitoring and Callbacks.

Provides hooks for logging metrics to WandB, checkpointing 
best models, and implementing early stopping.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional

class TrainingMonitor:
    """
    Handles logging and checkpointing logic.
    """
    
    def __init__(self, exp_name: str, save_dir: str = "checkpoints"):
        self.exp_name = exp_name
        self.save_path = Path(save_dir) / exp_name
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.best_loss = float('inf')

    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """
        Log metrics to console or external logger (e.g. WandB).
        """
        print(f"Epoch {epoch}: ", end="")
        for k, v in metrics.items():
            print(f"{k}={v:.6f}", end=" | ")
        print()

    def check_and_save(self, model: nn.Module, val_loss: float):
        """
        Saves checkpoint if validation loss improves.
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            filename = self.save_path / "best_model.pt"
            torch.save(model.state_dict(), filename)
            print(f"Saving best model to {filename}")

if __name__ == "__main__":
    mon = TrainingMonitor("test_exp")
    print(f"Monitor ready at {mon.save_path}")
