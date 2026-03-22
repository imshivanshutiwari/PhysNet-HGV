"""
PhysNet-HGV Model Trainer.

Generic trainer module providing training and validation loops 
for the various neural architectures in the project.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Optional

class HGVTrainer:
    """
    Unified trainer for HGV tracking models.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.history = {"train_loss": [], "val_loss": []}

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch in tqdm(dataloader, desc="Training"):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward
            # Assuming batch has 'state' as input and 'next_state' as target
            pred = self.model(batch["state"])
            loss = self.loss_fn(pred, batch["state"]) # Simplified placeholder
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        self.history["train_loss"].append(avg_loss)
        return avg_loss

    def validate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                pred = self.model(batch["state"])
                loss = self.loss_fn(pred, batch["state"])
                total_loss += loss.item()
                
        avg_loss = total_loss / len(dataloader)
        self.history["val_loss"].append(avg_loss)
        return avg_loss

if __name__ == "__main__":
    print("HGVTrainer module loaded.")
    # Test (Mock)
    m = nn.Linear(10, 10)
    opt = optim.Adam(m.parameters())
    lf = nn.MSELoss()
    trainer = HGVTrainer(m, opt, lf)
    print("Trainer instantiation OK.")
