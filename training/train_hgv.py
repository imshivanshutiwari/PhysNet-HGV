"""
Main HGV Training Script.

Orchestrates the data loading, model initialization, and training 
for PINN and Transformers on the trajectory dataset.
"""

import os
import torch
import torch.optim as optim
from preprocessing.data_pipeline import get_hgv_dataloader
from models.pinn_module import PINNModule
from training.trainer import HGVTrainer
from training.losses import PhysicsLoss
from training.callbacks import TrainingMonitor
from utils.config_loader import load_config

def train_pinn(config_path: str):
    """
    Entry point for PINN training.
    """
    # 1. Load Config
    cfg = load_config(config_path)
    
    # 2. Data
    data_dir = "data/trajectories"
    train_loader = get_hgv_dataloader(data_dir, batch_size=32, window_size=1)
    
    # 3. Model
    model = PINNModule(input_dim=1, hidden_dim=256)
    
    # 4. Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()
    
    # 5. Trainer (Custom loop for PINN)
    monitor = TrainingMonitor("pinn_hgv_exp")
    
    n_epochs = 2 
    print(f"Starting PINN training for {n_epochs} epochs (Demo Mode)...")
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            t = batch["time"].reshape(-1, 1)
            target = torch.cat([batch["state"].reshape(-1, 12), batch["electron_density"].reshape(-1, 1)], dim=1)
            
            optimizer.zero_grad()
            pred = model(t)
            loss = loss_fn(pred, target)
            
            # Add physics loss
            p_losses = model.physics_loss(model, t)
            total_loss = loss + 0.1 * sum(p_losses.values())
            
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        monitor.log_metrics(epoch, {"train_loss": avg_loss})
    
    print("Training demo completed successfully.")

if __name__ == "__main__":
    # In a full run, we'd pass the actual config path
    cfg_path = "configs/pinn_config.yaml"
    if os.path.exists(cfg_path):
        train_pinn(cfg_path)
    else:
        print("Config not found. Skipping main run.")
