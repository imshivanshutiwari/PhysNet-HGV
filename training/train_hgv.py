import os
import torch
import wandb
import numpy as np

from utils.seed import seed_everything
from utils.logger import get_logger
from utils.config_loader import load_all_configs
from simulation.trajectory_gen import TrajectoryGenerator
from preprocessing.data_pipeline import get_dataloaders
from preprocessing.state_normalizer import StateNormalizer
from preprocessing.blackout_labeler import BlackoutLabeler
from models.pinn_module import PINNModule
from models.neural_ode import NeuralODETracker
from models.srgan import Generator, Discriminator, SRGANLoss
from diffusion.ddpm_model import DDPMTrajectory
from training.losses import PINNLoss
from training.trainer import PINNTrainer

logger = get_logger("train_hgv")


def main():
    seed_everything(42)
    configs = load_all_configs()

    hgv_config = configs["hgv"]
    pinn_config = configs["pinn"]
    diffusion_config = configs["diffusion"]

    logger.info("Generating trajectories")
    generator = TrajectoryGenerator()
    trajectories = generator.generate_batch(100)

    logger.info("Preprocessing data")
    normalizer = StateNormalizer(state_dim=6)

    clean_trajectories = [traj["trajectory"] for traj in trajectories]
    normalizer.fit(clean_trajectories)

    labeler = BlackoutLabeler(blackout_threshold=pinn_config["blackout_Ne_threshold"])
    for traj in trajectories:
        traj["blackout_mask"] = labeler.label_trajectory(traj["plasma_profile"])

    n_train = int(len(trajectories) * hgv_config["splits"][0])
    train_data = trajectories[:n_train]
    val_data = trajectories[n_train:]

    train_loader, val_loader = get_dataloaders(
        train_data,
        val_data,
        batch_size=pinn_config["batch_size"],
        seq_len=10,
        normalizer=normalizer,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Starting Phase A: PINN Training")
    pinn_model = PINNModule(
        state_dim=6,
        hidden_dim=pinn_config["hidden_dim"],
        n_layers=pinn_config["n_layers"],
        activation=pinn_config["activation"],
    ).to(device)

    pinn_optimizer = torch.optim.AdamW(pinn_model.parameters(), lr=pinn_config["lr"])
    pinn_criterion = PINNLoss(
        lambda_momentum=pinn_config["lambda_momentum"],
        lambda_continuity=pinn_config["lambda_continuity"],
        lambda_plasma=pinn_config["lambda_plasma"],
    )

    pinn_trainer = PINNTrainer(pinn_model, pinn_optimizer, pinn_criterion, device)

    n_epochs = min(2, pinn_config["n_epochs"])
    for epoch in range(n_epochs):
        train_metrics = pinn_trainer.train_epoch(train_loader)
        val_loss = pinn_trainer.evaluate(val_loader)
        logger.info(
            f"Epoch {epoch+1}/{n_epochs} - Val Loss: {val_loss:.4f} - Train Metrics: {train_metrics}"
        )

    node_tracker = NeuralODETracker()

    logger.info("Starting Phase B: DDPM Training")
    ddpm_model = DDPMTrajectory(
        T=diffusion_config["T_diffusion"],
        beta_start=diffusion_config["beta_start"],
        beta_end=diffusion_config["beta_end"],
        condition_dim=diffusion_config["condition_dim"],
        state_dim=9,
        hidden_dim=diffusion_config["hidden_dim"],
    ).to(device)

    from diffusion.ddpm_trainer import DDPMTrainer

    ddpm_trainer = DDPMTrainer(ddpm_model, lr=diffusion_config["lr"], device=device)

    # Train DDPM
    n_epochs_ddpm = min(2, diffusion_config["n_epochs"])
    for epoch in range(n_epochs_ddpm):
        # Dummy data for DDPM testing
        for _ in range(5):
            x0 = torch.randn(diffusion_config["batch_size"], 9)
            cond = torch.randn(diffusion_config["batch_size"], diffusion_config["condition_dim"])
            loss = ddpm_trainer.train_step(x0, cond)
        logger.info(f"DDPM Epoch {epoch+1}/{n_epochs_ddpm} - Loss: {loss:.4f}")

    logger.info("Starting Phase C: SRGAN Training")
    srgan_generator = Generator().to(device)
    srgan_discriminator = Discriminator().to(device)
    srgan_criterion = SRGANLoss()

    gen_optimizer = torch.optim.Adam(srgan_generator.parameters(), lr=1e-4)
    disc_optimizer = torch.optim.Adam(srgan_discriminator.parameters(), lr=1e-4)

    # Train SRGAN
    n_epochs_srgan = 2
    for epoch in range(n_epochs_srgan):
        for _ in range(5):
            # Dummy images
            low_res = torch.randn(16, 3, 32, 32).to(device)
            high_res = torch.randn(16, 3, 128, 128).to(device)

            # Train Discriminator
            disc_optimizer.zero_grad()
            fake_high_res = srgan_generator(low_res)

            real_pred = srgan_discriminator(high_res)
            fake_pred = srgan_discriminator(fake_high_res.detach())

            real_loss = srgan_criterion.bce(real_pred, torch.ones_like(real_pred))
            fake_loss = srgan_criterion.bce(fake_pred, torch.zeros_like(fake_pred))
            d_loss = real_loss + fake_loss
            d_loss.backward()
            disc_optimizer.step()

            # Train Generator
            gen_optimizer.zero_grad()
            fake_pred_g = srgan_discriminator(fake_high_res)
            g_loss = srgan_criterion(fake_high_res, high_res, fake_pred_g)
            g_loss.backward()
            gen_optimizer.step()

        logger.info(
            f"SRGAN Epoch {epoch+1}/{n_epochs_srgan} - D Loss: {d_loss:.4f} - G Loss: {g_loss:.4f}"
        )

    # Checkpoint best model
    from utils.checkpoint import save_checkpoint

    save_checkpoint(
        pinn_model, pinn_optimizer, n_epochs, {"val_loss": val_loss}, "assets/results/best_pinn.pth"
    )

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
