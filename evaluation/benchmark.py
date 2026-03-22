import time
import numpy as np
import torch
from utils.logger import get_logger
from .evaluate import evaluate_model
from models.pinn_module import PINNModule
from filters.ukf_tracker import UKFTracker
from simulation.trajectory_gen import TrajectoryGenerator
from preprocessing.blackout_labeler import BlackoutLabeler

logger = get_logger("benchmark")


def run_benchmark():
    logger.info("Starting Benchmark")

    generator = TrajectoryGenerator()
    trajectories = generator.generate_batch(50)

    labeler = BlackoutLabeler()

    dt = 0.1
    std_ukf = UKFTracker(dt=dt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pinn_model = PINNModule().to(device)

    start_time = time.time()

    for traj in trajectories:
        true_traj = traj["trajectory"]
        meas_traj = traj["radar_returns"]
        ne_profile = traj["plasma_profile"]
        blackout_mask = labeler.label_trajectory(ne_profile)

        std_ukf_ests, std_ukf_covs = std_ukf.run_filter(meas_traj, blackout_mask, pinn_model=None)

        prop_ukf_ests, prop_ukf_covs = std_ukf.run_filter(
            meas_traj, blackout_mask, pinn_model=pinn_model
        )

    end_time = time.time()

    logger.info(f"Benchmark completed in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    run_benchmark()
