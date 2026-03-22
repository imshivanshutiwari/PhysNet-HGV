import numpy as np
from .hgv_dynamics import HGVDynamics
from .plasma_model import PlasmaModel
from typing import Dict, Any, List
from tqdm import tqdm


class TrajectoryGenerator:
    def __init__(self, config_path=None):
        self.dynamics = HGVDynamics()
        self.plasma = PlasmaModel()
        self.num_samples = 10000
        self.alt_range = (30000, 80000)
        self.vel_range = (5 * 340, 20 * 340)
        self.bank_range = (-30, 30)
        self.alpha_range = (0, 10)

    def generate_batch(self, batch_size=None, parallel=True) -> List[Dict[str, Any]]:
        if batch_size is None:
            batch_size = self.num_samples

        trajectories = []

        for _ in tqdm(range(batch_size), desc="Generating Trajectories"):
            alt = np.random.uniform(*self.alt_range)
            vel_mag = np.random.uniform(*self.vel_range)

            x, y, z = 0, 0, alt
            vx, vy, vz = vel_mag, 0, 0
            state = [x, y, z, vx, vy, vz]

            bank = np.random.uniform(*self.bank_range)
            alpha = np.random.uniform(*self.alpha_range)
            duration = np.random.uniform(10, 300)

            controls = [(0, alpha, bank)]
            traj = self.dynamics.integrate_trajectory(state, duration, controls)

            if len(traj) == 0:
                continue

            alt_array = np.array(
                [
                    np.linalg.norm([st[0], st[1], self.dynamics.Re + st[2]]) - self.dynamics.Re
                    for st in traj
                ]
            )
            vel_array = np.array([np.linalg.norm([st[3], st[4], st[5]]) for st in traj])

            Ne_array = self.plasma.compute_electron_density(alt_array, vel_array)
            attenuation_db = self.plasma.compute_radar_attenuation(Ne_array, 10e9)
            blackout_mask = self.plasma.is_blackout(Ne_array)

            clean_traj = traj
            noisy_traj = self.dynamics.add_measurement_noise(traj)

            traj_dict = {
                "trajectory": clean_traj,
                "radar_returns": noisy_traj,
                "plasma_profile": Ne_array,
                "attenuation_db": attenuation_db,
                "blackout_mask": blackout_mask,
                "controls": controls,
            }
            trajectories.append(traj_dict)

        return trajectories
