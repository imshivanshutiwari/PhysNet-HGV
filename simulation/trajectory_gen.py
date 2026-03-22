"""
HGV Trajectory Generation Pipeline.

Orchestrates the 6-DOF dynamics and plasma models to generate a 
large-scale dataset of maneuvering trajectories for training 
Physics-Informed Neural Networks and Diffusion Models.
"""

import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
import yaml

from simulation.hgv_dynamics import HGVDynamics
from simulation.plasma_model import PlasmaModel

class TrajectoryGenerator:
    """
    Automated generator for HGV trajectories with varying initial conditions.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the generator with project configuration.
        """
        self.config = config
        self.dynamics = HGVDynamics(config)
        self.plasma = PlasmaModel(config.get("pinn", {})) # Config for plasma threshold
        
        self.dt = config["trajectory"]["dt_s"]
        self.t_max = config["trajectory"]["t_max_s"]
        self.save_dir = Path(config["output"]["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def generate_single_trajectory(self, traj_id: int) -> str:
        """
        Generate one HGV trajectory with randomized initial conditions.
        """
        # 1. Randomized Initial Conditions (from config ranges)
        alt_range = self.config["trajectory"]["altitude_range_km"]
        vel_range = self.config["trajectory"]["velocity_range_mach"]
        bank_range = self.config["trajectory"]["bank_angle_range_deg"]
        
        alt_init = np.random.uniform(alt_range[0], alt_range[1]) * 1000.0
        mach_init = np.random.uniform(vel_range[0], vel_range[1])
        bank_init = np.random.uniform(bank_range[0], bank_range[1])
        
        # Initial State Construction (ECEF)
        r0_mag = self.dynamics.WGS84_A + alt_init
        r0 = np.array([r0_mag, 0, 0])
        
        # Approx velocity from Mach (using 300m/s as base at alt)
        v0_mag = mach_init * 300.0 
        v0 = np.array([0, v0_mag, 0])
        
        state = np.zeros(12)
        state[0:3] = r0
        state[3:6] = v0
        
        # 2. Integration Loop
        t = 0.0
        history = []
        
        n_steps = int(self.t_max / self.dt)
        
        for _ in range(n_steps):
            # Check for crash or escape
            alt_curr = np.linalg.norm(state[0:3]) - self.dynamics.WGS84_A
            if alt_curr < 0 or alt_curr > 120000:
                break
                
            # Control Input (Simplified: Constant alpha/bank for now)
            # In a more advanced version, these would vary over time.
            control = {"alpha": 15.0, "bank": bank_init}
            
            # Record State + Plasma
            # (Note: In a full sim we'd fetch actual T/P from atmosphere inside dynamics)
            # For logging, we'll re-calculate plasma state.
            v_mag = np.linalg.norm(state[3:6])
            T_stag = 250.0 * (1 + 0.2 * (v_mag/300.0)**2)
            P_stag = 101325.0 * np.exp(-alt_curr / 7000.0)
            plasma_state = self.plasma.get_plasma_state(T_stag, P_stag)
            
            # Combine all data
            record = np.concatenate([
                [t], state, 
                [plasma_state["electron_density"], float(plasma_state["is_blackout"])]
            ])
            history.append(record)
            
            # Step Dynamics
            state = self.dynamics.step(t, self.dt, state, control)
            t += self.dt
            
        # 3. Save to Disk
        history = np.array(history)
        save_path = self.save_dir / f"traj_{traj_id:05d}.npz"
        
        # Columns: t, x, y, z, vx, vy, vz, phi, theta, psi, p, q, r, ne, blackout
        np.savez_compressed(
            save_path, 
            data=history,
            columns=self.config["output"]["fields"] + ["ne", "blackout"]
        )
        
        return str(save_path)

    def run_batch(self, n_trajectories: int):
        """
        Generate a batch of trajectories.
        """
        print(f"Generating {n_trajectories} trajectories...")
        for i in tqdm(range(n_trajectories)):
            self.generate_single_trajectory(i)

if __name__ == "__main__":
    # Test Generation
    import sys
    from utils.config_loader import load_config
    
    # Use config if exists, else mock
    cfg_path = "configs/hgv_config.yaml"
    if os.path.exists(cfg_path):
        config = load_config(cfg_path)
    else:
        # Mocking for standalone test flow
        config = {
            "vehicle": {"mass_kg": 907.0, "reference_area_m2": 0.88, "Ixx": 120.0, "Iyy": 850.0, "Izz": 850.0},
            "aerodynamics": {"cd0": 0.015, "k_induced": 0.045, "cl_alpha": 0.08},
            "trajectory": {
                "altitude_range_km": [40, 60], "velocity_range_mach": [10.0, 15.0], 
                "bank_angle_range_deg": [-10, 10], "n_trajectories": 5, "dt_s": 1.0, "t_max_s": 100.0
            },
            "atmosphere": {"model": "us_standard_1976"},
            "integration": {"method": "RK45", "rtol": 1e-6, "atol": 1e-8},
            "output": {"save_dir": "data/trajectories", "fields": ["t", "x", "y", "z", "vx", "vy", "vz", "phi", "theta", "psi", "p", "q", "r"]},
            "pinn": {"Ne_threshold": 1e18}
        }
        
    generator = TrajectoryGenerator(config)
    generator.run_batch(10)
    print("Done.")
