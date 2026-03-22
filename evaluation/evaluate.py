"""
Main HGV Pipeline Evaluator.

Executes full-loop simulations from high-fidelity trajectory 
generation and radar observation to state estimation and 
blackout bridging, reporting end-to-end performance metrics.
"""

import numpy as np
from typing import Dict
from simulation.hgv_dynamics import HGVDynamics
from simulation.radar_simulator import RadarSimulator
from simulation.plasma_model import PlasmaModel
from filters.ukf_tracker import UKFTracker
from evaluation.metrics import TrackingMetrics
from utils.config_loader import load_config

class HGVEvaluator:
    """
    Evaluator for end-to-end HGV tracking pipeline.
    """
    
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.plasma = PlasmaModel(self.config["pinn"])
        self.dynamics = HGVDynamics(self.config)
        self.radar = RadarSimulator(self.config, self.plasma)
        self.metrics = TrackingMetrics()

    def run_eval(self, n_steps: int = 1000) -> Dict[str, float]:
        """
        Run a single tracking scenario and evaluate performance.
        """
        # 1. Initial State
        state_true = np.zeros(12)
        state_true[0] = self.dynamics.WGS84_A + 60000.0 # 60km
        state_true[4] = 6000.0 # Velocity
        
        # Tracker Init
        tracker_cfg = {
            "dt": 0.1, "alpha": 0.001, "beta": 2.0, "sigma_m": 10.0,
            "lambda_fading": 0.98
        }
        tracker = UKFTracker(tracker_cfg)
        tracker.x = np.array([state_true[0], state_true[3], 0, 
                             state_true[1], state_true[4], 0,
                             state_true[2], state_true[5], 0])
        
        hist_true = []
        hist_est = []
        hist_cov = []
        
        def h_func(x):
            r = np.linalg.norm(x[0:3*3:3] - self.radar.pos_radar)
            v_rad = np.dot(x[1:3*3:3], (x[0:3*3:3] - self.radar.pos_radar)) / r if r > 1e-3 else 0
            return np.array([r, 0, 0, v_rad])

        # 2. Loop
        for i in range(n_steps):
            # True Step
            state_true = self.dynamics.step(i*0.1, 0.1, state_true, {"alpha": 10})
            
            # Predict
            tracker.predict()
            
            # Measure
            z, _, out = self.radar.get_measurement(state_true, i*0.1)
            
            # Update
            tracker.update(z, h_func)
            
            # Log
            hist_true.append(state_true)
            # Align tracker 9D to 6D state for RMSE
            hist_est.append(np.array([tracker.x[0], tracker.x[3], tracker.x[6], 
                                     tracker.x[1], tracker.x[4], tracker.x[7]]))
            hist_cov.append(tracker.P[:6, :6])
            
        # 3. Compute Metrics
        est = np.array(hist_est)
        truth = np.array(hist_true)[:, 0:6]
        covs = np.array(hist_cov)
        
        results = {
            "pos_rmse": self.metrics.position_rmse(est, truth),
            "vel_rmse": self.metrics.velocity_rmse(est, truth),
            "nees": self.metrics.nees(est, truth, covs)
        }
        return results

if __name__ == "__main__":
    print("--- PhysNet-HGV End-to-End Evaluation Demo ---")
    # Using a temporary mock config for the demo
    import yaml
    mock_cfg = "tests/demo_config.yaml"
    with open(mock_cfg, "w") as f:
        yaml.dump({
            "vehicle": {"mass_kg": 907.0, "reference_area_m2": 0.88, "Ixx": 120.0, "Iyy": 850.0, "Izz": 850.0},
            "aerodynamics": {"cd0": 0.015, "k_induced": 0.045, "cl_alpha": 0.08},
            "trajectory": {"dt_s": 0.1},
            "integration": {"method": "RK45", "rtol": 1e-6, "atol": 1e-8},
            "atmosphere": {"model": "us_standard_1976"},
            "sensor": {"radar": {"noise_range": 5.0, "noise_angle_deg": 0.01, "noise_doppler": 1.0, "snr_base_db": 40.0, "position_ecef": [0, 0, 0]}},
            "output": {"save_dir": "tests/demo_output"},
            "pinn": {"Ne_threshold": 1e18}
        }, f)
    
    evaluator = HGVEvaluator(mock_cfg)
    results = evaluator.run_eval(n_steps=50)
    print("\nEvaluation Results (50 Steps):")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    print("\nDemo completed successfully.")
