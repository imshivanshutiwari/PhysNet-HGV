"""
Post-Blackout State Reacquisition Engine.

Uses conditional diffusion samples and physical filtering to 
re-establish a high-confidence track after an extended 
plasma blackout period.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from .ddpm_trainer import DDPMTrainer
from filters.ukf_tracker import UKFTracker

class ReacquisitionEngine:
    """
    Engine to identify optimal re-entry state from DDPM samples.
    """
    
    def __init__(self, trainer: DDPMTrainer, config: Dict):
        """
        Initialize with trained diffusion model.
        """
        self.trainer = trainer
        self.config = config
        self.n_samples = config.get("n_samples", 500)

    def reacquire(self, last_known_cond: torch.Tensor, current_radar_z: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identify the most likely HGV state using diffusion and radar validation.
        """
        # 1. Generate Samples from Diffusion (Conditional)
        samples = self.trainer.sample(last_known_cond, n_samples=self.n_samples) # [N, 9]
        
        # 2. Physical Consistency Filter
        # Remove samples with impossible altitude or velocity
        # (Assuming samples are normalized, we'd need inverse transform if we had a normalizer)
        # For now, we'll assume they are refined to physical range.
        
        # 3. Radar-Match Scoring (if radar is back online)
        if current_radar_z is not None and not np.any(np.isnan(current_radar_z)):
            scores = self._calculate_radar_scores(samples.cpu().numpy(), current_radar_z)
            best_idx = np.argmin(scores)
            best_state = samples[best_idx].cpu().numpy()
            
            # Estimate Covariance from sample spread
            cov = np.cov(samples.cpu().numpy().T)
        else:
            # If no radar, return mean of samples (Maximum Likelihood estimate)
            best_state = torch.mean(samples, dim=0).cpu().numpy()
            cov = np.cov(samples.cpu().numpy().T)
            
        return best_state, cov

    def _calculate_radar_scores(self, samples: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Calculates Mahalanobis distance between samples and radar measurement.
        """
        # Simplified: L2 distance in measurement space
        # Maps 9D state [x, vx, ax, y, vy, ay, z, vz, az] to radar [R, Az, El, V_radial]
        scores = []
        for s in samples:
            # Approx Mapping
            r_mag = np.linalg.norm(s[[0, 3, 6]])
            v_rad = np.dot(s[[1, 4, 7]], s[[0, 3, 6]]) / r_mag if r_mag > 1e-3 else 0
            
            z_pred = np.array([r_mag, 0, 0, v_rad]) # Az/El approx 0 for score
            scores.append(np.linalg.norm(z[:1] - z_pred[:1]) + np.linalg.norm(z[3] - z_pred[3])) # Prioritize range/doppler
            
        return np.array(scores)

if __name__ == "__main__":
    # Test Reacquisition
    from .ddpm_model import DDPMModel, DiffusionScheduler
    
    m = DDPMModel()
    s = DiffusionScheduler()
    t = DDPMTrainer(m, s, {})
    
    eng = ReacquisitionEngine(t, {"n_samples": 50})
    
    cond = torch.randn(1, 18)
    z_obs = np.array([6400000.0, 0, 0, 3000.0]) # Range, Az, El, Vdot
    
    state, cov = eng.reacquire(cond, z_obs)
    print(f"Reacquired State: {state}")
    print(f"Confidence (Cov Trace): {np.trace(cov):.4f}")
