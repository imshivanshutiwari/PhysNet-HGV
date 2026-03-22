"""
Adaptive Noise Covariance Estimation.

Dynamically adjusts the measurement noise covariance (R) and 
process noise covariance (Q) using innovation residuals to 
compensate for model mismatches and sensor artifacts.
"""

import numpy as np
from typing import Optional

class AdaptiveNoiseEstimator:
    """
    Implements windowed adaptive estimation for noise statistics.
    """
    
    def __init__(self, window_size: int = 10, forgetting_factor: float = 0.95):
        self.window = window_size
        self.alpha = forgetting_factor
        
        self.innovation_buffer = []
        self.R_fixed = None

    def update_R(self, innovation: np.ndarray, H_jac: np.ndarray, P_pred: np.ndarray, R_cur: np.ndarray) -> np.ndarray:
        """
        Estimate R using Sage-Husa adaptive filtering approach.
        R = (1-alpha)*R + alpha*(innov * innov.T - H*P_pred*H.T)
        """
        innov = innovation.reshape(-1, 1)
        innovation_cov = np.dot(innov, innov.T)
        
        # Innovation Covariance predicted by filter
        S_pred = np.dot(H_jac, np.dot(P_pred, H_jac.T))
        
        # New estimate of R
        R_est = innovation_cov - S_pred
        
        # Guard: R must be positive definite
        R_est = (R_est + R_est.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(R_est)
        eigvals = np.maximum(eigvals, 1e-6)
        R_est = np.dot(eigvecs, np.dot(np.diag(eigvals), eigvecs.T))
        
        # Smoothed update
        R_new = (1.0 - self.alpha) * R_cur + self.alpha * R_est
        
        return R_new

    def update_Q(self, K_gain: np.ndarray, innovation: np.ndarray, Q_cur: np.ndarray) -> np.ndarray:
        """
        Simplified adaptive process noise update.
        """
        innov = innovation.reshape(-1, 1)
        Q_est = np.dot(K_gain, np.dot(np.dot(innov, innov.T), K_gain.T))
        
        Q_new = (1.0 - self.alpha) * Q_cur + self.alpha * Q_est
        return Q_new

if __name__ == "__main__":
    # Test Adaptive Noise
    estimator = AdaptiveNoiseEstimator(forgetting_factor=0.1)
    
    innov = np.array([0.5, -0.2])
    H = np.eye(2)
    P = np.eye(2) * 0.1
    R = np.eye(2) * 1.0
    
    R_new = estimator.update_R(innov, H, P, R)
    print(f"Adaptive R update:\n{R_new}")
