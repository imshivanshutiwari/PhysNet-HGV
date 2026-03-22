"""
Unscented Kalman Filter (UKF) Tracker for HGV.

Implements a robust nonlinear estimator using the Unscented Transform, 
integrated with the Singer maneuver model and PINN-based blackout 
trajectory bridging.
"""

import numpy as np
from typing import Callable, Tuple, Optional, Dict
from .singer_model import SingerModel

class UKFTracker:
    """
    Advanced UKF for 9D state estimation [x, vx, ax, y, vy, ay, z, vz, az].
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the UKF with config parameters.
        """
        self.dt = config.get("dt", 0.1)
        self.state_dim = 9
        
        # UKF Parameters (Julier-Uhlmann)
        self.alpha = config.get("alpha", 0.001)
        self.beta = config.get("beta", 2.0)
        self.kappa = config.get("kappa", 0.0)
        
        self.set_weights()
        
        # Singer Model
        self.singer = SingerModel(
            alpha=config.get("singer_alpha", 0.05),
            sigma_m=config.get("sigma_m", 10.0),
            dt=self.dt
        )
        
        # State and Covariance
        self.x = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim) * 100.0
        
        # Process and Measurement Noise
        self.Q = self.singer.get_process_noise()
        self.R = np.eye(4) * 1.0 # Default [R, Az, El, Vdot]
        
        # Fading Memory Factor
        self.lambd = config.get("lambda_fading", 0.98)

    def set_weights(self):
        """
        Precompute weights for the Unscented Transform.
        """
        n = self.state_dim
        lam = self.alpha**2 * (n + self.kappa) - n
        
        self.weights_m = np.full(2*n + 1, 1 / (2 * (n + lam)))
        self.weights_c = np.full(2*n + 1, 1 / (2 * (n + lam)))
        
        self.weights_m[0] = lam / (n + lam)
        self.weights_c[0] = self.weights_m[0] + (1 - self.alpha**2 + self.beta)
        
        self.scale = np.sqrt(n + lam)

    def get_sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Generate 2n+1 sigma points.
        """
        n = len(x)
        # Numerical stability: enforce symmetry and check for non-positive definite
        P = (P + P.T) / 2.0
        try:
            L = np.linalg.cholesky(P)
        except np.linalg.norm(P) > 1e10:
             L = np.eye(n) * 1e5
        except np.linalg.LinAlgError:
             L = np.eye(n) # Fallback
             
        sigmas = np.zeros((2*n + 1, n))
        sigmas[0] = x
        for i in range(n):
            sigmas[i+1] = x + self.scale * L[:, i]
            sigmas[n+i+1] = x - self.scale * L[:, i]
        return sigmas

    def predict(self):
        """
        UKF Prediction step using Singer transition model.
        """
        F = self.singer.get_transition_matrix()
        
        # Prediction of Sigma Points
        sigmas = self.get_sigma_points(self.x, self.P)
        sigmas_pred = sigmas @ F.T
        
        # Predicted Mean
        self.x = np.dot(self.weights_m, sigmas_pred)
        
        # Predicted Covariance (with fading memory)
        delta = sigmas_pred - self.x
        self.P = np.dot(self.weights_c * delta.T, delta) + self.Q
        self.P /= self.lambd # Fading memory inflation

    def update(self, z: np.ndarray, h_func: Callable):
        """
        UKF Update step.
        h_func: measurement function (maps 9D state to measurement vector)
        z: measurement vector [Range, Az, El, V_radial]
        """
        # If z contains NaNs (blackout), skip measurement update
        if np.any(np.isnan(z)):
            return
            
        n = self.state_dim
        sigmas_pred = self.get_sigma_points(self.x, self.P)
        
        # Project sigmas to measurement space
        dim_z = len(z)
        sigmas_z = np.zeros((2*n + 1, dim_z))
        for i in range(2*n + 1):
            sigmas_z[i] = h_func(sigmas_pred[i])
            
        # Measurement Mean
        z_mean = np.dot(self.weights_m, sigmas_z)
        
        # Measurement Covariance and Cross-Covariance
        dz = sigmas_z - z_mean
        dx = sigmas_pred - self.x
        
        Pzz = np.dot(self.weights_c * dz.T, dz) + self.R
        Pxz = np.dot(self.weights_c * dx.T, dz)
        
        # Kalman Gain
        K = np.dot(Pxz, np.linalg.inv(Pzz))
        
        # Update
        self.x = self.x + np.dot(K, (z - z_mean))
        self.P = self.P - np.dot(K, np.dot(Pzz, K.T))

    def update_pinn_blackout(self, pinn_state: np.ndarray, pinn_cov: np.ndarray):
        """
        Blackout Bridge: Use PINN output as a pseudo-measurement.
        """
        # PINN gives state in 12D or 9D. Align to 9D.
        z_pinn = pinn_state[:9]
        R_pinn = pinn_cov
        
        # Standard KF update with PINN as measurement
        H = np.eye(self.state_dim)
        S = self.P + R_pinn
        K = self.P @ np.linalg.inv(S)
        
        self.x = self.x + K @ (z_pinn - self.x)
        self.P = (np.eye(self.state_dim) - K) @ self.P

if __name__ == "__main__":
    # Test UKF
    cfg = {"dt": 0.1, "alpha": 0.001, "beta": 2.0, "sigma_m": 10.0}
    tracker = UKFTracker(cfg)
    
    # Simple predict/update cycle
    def h_mock(x): return x[:4] # [pos_x, vel_x, acc_x, pos_y]
    
    tracker.predict()
    z_mock = np.array([100.0, 10.0, 1.0, 50.0])
    tracker.update(z_mock, h_mock)
    
    print(f"Post-update state: {tracker.x[:4]}")
    print(f"Post-update cov trace: {np.trace(tracker.P):.4f}")
