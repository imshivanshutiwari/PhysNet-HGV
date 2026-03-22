"""
Covariance Intersection (CI) for Multi-Sensor Fusion.

Fuses multiple state estimates (e.g., Radar-UKF and PINN-ODE) 
without requiring knowledge of their mutual cross-correlation.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from typing import Tuple, List

class CovarianceIntersection:
    """
    Implements the Covariance Intersection algorithm for consistent fusion.
    """
    
    @staticmethod
    def fuse(x1: np.ndarray, P1: np.ndarray, x2: np.ndarray, P2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fuse two estimates using a scalar weight omega that minimizes 
        the fused covariance trace.
        
        P_fused = (w * P1^-1 + (1-w) * P2^-1)^-1
        x_fused = P_fused * (w * P1^-1 * x1 + (1-w) * P2^-1 * x2)
        """
        # Optimize w (0 < w < 1)
        res = minimize_scalar(
            lambda w: np.trace(np.linalg.inv(
                w * np.linalg.inv(P1) + (1 - w) * np.linalg.inv(P2)
            )),
            bounds=(0, 1), 
            method='bounded'
        )
        w = res.x
        
        P1_inv = np.linalg.inv(P1)
        P2_inv = np.linalg.inv(P2)
        
        P_fused = np.linalg.inv(w * P1_inv + (1 - w) * P2_inv)
        x_fused = np.dot(P_fused, (w * np.dot(P1_inv, x1) + (1 - w) * np.dot(P2_inv, x2)))
        
        return x_fused, P_fused

    @staticmethod
    def fuse_batch(states: List[np.ndarray], covs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fused multiple estimates iteratively.
        """
        x_fused = states[0]
        P_fused = covs[0]
        
        for i in range(1, len(states)):
            x_fused, P_fused = CovarianceIntersection.fuse(x_fused, P_fused, states[i], covs[i])
            
        return x_fused, P_fused

if __name__ == "__main__":
    # Test fusion
    x1 = np.array([10, 0, 0])
    P1 = np.eye(3) * 1.0
    
    x2 = np.array([12, 0, 0])
    P2 = np.eye(3) * 2.0
    
    ci = CovarianceIntersection()
    xf, Pf = ci.fuse(x1, P1, x2, P2)
    
    print(f"Fused State: {xf}")
    print(f"Fused Cov Trace: {np.trace(Pf):.4f} (Originals: 3.0, 6.0)")
