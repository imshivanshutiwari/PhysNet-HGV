"""
Advanced Tracking Metrics for Hypersonic Defense.

Implements POS/VEL RMSE, NEES, OSPA (Optimal Sub-Pattern Assignment), 
and GOSPA metrics for rigorous performance evaluation.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple

class TrackingMetrics:
    """
    Collection of high-fidelity tracking evaluation metrics.
    """
    
    @staticmethod
    def position_rmse(est: np.ndarray, truth: np.ndarray) -> float:
        """RMSE for 3D position [x,y,z]."""
        return np.sqrt(np.mean(np.sum((est[:, 0:3] - truth[:, 0:3])**2, axis=1)))

    @staticmethod
    def velocity_rmse(est: np.ndarray, truth: np.ndarray) -> float:
        """RMSE for 3D velocity [vx,vy,vz]."""
        return np.sqrt(np.mean(np.sum((est[:, 3:6] - truth[:, 3:6])**2, axis=1)))

    @staticmethod
    def nees(est: np.ndarray, truth: np.ndarray, covs: np.ndarray) -> float:
        """Normalized Estimation Error Squared (NEES). Should be ~dim_state."""
        errors = est - truth
        n = len(errors)
        dim = errors.shape[1]
        nees_vals = []
        for i in range(n):
            e = errors[i].reshape(-1, 1)
            # (e.T * P^-1 * e)
            val = np.dot(e.T, np.dot(np.linalg.inv(covs[i]), e))
            nees_vals.append(val.item())
        return np.mean(nees_vals)

    @staticmethod
    def ospa(est: np.ndarray, truth: np.ndarray, c: float = 100, p: int = 2) -> float:
        """
        Optimal Sub-Pattern Assignment (OSPA) metric.
        Handles missing tracks and false alarms.
        """
        m, n = len(est), len(truth)
        if m == 0 and n == 0: return 0.0
        if m == 0 or n == 0: return c
        
        # Distance matrix (cutoff at c)
        dist = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                d = np.linalg.norm(est[i] - truth[j])
                dist[i, j] = min(c, d)**p
                
        # Assignment problem
        row_ind, col_ind = linear_sum_assignment(dist)
        sum_dist = dist[row_ind, col_ind].sum()
        
        # Penalize for Cardinality difference
        penalty = (abs(m - n) * (c**p))
        
        ospa_val = ((1.0 / max(m, n)) * (sum_dist + penalty))**(1.0 / p)
        return ospa_val

    @staticmethod
    def track_continuity(blackout_mask: np.ndarray, tracking_mask: np.ndarray) -> float:
        """Percentage of total flight time where state estimate was within error bounds."""
        return (np.sum(tracking_mask) / len(blackout_mask)) * 100.0

if __name__ == "__main__":
    # Test Metrics
    truth = np.array([[10, 0, 0, 100, 0, 0]])
    est = np.array([[11, 0, 0, 105, 0, 0]])
    cov = np.array([np.eye(6) * 1.0])
    
    print(f"Pos RMSE: {TrackingMetrics.position_rmse(est, truth):.2f}")
    print(f"NEES: {TrackingMetrics.nees(est, truth, cov):.2f}")
    print(f"OSPA: {TrackingMetrics.ospa(est, truth):.2f}")
