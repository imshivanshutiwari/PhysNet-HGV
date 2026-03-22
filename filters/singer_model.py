"""
Singer Maneuvering Acceleration Model.

A high-fidelity state-space model for maneuvering targets, representing 
acceleration as a first-order Gauss-Markov process.
"""

import numpy as np
from typing import Tuple

class SingerModel:
    """
    Singer Acceleration Model for 3D trajectory prediction.
    """
    
    def __init__(self, alpha: float = 0.05, sigma_m: float = 10.0, dt: float = 0.1):
        """
        Parameters:
            alpha: 1/tau (reciprocal of maneuver time constant).
            sigma_m: Maneuver acceleration standard deviation.
            dt: Sampling interval.
        """
        self.alpha = alpha
        self.sigma_m = sigma_m
        self.dt = dt
        
    def get_transition_matrix(self) -> np.ndarray:
        """
        Returns the 9x9 transition matrix F for [x, dx, ddx, y, dy, ddy, z, dz, ddz].
        """
        a = self.alpha
        dt = self.dt
        adt = a * dt
        e_adt = np.exp(-adt)
        
        # Fundamental block for 1D
        f1 = np.array([
            [1, dt, (adt - 1 + e_adt) / (a**2)],
            [0, 1, (1 - e_adt) / a],
            [0, 0, e_adt]
        ])
        
        # Expand to 3D
        F = np.zeros((9, 9))
        for i in range(3):
            F[i*3:(i+1)*3, i*3:(i+1)*3] = f1
            
        return F

    def get_process_noise(self) -> np.ndarray:
        """
        Returns the 9x9 process noise covariance matrix Q.
        """
        a = self.alpha
        dt = self.dt
        adt = a * dt
        e_adt = np.exp(-adt)
        e_2adt = np.exp(-2*adt)
        
        # 1D Q matrix elements (simplified from Singer 1970)
        q11 = (1 / (2 * a**5)) * (1 - e_2adt + 2*adt + (2/3)*adt**3 - 2*adt**2 - 4*adt*e_adt)
        q12 = (1 / (2 * a**4)) * (e_2adt + 1 - 2*e_adt + 2*adt*e_adt - 2*adt + adt**2)
        q13 = (1 / (2 * a**3)) * (1 - e_2adt - 2*adt*e_adt)
        q22 = (1 / (2 * a**3)) * (4*e_adt - 3 - e_2adt + 2*adt)
        q23 = (1 / (2 * a**2)) * (e_2adt + 1 - 2*e_adt)
        q33 = (1 / (2 * a)) * (1 - e_2adt)
        
        q_block = np.array([
            [q11, q12, q13],
            [q12, q22, q23],
            [q13, q23, q33]
        ]) * (2 * a * self.sigma_m**2)
        
        # Expand to 3D
        Q = np.zeros((9, 9))
        for i in range(3):
            Q[i*3:(i+1)*3, i*3:(i+1)*3] = q_block
            
        return Q

if __name__ == "__main__":
    # Test Singer Model
    model = SingerModel(alpha=0.05, sigma_m=10.0, dt=0.1)
    F = model.get_transition_matrix()
    Q = model.get_process_noise()
    
    print(f"F shape: {F.shape}")
    print(f"Q top-left block:\n{Q[:3, :3]}")
