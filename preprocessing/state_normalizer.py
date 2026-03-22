"""
Robust State Normalization for Hypersonic Tracking.

Provides statistical scaling for HGV states (position, velocity, 
orientation, angular rates) to ensure numerical stability in 
Deep Learning models (PINN, Transformers).
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, Union

class StateNormalizer:
    """
    Normalizes 12D state vectors [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
    using pre-calculated or online statistics.
    """
    
    def __init__(self, stats_path: Optional[str] = None):
        """
        Initialize the normalizer.
        
        Parameters:
            stats_path: Path to a JSON file containing mean and standard deviation.
        """
        self.mean = None
        self.std = None
        
        if stats_path and Path(stats_path).exists():
            with open(stats_path, "r") as f:
                stats = json.load(f)
                self.mean = np.array(stats["mean"])
                self.std = np.array(stats["std"])

    def fit(self, data: np.ndarray):
        """
        Calculate mean and standard deviation from a dataset.
        data: Shape (N, 12)
        """
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        # Avoid division by zero
        self.std[self.std < 1e-8] = 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Apply normalization: (x - mean) / std
        """
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer not fitted.")
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Reverse normalization: x * std + mean
        """
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer not fitted.")
        return (data * self.std) + self.mean

    def save(self, path: str):
        """
        Save statistics to JSON.
        """
        stats = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist()
        }
        with open(path, "w") as f:
            json.dump(stats, f)

if __name__ == "__main__":
    # Test Normalizer
    norm = StateNormalizer()
    sample_data = np.random.normal(100.0, 50.0, (1000, 12))
    
    norm.fit(sample_data)
    transformed = norm.transform(sample_data)
    
    print(f"Transformed Mean (approx 0): {np.mean(transformed, axis=0)[0]:.4f}")
    print(f"Transformed Std (approx 1): {np.std(transformed, axis=0)[0]:.4f}")
    
    back = norm.inverse_transform(transformed)
    print(f"Inverse Transform Reconstruction Error: {np.mean(np.abs(back - sample_data)):.4e}")
