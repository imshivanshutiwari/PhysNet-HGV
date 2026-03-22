"""
PyTorch-based Data Pipeline for PhysNet-HGV.

Handles loading, batching, and windowing of HGV trajectories 
for training Neural ODEs and Cross-Modal Transformers.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from .state_normalizer import StateNormalizer

class HGVDataset(Dataset):
    """
    Dataset class for windowed trajectory sequences.
    """
    
    def __init__(
        self, 
        data_dir: str, 
        window_size: int = 20, 
        stride: int = 5,
        normalizer: Optional[StateNormalizer] = None,
        max_samples: Optional[int] = None
    ):
        """
        Initialize the dataset.
        """
        self.data_dir = Path(data_dir)
        self.window = window_size
        self.stride = stride
        self.norm = normalizer
        
        self.trajectories = sorted(list(self.data_dir.glob("*.npz")))
        if max_samples:
            self.trajectories = self.trajectories[:max_samples]
            
        self.indices = self._precompute_indices()

    def _precompute_indices(self) -> List[Tuple[int, int]]:
        """
        Precompute mapping from global index to (file_idx, start_idx).
        """
        indices = []
        for file_idx, path in enumerate(self.trajectories):
            data = np.load(path)["data"]
            n_points = data.shape[0]
            if n_points >= self.window:
                for start in range(0, n_points - self.window + 1, self.stride):
                    indices.append((file_idx, start))
        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx, start = self.indices[idx]
        file_path = self.trajectories[file_idx]
        
        # Load and slice
        with np.load(file_path) as f:
            data = f["data"][start : start + self.window]
            
        # Split fields
        # t (0), state (1-13), ne (14), blackout (15)
        t = data[:, 0]
        state = data[:, 1:13] # 12D state
        ne = data[:, 13]
        blackout = data[:, 14]
        
        # Normalize
        if self.norm:
            state = self.norm.transform(state)
            
        return {
            "time": torch.tensor(t, dtype=torch.float32),
            "state": torch.tensor(state, dtype=torch.float32),
            "electron_density": torch.tensor(ne, dtype=torch.float32),
            "blackout": torch.tensor(blackout, dtype=torch.float32)
        }

def get_hgv_dataloader(data_dir: str, batch_size: int = 64, **kwargs) -> DataLoader:
    """
    Factory function for DataLoader.
    """
    dataset = HGVDataset(data_dir, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    # Test Dataset (Requires generated data)
    # Check if data exists
    test_dir = "data/test_trajectories"
    if Path(test_dir).exists():
        loader = get_hgv_dataloader(test_dir, batch_size=4, window_size=10)
        for batch in loader:
            print(f"Batch state shape: {batch['state'].shape}")
            break
    else:
        print(f"No test data found at {test_dir}. Skipping loader test.")
