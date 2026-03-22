"""
Global seed management for reproducibility across all frameworks.

Sets seeds for Python, NumPy, PyTorch (CPU + CUDA), and configures
deterministic CuDNN behavior.
"""

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """Set deterministic seeds across all frameworks.

    Parameters
    ----------
    seed : int
        The seed value to use across all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
