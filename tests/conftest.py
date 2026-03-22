import pytest
import torch
import numpy as np


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def sample_trajectory():
    return np.random.randn(100, 6)


@pytest.fixture
def sample_blackout_mask():
    mask = np.zeros(100, dtype=bool)
    mask[30:70] = True
    return mask
