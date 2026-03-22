"""
Pytest Configuration and Fixtures.

Provides shared resources for testing, including mocked configurations 
and sample trajectory data.
"""

import sys
import os
from pathlib import Path

# Ensure the project root is in sys.path for absolute imports
root_path = str(Path(__file__).parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import pytest
import numpy as np
import torch
import yaml

@pytest.fixture
def mock_config():
    """Returns a basic configuration for testing."""
    return {
        "vehicle": {"mass_kg": 907.0, "reference_area_m2": 0.88, "Ixx": 120.0, "Iyy": 850.0, "Izz": 850.0},
        "aerodynamics": {"cd0": 0.015, "k_induced": 0.045, "cl_alpha": 0.08},
        "trajectory": {
            "altitude_range_km": [40, 60], "velocity_range_mach": [10.0, 15.0], 
            "bank_angle_range_deg": [-10, 10], "n_trajectories": 5, "dt_s": 0.1, "t_max_s": 10.0
        },
        "atmosphere": {"model": "us_standard_1976"},
        "integration": {"method": "RK45", "rtol": 1e-6, "atol": 1e-8},
        "sensor": {
            "radar": {
                "noise_range": 5.0, "noise_angle_deg": 0.01, "noise_doppler": 1.0,
                "snr_base_db": 40.0, "position_ecef": [0, 0, 0]
            }
        },
        "output": {"save_dir": "tests/test_output"},
        "pinn": {"Ne_threshold": 1e18}
    }

@pytest.fixture
def sample_state():
    """Standard HGV state [12D]."""
    state = np.zeros(12)
    state[0] = 6378137.0 + 50000.0 # 50km
    state[4] = 3000.0 # Velocity
    return state
