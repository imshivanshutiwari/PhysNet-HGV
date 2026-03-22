"""
PhysNet-HGV Kalman Filtering Package.

State estimation and tracking algorithms including the Unscented 
Kalman Filter (UKF) and the Singer acceleration model.
"""

from .ukf_tracker import UKFTracker
from .singer_model import SingerModel
from .covariance_intersection import CovarianceIntersection
from .adaptive_noise import AdaptiveNoiseEstimator

__all__ = [
    "UKFTracker",
    "SingerModel",
    "CovarianceIntersection",
    "AdaptiveNoiseEstimator",
]
