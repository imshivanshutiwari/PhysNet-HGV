"""
PhysNet-HGV Preprocessing Package.

Algorithms for radar detection (CFAR), state normalization, 
blackout sequence labeling, and training data pipelines.
"""

from .cfar_detector import CFARDetector
from .state_normalizer import StateNormalizer
from .blackout_labeler import BlackoutLabeler
from .data_pipeline import HGVDataset

__all__ = [
    "CFARDetector",
    "StateNormalizer",
    "BlackoutLabeler",
    "HGVDataset",
]
