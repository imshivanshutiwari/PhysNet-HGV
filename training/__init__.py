"""
PhysNet-HGV Training Package.

Orchestrates the optimization of PINNs, Transformers, and 
Diffusion models using specialized physics-informed losses 
and multi-phase learning schedules.
"""

from .trainer import HGVTrainer
from .losses import PhysicsLoss, CombinedLoss
from .callbacks import TrainingMonitor

__all__ = [
    "HGVTrainer",
    "PhysicsLoss",
    "CombinedLoss",
    "TrainingMonitor",
]
