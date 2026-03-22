"""
PhysNet-HGV Model Architectures.

Includes Physics-Informed Neural Networks (PINN), Neural ODEs, 
Cross-Modal Transformers for sensor fusion, and SRGAN for 
optical super-resolution.
"""

from .encoder import SensorEncoder
from .pinn_module import PINNModule
from .neural_ode import NeuralODETrainer
from .cross_modal_transformer import HGVTransformer
from .srgan import SRGANGenerator, SRGANDiscriminator

__all__ = [
    "SensorEncoder",
    "PINNModule",
    "NeuralODETrainer",
    "HGVTransformer",
    "SRGANGenerator",
    "SRGANDiscriminator",
]
