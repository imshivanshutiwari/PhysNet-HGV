"""
Multi-Modal Sensor Encoders for HGV Tracking.

Provides specialized feature extraction for Radar (1D), 
IR (Time-series), and Optical (2D) sensor modalities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class SensorEncoder(nn.Module):
    """
    Generic Feature Encoder with Multi-Head Self-Attention.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        feat_dim: int = 128, 
        num_heads: int = 8,
        num_layers: int = 2
    ):
        """
        Initialize the encoder.
        """
        super().__init__()
        
        # Initial projection
        self.proj = nn.Linear(input_dim, feat_dim)
        
        # Positional Encoding (for time-series/sequential inputs)
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, feat_dim)) # Max 100 timesteps
        
        # Transformer Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, 
            nhead=num_heads,
            dim_feedforward=feat_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        x: Shape (Batch, Seq, Input_Dim)
        """
        b, s, d = x.shape
        
        # Project
        feat = self.proj(x)
        
        # Add Positional Encoding
        feat = feat + self.pos_encoding[:, :s, :]
        
        # Self-Attention
        out = self.transformer(feat, src_key_padding_mask=mask)
        
        return self.norm(out)

class RadarEncoder(nn.Module):
    """
    Specialized Radar Encoder for Range-Doppler-AER inputs.
    """
    def __init__(self, input_dim: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.encoder = SensorEncoder(input_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

if __name__ == "__main__":
    # Test Encoders
    model = SensorEncoder(input_dim=10, feat_dim=128)
    dummy_input = torch.randn(16, 20, 10) # B, S, D
    out = model(dummy_input)
    print(f"Encoder Output Shape: {out.shape}")
