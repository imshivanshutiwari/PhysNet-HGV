"""
Cross-Modal Transformer for Sensor Fusion.

Integrates Radar, IR, and Optical sensor data using multi-sensor 
cross-attention to provide robust state estimates during varying 
channel reliability and blackout conditions.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from .encoder import SensorEncoder

class CrossAttention(nn.Module):
    """
    Standard Multi-Head Cross Attention layer.
    """
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None):
        out, weights = self.attn(query, key, value, key_padding_mask=mask)
        return self.norm(out + query), weights

class HGVTransformer(nn.Module):
    """
    Fusion Transformer for Radar, IR, and Optical sensors.
    """
    
    def __init__(self, feat_dim: int = 128, output_dim: int = 12):
        super().__init__()
        
        # Modal Encoders
        self.radar_enc = SensorEncoder(input_dim=4, feat_dim=feat_dim)
        self.ir_enc = SensorEncoder(input_dim=3, feat_dim=feat_dim)
        self.optical_enc = SensorEncoder(input_dim=2, feat_dim=feat_dim)
        
        # Cross-Attention Modules
        self.cross_attn = CrossAttention(dim=feat_dim)
        
        # Final Fusion and Prediction
        # Cat (3 * 128) -> 256 -> 128 -> 12
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(
        self, 
        radar: torch.Tensor, 
        ir: torch.Tensor, 
        optical: torch.Tensor,
        masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with sensor masking for blackout resilience.
        radar: [B, S, 4]
        ir: [B, S, 3]
        optical: [B, S, 2]
        """
        masks = masks or {}
        
        # 1. Encode each modality
        f_radar = self.radar_enc(radar, mask=masks.get("radar"))
        f_ir = self.ir_enc(ir, mask=masks.get("ir"))
        f_optical = self.optical_enc(optical, mask=masks.get("optical"))
        
        # 2. Sequential Cross-Attention (Simplify: sum or concat features)
        # In a more complex version, we'd use iterative cross-attention.
        # Here we concatenate for the fusion bottleneck.
        f_combined = torch.cat([f_radar, f_ir, f_optical], dim=-1)
        
        # 3. Predict state
        # We take the last timestep for point prediction or entire sequence.
        state_pred = self.fusion(f_combined)
        
        return state_pred, {}

if __name__ == "__main__":
    # Test Transformer
    model = HGVTransformer()
    
    r = torch.randn(2, 20, 4)
    ir = torch.randn(2, 20, 3)
    o = torch.randn(2, 20, 2)
    
    pred, _ = model(r, ir, o)
    print(f"Transformer Prediction Shape: {pred.shape}")
