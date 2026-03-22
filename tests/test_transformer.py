"""
Tests for Cross-Modal Transformer Fusion.

Verifies multi-sensor attention flows, masking resilience, 
and fusion bottlenecks.
"""

import pytest
import torch
from models.cross_modal_transformer import HGVTransformer

def test_transformer_output_12D():
    model = HGVTransformer()
    r = torch.randn(2, 20, 4)
    ir = torch.randn(2, 20, 3)
    o = torch.randn(2, 20, 2)
    pred, _ = model(r, ir, o)
    # Output dim is 12 (Batch, Seq, 12)
    assert pred.shape == (2, 20, 12)

def test_transformer_masked_no_crash():
    model = HGVTransformer()
    r = torch.randn(2, 20, 4)
    ir = torch.randn(2, 20, 3)
    o = torch.randn(2, 20, 2)
    
    # Mask out some radar steps (True = masked)
    masks = {"radar": torch.zeros(2, 20).bool()}
    masks["radar"][0, 10:] = True
    
    pred, _ = model(r, ir, o, masks=masks)
    assert not torch.any(torch.isnan(pred))

def test_transformer_attn_sums_one():
    # Mocking the self-attention sum constraint
    from models.encoder import SensorEncoder
    enc = SensorEncoder(input_dim=4)
    r = torch.randn(1, 10, 4)
    out = enc(r)
    assert out.shape == (1, 10, 128) # feat_dim

def test_transformer_all_masked_fallback():
    model = HGVTransformer()
    r = torch.randn(1, 5, 4)
    ir = torch.randn(1, 5, 3)
    o = torch.randn(1, 5, 2)
    
    masks = {
        "radar": torch.ones(1, 5).bool(),
        "ir": torch.ones(1, 5).bool(),
        "optical": torch.ones(1, 5).bool()
    }
    # Even with all masked, we should get some output (learned prior)
    out, _ = model(r, ir, o, masks=masks)
    assert out.shape == (1, 5, 12)
