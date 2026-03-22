import pytest
import torch
from models.cross_modal_transformer import CrossModalTransformer


@pytest.fixture
def cmt():
    return CrossModalTransformer(radar_dim=4, ir_dim=3, optical_dim=2, hidden_dim=128, num_heads=8)


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def inputs(batch_size):
    radar = torch.randn(batch_size, 4)
    ir = torch.randn(batch_size, 3)
    optical = torch.randn(batch_size, 2)
    return radar, ir, optical


def test_fusion_output_shape(cmt, inputs, batch_size):
    radar, ir, optical = inputs

    radar_mask = torch.zeros(batch_size, 1, dtype=torch.bool)
    ir_mask = torch.zeros(batch_size, 1, dtype=torch.bool)
    optical_mask = torch.zeros(batch_size, 1, dtype=torch.bool)

    fused_state, attn_weights = cmt(radar, ir, optical, radar_mask, ir_mask, optical_mask)

    assert fused_state.shape == (batch_size, 12)
    assert attn_weights.shape == (batch_size, 3)


def test_masked_sensor_no_exception(cmt, inputs, batch_size):
    radar, ir, optical = inputs

    radar_mask = torch.ones(batch_size, 1, dtype=torch.bool)
    ir_mask = torch.zeros(batch_size, 1, dtype=torch.bool)
    optical_mask = torch.zeros(batch_size, 1, dtype=torch.bool)

    fused_state, attn_weights = cmt(radar, ir, optical, radar_mask, ir_mask, optical_mask)

    assert fused_state.shape == (batch_size, 12)


def test_attention_weights_normalized(cmt, inputs, batch_size):
    radar, ir, optical = inputs

    radar_mask = torch.zeros(batch_size, 1, dtype=torch.bool)
    ir_mask = torch.zeros(batch_size, 1, dtype=torch.bool)
    optical_mask = torch.zeros(batch_size, 1, dtype=torch.bool)

    fused_state, attn_weights = cmt(radar, ir, optical, radar_mask, ir_mask, optical_mask)

    sums = attn_weights.sum(dim=1)

    assert torch.allclose(sums, torch.ones(batch_size))


def test_all_masked_returns_fallback(cmt, inputs, batch_size):
    radar, ir, optical = inputs

    radar_mask = torch.ones(batch_size, 1, dtype=torch.bool)
    ir_mask = torch.ones(batch_size, 1, dtype=torch.bool)
    optical_mask = torch.ones(batch_size, 1, dtype=torch.bool)

    fused_state, attn_weights = cmt(radar, ir, optical, radar_mask, ir_mask, optical_mask)

    assert fused_state.shape == (batch_size, 12)
