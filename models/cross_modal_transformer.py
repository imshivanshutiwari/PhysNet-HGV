import torch
import torch.nn as nn


class CrossModalTransformer(nn.Module):
    def __init__(self, radar_dim=4, ir_dim=3, optical_dim=2, hidden_dim=128, num_heads=8):
        super(CrossModalTransformer, self).__init__()

        self.radar_enc = nn.Linear(radar_dim, hidden_dim)
        self.ir_enc = nn.Linear(ir_dim, hidden_dim)
        self.optical_enc = nn.Linear(optical_dim, hidden_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 12),
        )

    def forward(self, radar, ir, optical, radar_mask, ir_mask, optical_mask):
        B = radar.shape[0]

        r_emb = self.radar_enc(radar).unsqueeze(1)
        i_emb = self.ir_enc(ir).unsqueeze(1)
        o_emb = self.optical_enc(optical).unsqueeze(1)

        seq = torch.cat([r_emb, i_emb, o_emb], dim=1)

        key_padding_mask = torch.cat([radar_mask, ir_mask, optical_mask], dim=1)

        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask[all_masked] = False

        attn_out, attn_weights = self.attn(seq, seq, seq, key_padding_mask=key_padding_mask)

        if all_masked.any():
            attn_out[all_masked] = 0.0

        flattened = attn_out.reshape(B, -1)
        fused_state = self.fusion(flattened)

        attn_weights = attn_weights.mean(dim=1)

        return fused_state, attn_weights
