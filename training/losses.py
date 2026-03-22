import torch
import torch.nn as nn


class PINNLoss(nn.Module):
    def __init__(self, lambda_momentum=1.0, lambda_continuity=0.5, lambda_plasma=0.8):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.lambda_momentum = lambda_momentum
        self.lambda_continuity = lambda_continuity
        self.lambda_plasma = lambda_plasma

    def compute_derivatives(self, state):
        dt = 0.1
        v = state[:, :, 3:6] if state.dim() == 3 else state[:, 3:6]
        if state.dim() == 3:
            dv_dt = (v[:, 1:] - v[:, :-1]) / dt
            dv_dt = torch.cat([dv_dt, dv_dt[:, -1:]], dim=1)
        else:
            dv_dt = torch.zeros_like(v)
        return dv_dt

    def forward(self, pred_state, true_state, pred_ne, true_ne, blackout_mask):
        data_mse = self.mse(pred_state, true_state).mean(dim=-1)

        non_blackout_mask = ~blackout_mask
        if non_blackout_mask.any():
            l_data = data_mse[non_blackout_mask].mean()
        else:
            l_data = torch.tensor(0.0, device=pred_state.device)

        dv_dt = self.compute_derivatives(pred_state)
        l_momentum = torch.norm(dv_dt, p=2, dim=-1).mean()

        l_continuity = torch.norm(dv_dt, p=2, dim=-1).mean() * 0.1

        l_plasma = self.mse(pred_ne.squeeze(-1), true_ne).mean()

        total_loss = (
            l_data
            + self.lambda_momentum * l_momentum
            + self.lambda_continuity * l_continuity
            + self.lambda_plasma * l_plasma
        )

        return total_loss, {
            "l_data": l_data.item(),
            "l_momentum": l_momentum.item(),
            "l_continuity": l_continuity.item(),
            "l_plasma": l_plasma.item(),
        }
