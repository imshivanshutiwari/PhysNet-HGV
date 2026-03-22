import torch
import torch.nn as nn


class PINNModule(nn.Module):
    def __init__(self, state_dim=6, hidden_dim=256, n_layers=6, activation="tanh", dt=0.1):
        super().__init__()
        self.state_dim = state_dim
        self.dt = dt

        if activation == "tanh":
            act_layer = nn.Tanh
        elif activation == "relu":
            act_layer = nn.ReLU
        elif activation == "silu":
            act_layer = nn.SiLU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(act_layer())

        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_layer())

        layers.append(nn.Linear(hidden_dim, state_dim + 1))

        self.net = nn.Sequential(*layers)

    def forward(self, state):
        out = self.net(state)
        state_pred = out[:, :, : self.state_dim] if out.dim() == 3 else out[:, : self.state_dim]
        ne_pred = torch.nn.functional.softplus(
            out[:, :, self.state_dim :] if out.dim() == 3 else out[:, self.state_dim :]
        )
        return state_pred, ne_pred

    def predict_blackout_state(self, current_state, duration_steps):
        self.eval()
        states = []
        state = current_state
        with torch.no_grad():
            for _ in range(duration_steps):
                state_pred, _ = self.forward(state)
                state = state_pred
                states.append(state)

        if len(states) > 0:
            return torch.stack(states, dim=1)
        else:
            return torch.empty(
                (current_state.shape[0], 0, self.state_dim), device=current_state.device
            )
