import torch
import torch.nn as nn
from torchdiffeq import odeint


class ODEFunc(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=256, output_dim=12, activation=nn.SiLU):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, 128),
            activation(),
            nn.Linear(128, output_dim),
        )
        self.output_dim = output_dim

    def forward(self, t, state):
        t_tensor = torch.ones(state.shape[0], 1, device=state.device) * t

        if state.shape[1] == 12:
            x = torch.cat([state, t_tensor], dim=1)
        elif state.shape[1] == 13:
            x = state
        else:
            pad = torch.zeros(state.shape[0], 13 - state.shape[1], device=state.device)
            if 13 - state.shape[1] > 0:
                x = torch.cat([state, pad], dim=1)
            else:
                x = state[:, :13]

        out = self.net(x)
        return out


class NeuralODETracker(nn.Module):
    def __init__(self, method="dopri5"):
        super(NeuralODETracker, self).__init__()
        self.ode_func = ODEFunc()
        self.method = method

    def forward(self, state, t_span):
        if state.shape[1] < 12:
            pad = torch.zeros(state.shape[0], 12 - state.shape[1], device=state.device)
            state_padded = torch.cat([state, pad], dim=1)
        else:
            state_padded = state[:, :12]

        trajectory = odeint(self.ode_func, state_padded, t_span, method=self.method)
        trajectory = trajectory.permute(1, 0, 2)

        return trajectory
