import torch
import numpy as np


def reacquire(ddpm_model, last_pre_blackout, pinn_exit_state, first_measurement):
    device = next(ddpm_model.parameters()).device

    condition = torch.cat(
        [
            torch.tensor(last_pre_blackout, dtype=torch.float32),
            torch.tensor(pinn_exit_state, dtype=torch.float32),
        ]
    ).to(device)

    with torch.no_grad():
        samples = ddpm_model.sample(condition, num_samples=500)

    samples_np = samples.cpu().numpy()

    lower_bound = np.percentile(samples_np, 2.5, axis=0)
    upper_bound = np.percentile(samples_np, 97.5, axis=0)
    ci_95 = (lower_bound, upper_bound)

    map_estimate = np.mean(samples_np, axis=0)

    fused_state = np.copy(map_estimate)
    if not np.isnan(first_measurement).any():
        fused_state[:6] = first_measurement

    return fused_state, ci_95
