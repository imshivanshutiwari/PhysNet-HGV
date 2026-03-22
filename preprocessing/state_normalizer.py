import numpy as np


class StateNormalizer:
    def __init__(self, state_dim=6):
        self.state_dim = state_dim
        self.mean = np.zeros(state_dim)
        self.std = np.ones(state_dim)

    def fit(self, trajectories):
        all_states = np.concatenate(trajectories, axis=0)
        self.mean = np.nanmean(all_states, axis=0)
        self.std = np.nanstd(all_states, axis=0)
        self.std[self.std == 0] = 1e-6

    def transform(self, trajectory):
        return (trajectory - self.mean) / self.std

    def inverse_transform(self, normalized_trajectory):
        return (normalized_trajectory * self.std) + self.mean

    def normalize_batch(self, trajectories):
        return [self.transform(traj) for traj in trajectories]

    def denormalize_batch(self, normalized_trajectories):
        return [self.inverse_transform(traj) for traj in normalized_trajectories]
