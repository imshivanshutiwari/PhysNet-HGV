import numpy as np


class RadarSimulator:
    def __init__(self, pos_noise=50.0, vel_noise=5.0, refresh_hz=10.0, range_km=800.0):
        self.pos_noise = pos_noise
        self.vel_noise = vel_noise
        self.refresh_hz = refresh_hz
        self.range_km = range_km
        self.dt = 1.0 / refresh_hz

    def add_noise(self, true_states):
        noisy_states = np.copy(true_states)
        pos_noise = np.random.normal(0, self.pos_noise, (len(true_states), 3))
        vel_noise = np.random.normal(0, self.vel_noise, (len(true_states), 3))

        noisy_states[:, :3] += pos_noise
        noisy_states[:, 3:] += vel_noise
        return noisy_states

    def generate_returns(self, true_trajectory, blackout_mask):
        noisy_trajectory = self.add_noise(true_trajectory)
        radar_returns = []

        for i, (state, blackout) in enumerate(zip(noisy_trajectory, blackout_mask)):
            if not blackout:
                radar_returns.append(state)
            else:
                radar_returns.append(np.full(6, np.nan))

        return np.array(radar_returns)
