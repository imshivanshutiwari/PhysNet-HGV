import numpy as np


class AdaptiveNoiseEstimator:
    def __init__(self, state_dim, fading_factor=0.98):
        self.state_dim = state_dim
        self.fading_factor = fading_factor
        self.residual_cov = np.eye(state_dim)

    def update(self, residual):
        residual = np.atleast_2d(residual)
        self.residual_cov = self.fading_factor * self.residual_cov + (1 - self.fading_factor) * (
            residual.T @ residual
        )
        return self.residual_cov

    def get_fading_Q(self, Q_base, scale=1.0):
        return Q_base * scale
