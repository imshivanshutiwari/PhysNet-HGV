import numpy as np


def covariance_intersection(means, covariances, omegas=None):
    N = len(means)
    dim = len(means[0])

    if omegas is None:
        omegas = np.ones(N) / N

    P_inv_sum = np.zeros((dim, dim))
    x_sum = np.zeros(dim)

    for w, mu, cov in zip(omegas, means, covariances):
        cov_inv = np.linalg.inv(cov + np.eye(dim) * 1e-6)
        P_inv_sum += w * cov_inv
        x_sum += w * (cov_inv @ mu)

    P_fused = np.linalg.inv(P_inv_sum + np.eye(dim) * 1e-6)
    mu_fused = P_fused @ x_sum

    return mu_fused, P_fused
