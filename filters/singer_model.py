import numpy as np


def create_singer_model(dt, alpha, sigma_m2):
    F = np.array(
        [
            [1, dt, (np.exp(-alpha * dt) + alpha * dt - 1) / alpha**2],
            [0, 1, (1 - np.exp(-alpha * dt)) / alpha],
            [0, 0, np.exp(-alpha * dt)],
        ]
    )

    q11 = (
        1
        - np.exp(-2 * alpha * dt)
        + 2 * alpha * dt
        + 2 * alpha**3 * dt**3 / 3
        - 2 * alpha**2 * dt**2
        - 4 * alpha * dt * np.exp(-alpha * dt)
    ) / alpha**4
    q12 = (
        np.exp(-2 * alpha * dt)
        + 1
        - 2 * np.exp(-alpha * dt)
        + 2 * alpha * dt * np.exp(-alpha * dt)
        - 2 * alpha * dt
        + alpha**2 * dt**2
    ) / alpha**3
    q13 = (1 - np.exp(-2 * alpha * dt) - 2 * alpha * dt * np.exp(-alpha * dt)) / alpha**2

    q21 = q12
    q22 = (4 * np.exp(-alpha * dt) - 3 - np.exp(-2 * alpha * dt) + 2 * alpha * dt) / alpha**2
    q23 = (np.exp(-2 * alpha * dt) + 1 - 2 * np.exp(-alpha * dt)) / alpha

    q31 = q13
    q32 = q23
    q33 = 1 - np.exp(-2 * alpha * dt)

    Q = 2 * alpha * sigma_m2 * np.array([[q11, q12, q13], [q21, q22, q23], [q31, q32, q33]])

    return F, Q


def create_singer_3d(dt, alpha, sigma_m2):
    F_1d, Q_1d = create_singer_model(dt, alpha, sigma_m2)

    F = np.zeros((9, 9))
    Q = np.zeros((9, 9))

    for i in range(3):
        for j in range(3):
            for k in range(3):
                F[i * 3 + k, j * 3 + k] = F_1d[i, j]
                Q[i * 3 + k, j * 3 + k] = Q_1d[i, j]

    return F, Q
