import numpy as np
import scipy.linalg
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from .singer_model import create_singer_3d


class UKFTracker:
    def __init__(
        self,
        dt=0.1,
        alpha=0.001,
        kappa=0,
        beta=2,
        singer_alpha=0.05,
        sigma_m2=100.0,
        fading_factor=0.98,
    ):
        self.dt = dt
        self.state_dim = 9
        self.meas_dim = 6

        self.fading_factor = fading_factor

        self.points = MerweScaledSigmaPoints(self.state_dim, alpha=alpha, beta=beta, kappa=kappa)
        self.F, self.Q = create_singer_3d(dt, singer_alpha, sigma_m2)

        def fx(x, dt):
            return self.F @ x

        def hx(x):
            return x[:6]

        self.ukf = UnscentedKalmanFilter(
            dim_x=self.state_dim, dim_z=self.meas_dim, dt=dt, fx=fx, hx=hx, points=self.points
        )

        self.ukf.Q = self.Q
        self.ukf.R = np.eye(self.meas_dim) * 100.0
        self.ukf.P = np.diag([100, 100, 100, 10, 10, 10, 1, 1, 1])

    def reset(self, initial_state, initial_P=None):
        if initial_state.shape == (6,):
            self.ukf.x = np.concatenate([initial_state, np.zeros(3)])
        else:
            self.ukf.x = initial_state

        if initial_P is not None:
            self.ukf.P = initial_P
        else:
            self.ukf.P = np.diag([100, 100, 100, 10, 10, 10, 1, 1, 1])

    def predict(self):
        self.ukf.predict()
        self.ukf.P = self.ukf.P / self.fading_factor

    def update(self, measurement, R=None):
        if R is not None:
            self.ukf.update(measurement, R=R)
        else:
            self.ukf.update(measurement)

    def update_pinn_blackout(self, pinn_prediction):
        pseudo_measurement = pinn_prediction[:6]
        pseudo_R = np.eye(self.meas_dim) * 50.0
        self.update(pseudo_measurement, R=pseudo_R)

    def run_filter(self, trajectory_measurements, blackout_mask, pinn_model=None):
        num_steps = len(trajectory_measurements)
        estimates = np.zeros((num_steps, self.state_dim))
        covariances = np.zeros((num_steps, self.state_dim, self.state_dim))

        if not blackout_mask[0]:
            self.reset(trajectory_measurements[0])
            estimates[0] = self.ukf.x
            covariances[0] = self.ukf.P
        else:
            self.reset(np.zeros(6))

        for i in range(1, num_steps):
            self.predict()

            is_blackout = blackout_mask[i]
            meas = trajectory_measurements[i]

            if not is_blackout and not np.isnan(meas).all():
                self.update(meas)
            else:
                if pinn_model is not None:
                    import torch

                    with torch.no_grad():
                        state_tensor = torch.tensor(self.ukf.x[:6], dtype=torch.float32).unsqueeze(
                            0
                        )
                        pinn_pred, _ = pinn_model(state_tensor)
                        pinn_pred = pinn_pred.squeeze().numpy()
                    self.update_pinn_blackout(pinn_pred)

            estimates[i] = self.ukf.x
            covariances[i] = self.ukf.P

        return estimates, covariances
