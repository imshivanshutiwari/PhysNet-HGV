import numpy as np
from scipy.integrate import solve_ivp


class HGVDynamics:
    def __init__(self, mass_kg=907.0, ref_area_m2=0.88, dt=0.1):
        self.mass = mass_kg
        self.S = ref_area_m2
        self.dt = dt
        self.Re = 6378137.0
        self.g0 = 9.80665
        self.omega_e = 7.292115e-5

    def _atmosphere(self, altitude):
        h = np.maximum(0, altitude)
        if h < 11000:
            T = 288.15 - 0.0065 * h
            p = 101325 * (288.15 / T) ** -5.256
        elif h < 20000:
            T = 216.65
            p = 22632 * np.exp(-0.000157 * (h - 11000))
        elif h < 32000:
            T = 216.65 + 0.001 * (h - 20000)
            p = 5474 * (216.65 / T) ** 34.163
        elif h < 47000:
            T = 228.65 + 0.0028 * (h - 32000)
            p = 868 * (228.65 / T) ** 12.201
        elif h < 51000:
            T = 270.65
            p = 110.9 * np.exp(-0.000126 * (h - 47000))
        elif h < 71000:
            T = 270.65 - 0.0028 * (h - 51000)
            p = 66.9 * (270.65 / T) ** -12.201
        else:
            T = 214.65 - 0.002 * (h - 71000)
            p = 3.96 * (214.65 / T) ** -17.082

        rho = p / (287.05 * T)
        a = np.sqrt(1.4 * 287.05 * T)
        return rho, a

    def _aerodynamics(self, mach, alpha_deg):
        alpha = np.radians(alpha_deg)
        Cd = 0.05 + 1.2 * np.sin(alpha) ** 3
        Cl = 1.2 * np.sin(alpha) ** 2 * np.cos(alpha) * np.sign(alpha)
        Cd *= 1 + 1 / np.maximum(mach, 0.1)
        Cl *= 1 + 1 / np.maximum(mach, 0.1)
        return Cl, Cd

    def compute_mach(self, v_mag, altitude):
        _, a = self._atmosphere(altitude)
        return v_mag / a

    def equations_of_motion(self, t, state, control):
        x, y, z, vx, vy, vz = state
        r_vec = np.array([x, y, self.Re + z])
        r = np.linalg.norm(r_vec)
        altitude = r - self.Re

        v_vec = np.array([vx, vy, vz])
        v_mag = np.linalg.norm(v_vec)

        if altitude < 0 or v_mag < 1.0:
            return np.zeros(6)

        rho, a = self._atmosphere(altitude)
        mach = v_mag / a

        alpha_deg, bank_deg = control
        Cl, Cd = self._aerodynamics(mach, alpha_deg)

        q = 0.5 * rho * v_mag**2
        L = q * self.S * Cl
        D = q * self.S * Cd

        u_v = v_vec / v_mag
        u_up = r_vec / r

        u_h = np.cross(u_up, np.cross(v_vec, u_up))
        norm_h = np.linalg.norm(u_h)
        if norm_h > 1e-6:
            u_h /= norm_h
        else:
            u_h = np.array([1, 0, 0])

        u_c = np.cross(u_v, u_up)
        norm_c = np.linalg.norm(u_c)
        if norm_c > 1e-6:
            u_c /= norm_c
        else:
            u_c = np.array([0, 1, 0])

        bank = np.radians(bank_deg)
        u_L = u_up * np.cos(bank) + u_c * np.sin(bank)

        u_L = u_L - np.dot(u_L, u_v) * u_v
        norm_L = np.linalg.norm(u_L)
        if norm_L > 1e-6:
            u_L /= norm_L

        F_aero = -D * u_v + L * u_L
        g = self.g0 * (self.Re / r) ** 2
        F_grav = -self.mass * g * u_up

        omega_vec = np.array([0, 0, self.omega_e])
        F_cor = -2 * self.mass * np.cross(omega_vec, v_vec)
        F_cent = -self.mass * np.cross(omega_vec, np.cross(omega_vec, r_vec))

        a_vec = (F_aero + F_grav + F_cor + F_cent) / self.mass

        return np.concatenate((v_vec, a_vec))

    def integrate_trajectory(self, initial_state, duration_s, controls):
        t_span = (0, duration_s)
        t_eval = np.arange(0, duration_s, self.dt)

        def control_interp(t):
            return controls[0][1:]

        def ode_func(t, y):
            ctrl = control_interp(t)
            return self.equations_of_motion(t, y, ctrl)

        sol = solve_ivp(ode_func, t_span, initial_state, method="RK45", t_eval=t_eval)
        return sol.y.T

    def add_measurement_noise(self, trajectory, pos_std=10.0, vel_std=1.0):
        noise = np.random.randn(*trajectory.shape)
        noise[:, :3] *= pos_std
        noise[:, 3:] *= vel_std
        return trajectory + noise
