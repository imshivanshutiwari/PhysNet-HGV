import numpy as np


class PlasmaModel:
    def __init__(self, blackout_threshold=1e18):
        self.blackout_threshold = blackout_threshold
        self.k_b = 1.380649e-23
        self.h = 6.62607015e-34
        self.m_e = 9.10938356e-31
        self.ionization_energy = 14.5 * 1.60218e-19

    def compute_electron_density(self, alt_m, vel_ms, temp_k=None):
        if temp_k is None:
            temp_k = 250 + (vel_ms**2) / 2000.0

        rho0 = 1.225
        H = 8500.0
        rho = rho0 * np.exp(-alt_m / H)

        m_air = 28.97 * 1.660539e-27
        n_neutral = rho / m_air

        temp_k = np.maximum(temp_k, 300.0)

        thermal_de_broglie_wavelength_cubed = (
            (self.h**2) / (2 * np.pi * self.m_e * self.k_b * temp_k)
        ) ** 1.5
        exponent = -self.ionization_energy / (self.k_b * temp_k)
        exponent = np.clip(exponent, -100, 100)

        saha_factor = (2.0 / thermal_de_broglie_wavelength_cubed) * np.exp(exponent)

        a = 1.0
        b = saha_factor
        c = -saha_factor * n_neutral

        discriminant = np.maximum(b**2 - 4 * a * c, 0.0)
        Ne = (-b + np.sqrt(discriminant)) / (2 * a)

        Ne_scaled = Ne * 1e10
        return Ne_scaled

    def compute_radar_attenuation(self, Ne, freq_hz):
        e = 1.60217663e-19
        epsilon_0 = 8.85418782e-12
        m_e = 9.10938356e-31

        wp_sq = Ne * (e**2) / (m_e * epsilon_0)
        wp = np.sqrt(np.maximum(wp_sq, 0.0))

        nu_c = 1e9
        w = 2 * np.pi * freq_hz
        c = 299792458.0

        attenuation_np = (nu_c / (2 * c)) * (wp_sq / (w**2 + nu_c**2))
        attenuation_db_per_m = attenuation_np * 8.686
        sheath_thickness = 0.1
        total_attenuation_db = attenuation_db_per_m * sheath_thickness

        return total_attenuation_db

    def is_blackout(self, Ne):
        return Ne >= self.blackout_threshold

    def get_blackout_duration(self, traj):
        if isinstance(traj, dict):
            if "ne" in traj:
                Ne_array = traj["ne"]
            else:
                alt = traj["altitude"]
                vel = traj["velocity"]
                Ne_array = self.compute_electron_density(alt, vel)
        else:
            Ne_array = traj

        blackout_mask = self.is_blackout(Ne_array)
        durations = []
        in_blackout = False
        start_idx = 0

        for i, b in enumerate(blackout_mask):
            if b and not in_blackout:
                in_blackout = True
                start_idx = i
            elif not b and in_blackout:
                in_blackout = False
                durations.append((start_idx, i - 1))

        if in_blackout:
            durations.append((start_idx, len(blackout_mask) - 1))

        return durations

    def generate_attenuation_profile(self, traj, freq_hz=10e9):
        if isinstance(traj, dict):
            if "ne" in traj:
                Ne_array = traj["ne"]
            else:
                alt = traj["altitude"]
                vel = traj["velocity"]
                Ne_array = self.compute_electron_density(alt, vel)
        else:
            Ne_array = traj

        return self.compute_radar_attenuation(Ne_array, freq_hz)
