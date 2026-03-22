"""
Radar Measurement Simulator with Plasma Interference.

Generates corrupted radar observations (Range, Azimuth, Elevation, Doppler)
based on the vehicle state and local plasma attenuation.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from .plasma_model import PlasmaModel

class RadarSimulator:
    """
    Simulates ground-based radar measurements with plasma-induced SNR degradation.
    """
    
    def __init__(self, config: Dict, plasma_model: PlasmaModel):
        """
        Initialize radar with sensor config and a plasma model instance.
        """
        self.config = config["sensor"]["radar"]
        self.plasma = plasma_model
        
        # Radar Position (ECEF) - Default to origin-centered or fixed
        self.pos_radar = np.array(self.config.get("position_ecef", [0, 0, 0]))
        
        # Noise Levels (Standard Deviations)
        self.sigma_range = self.config.get("noise_range", 50.0) # meters
        self.sigma_angle = np.deg2rad(self.config.get("noise_angle_deg", 0.05)) # radians
        self.sigma_doppler = self.config.get("noise_doppler", 5.0) # m/s
        
        # Nominal SNR
        self.snr_base_db = self.config.get("snr_base_db", 30.0)

    def get_measurement(self, state: np.ndarray, t: float) -> Tuple[np.ndarray, float, bool]:
        """
        Produces radar measurement vector [Range, Azimuth, Elevation, Doppler].
        
        Returns:
            z: Vector of measurements
            snr_eff: Effective SNR in dB
            is_blackout: True if SNR is below detection threshold
        """
        pos_hgv = state[0:3]
        vel_hgv = state[3:6]
        
        # 1. Relative Position (Radar -> HGV)
        rel_pos = pos_hgv - self.pos_radar
        range_true = np.linalg.norm(rel_pos)
        
        # 2. Geometry (AER)
        # Azimuth/Elevation in ENU or ECEF-local frame
        # For simplicity in ECEF frame (Relative to ECEF coords):
        az_true = np.arctan2(rel_pos[1], rel_pos[0])
        el_true = np.arcsin(rel_pos[2] / range_true) if range_true > 1e-3 else 0
        
        # 3. Radial Velocity (Doppler)
        # Project velocity onto range vector
        if range_true > 1e-3:
            unit_range = rel_pos / range_true
            v_radial_true = np.dot(vel_hgv, unit_range)
        else:
            v_radial_true = 0
            
        # 4. Plasma Effects: Stagnation Point Thermodynamics
        # Using aero-thermodynamic correlation: T_stag = T_inf * (1 + r * (gamma-1)/2 * M^2)
        # and P_stag from Rayleigh Pitot formula
        v_mag = np.linalg.norm(vel_hgv)
        alt = np.linalg.norm(pos_hgv) - 6378137.0
        
        # US Standard Atmosphere 1976 (Simplified but rigorous lookup)
        T_inf = 288.15 - 0.0065 * min(alt, 11000) # Basic troposphere/stratosphere
        P_inf = 101325.0 * (1 - 0.0065 * min(alt, 11000) / 288.15)**5.25
        
        mach = v_mag / np.sqrt(1.4 * 287.0 * T_inf) if T_inf > 0 else 1.0
        # Recovery factor r=0.9
        T_stag = T_inf * (1 + 0.18 * mach**2)
        # Rayleigh Pitot for Mach > 1
        if mach > 1.0:
            P_stag = P_inf * ((1.2 * mach**2)**3.5 / ( (7*mach**2 - 1)/6 )**2.5)
        else:
            P_stag = P_inf * (1 + 0.2 * mach**2)**3.5
            
        plasma_state = self.plasma.get_plasma_state(T_stag, P_stag)
        # Total attenuation depends on plasma layer thickness (~0.1m stagnation)
        attenuation_tot = plasma_state["attenuation_db_m"] * 0.1 
        
        # Effective SNR
        snr_eff = self.snr_base_db - attenuation_tot
        is_blackout = (snr_eff < 0) or plasma_state["is_blackout"]
        
        # 5. Noise Injection (Scale noise by signal loss)
        if not is_blackout:
            # Noise increases as SNR decreases
            noise_scale = 10 ** (-min(0, snr_eff) / 20.0)
            z_obs = np.array([
                range_true + np.random.normal(0, self.sigma_range * noise_scale),
                az_true + np.random.normal(0, self.sigma_angle * noise_scale),
                el_true + np.random.normal(0, self.sigma_angle * noise_scale),
                v_radial_true + np.random.normal(0, self.sigma_doppler * noise_scale)
            ])
        else:
            # Returned masked or null if total blackout (NaNs)
            z_obs = np.array([np.nan, np.nan, np.nan, np.nan])
            
        return z_obs, snr_eff, is_blackout

if __name__ == "__main__":
    # Standalone Test
    plasma_cfg = {"Ne_threshold": 1e18, "radar_frequency_ghz": 10.0}
    pm = PlasmaModel(plasma_cfg)
    
    radar_cfg = {
        "sensor": {
            "radar": {
                "noise_range": 5.0,
                "noise_angle_deg": 0.01,
                "noise_doppler": 1.0,
                "snr_base_db": 40.0,
                "position_ecef": [0, 0, 0]
            }
        }
    }
    
    radar = RadarSimulator(radar_cfg, pm)
    
    # State: 50km altitude, Mach 15 (ECEF)
    test_state = np.array([6378137.0 + 50000, 0, 0, 0, 5000.0, 0, 0, 0, 0, 0, 0, 0])
    
    z, snr, out = radar.get_measurement(test_state, 0.0)
    print(f"Radar Observation: {z}")
    print(f"Effective SNR: {snr:.2f} dB")
    print(f"Blackout: {out}")
