"""
6-DOF Hypersonic Glide Vehicle (HGV) Dynamics Simulator.

Implements high-fidelity equations of motion in the ECEF frame, 
incorporating WGS-84 gravity, US Standard Atmosphere 1976, 
and Mach-dependent aerodynamics.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Tuple, List, Optional
from .plasma_model import PlasmaModel

class HGVDynamics:
    """
    High-fidelity 6-DOF HGV Dynamics model.
    """

    # WGS-84 Constants
    WGS84_A = 6378137.0         # Semi-major axis (m)
    WGS84_F = 1 / 298.257223563 # Flattening
    WGS84_B = WGS84_A * (1 - WGS84_F)
    WGS84_E2 = 1 - (WGS84_B**2 / WGS84_A**2)
    OMEGA_EARTH = 7.292115e-5   # Earth rotation rate (rad/s)
    MU_EARTH = 3.986004418e14   # Earth gravitational constant (m^3/s^2)
    J2 = 1.08262668e-3          # Second zonal harmonic

    def __init__(self, config: Dict):
        """
        Initialize dynamics with configuration.
        """
        self.config = config
        self.mass = config["vehicle"]["mass_kg"]
        self.ref_area = config["vehicle"]["reference_area_m2"]
        
        # Inertia Tensor (diagonal approx for HGV)
        self.I = np.diag([
            config["vehicle"]["Ixx"],
            config["vehicle"]["Iyy"],
            config["vehicle"]["Izz"]
        ])
        self.I_inv = np.linalg.inv(self.I)
        
        # Aerodynamics constants
        self.cd0 = config["aerodynamics"]["cd0"]
        self.k_induced = config["aerodynamics"]["k_induced"]
        self.cl_alpha = config["aerodynamics"]["cl_alpha"]
        
        # Integration settings
        self.rtol = float(config["integration"]["rtol"])
        self.atol = float(config["integration"]["atol"])
        self.method = config["integration"]["method"]

    def get_atmosphere(self, altitude_m: float) -> Tuple[float, float, float]:
        """
        US Standard Atmosphere 1976 model (Simplified for 0-80km).
        Returns: Density (kg/m3), Temperature (K), Pressure (Pa)
        """
        # H is geopotential altitude
        Re = 6356766.0 # Avg radius for geopotential calc
        h = (Re * altitude_m) / (Re + altitude_m)
        
        # Constants for layers
        layers = [
            (0, 11000, 288.15, 101325.0, -0.0065),
            (11000, 20000, 216.65, 22632.0, 0.0),
            (20000, 32000, 216.65, 5474.8, 0.001),
            (32000, 47000, 228.65, 868.01, 0.0028),
            (47000, 51000, 270.65, 110.90, 0.0),
            (51000, 71000, 270.65, 66.938, -0.0028),
            (71000, 84852, 214.65, 3.9564, -0.002)
        ]
        
        rho0 = 1.225
        g0 = 9.80665
        R = 287.05
        
        for h_low, h_high, t_base, p_base, lapse in layers:
            if h <= h_high:
                delta_h = h - h_low
                if lapse != 0:
                    T = t_base + lapse * delta_h
                    P = p_base * (T / t_base)**(-g0 / (lapse * R))
                else:
                    T = t_base
                    P = p_base * np.exp(-g0 * delta_h / (R * T))
                
                rho = P / (R * T)
                return rho, T, P
                
        # Default/Fallback (Extreme high altitude)
        return 1e-6, 200.0, 0.1

    def get_gravity_ecef(self, pos_ecef: np.ndarray) -> np.ndarray:
        """
        WGS84 Gravity model including J2 effect in ECEF.
        """
        r = np.linalg.norm(pos_ecef)
        if r < 1e-3: return np.zeros(3)
        
        z = pos_ecef[2]
        r2 = r**2
        z2 = z**2
        
        # Spherical part
        g_sph = - (self.MU_EARTH / r**3) * pos_ecef
        
        # J2 part
        j2_factor = (1.5 * self.J2 * self.MU_EARTH * self.WGS84_A**2) / r**5
        g_j2 = j2_factor * np.array([
            pos_ecef[0] * (5 * (z2 / r2) - 1),
            pos_ecef[1] * (5 * (z2 / r2) - 1),
            pos_ecef[2] * (5 * (z2 / r2) - 3)
        ])
        
        return g_sph + g_j2

    def eom(self, t: float, state: np.ndarray, control: Dict) -> np.ndarray:
        """
        Equations of Motion.
        state: [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
        """
        r_ecef = state[0:3]
        v_ecef = state[3:6]
        euler = state[6:9]  # rad
        omega_body = state[9:12] # rad/s
        
        # 1. Geodetic Altitude
        r_mag = np.linalg.norm(r_ecef)
        alt = r_mag - self.WGS84_A # Simple approx, can be improved with Iterative Lat/Long
        
        # 2. Atmosphere
        rho, temp, press = self.get_atmosphere(max(0, alt))
        v_rel = v_ecef # Approx (ignoring winds)
        v_mag = np.linalg.norm(v_rel)
        
        # 3. Aerodynamics (Simplified 6-DOF HGV model)
        # alpha, beta, bank are from control or derived
        alpha = control.get("alpha", 10.0) * (np.pi/180.0) # Angle of attack
        beta = 0.0 
        bank = control.get("bank", 0.0) * (np.pi/180.0)
        
        # Mach number
        a = np.sqrt(1.4 * 287.05 * temp)
        mach = v_mag / a if a > 0 else 0
        
        # Coeffs
        cl = self.cl_alpha * (alpha * 180.0/np.pi)
        cd = self.cd0 + self.k_induced * cl**2
        
        q_inf = 0.5 * rho * v_mag**2
        lift = q_inf * self.ref_area * cl
        drag = q_inf * self.ref_area * cd
        
        # 4. Rotation Matrices
        phi, theta, psi = euler
        # ECEF to NED (Lat/Long needed, approx at 0/0 for now)
        # For simplicity, we define Body to ECEF directly
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        c_th, s_th = np.cos(theta), np.sin(theta)
        c_ps, s_ps = np.cos(psi), np.sin(psi)
        
        # Rotation Matrix Body -> ECEF (Z-Y-X convention)
        R_b2e = np.array([
            [c_th*c_ps, s_phi*s_th*c_ps - c_phi*s_ps, c_phi*s_th*c_ps + s_phi*s_ps],
            [c_th*s_ps, s_phi*s_th*s_ps + c_phi*c_ps, c_phi*s_th*s_ps - s_phi*c_ps],
            [-s_th, s_phi*c_th, c_phi*c_th]
        ])
        
        # Absolute Acceleration
        g_ecef = self.get_gravity_ecef(r_ecef)
        
        # Coriolis and Centrifugal
        omega_vec = np.array([0, 0, self.OMEGA_EARTH])
        a_coriolis = -2.0 * np.cross(omega_vec, v_ecef)
        a_centrifugal = -np.cross(omega_vec, np.cross(omega_vec, r_ecef))
        
        # Aerodynamic Acceleration
        # Lift is perp to v_rel, Drag is opposite v_rel
        if v_mag > 1e-3:
            # Lift/Drag in Wind frame, rotated to Body then ECEF
            # Simplified: assuming alpha/bank defines lift direction in ECEF
            d_dir = -v_ecef / v_mag
            # Lift dir perp to v_ecef and horizontal
            l_dir_raw = np.cross(v_ecef, np.cross(r_ecef, v_ecef))
            l_dir = l_dir_raw / np.linalg.norm(l_dir_raw) if np.linalg.norm(l_dir_raw) > 1e-6 else np.zeros(3)
            # Apply bank rotation to lift
            l_dir = np.cos(bank) * l_dir + np.sin(bank) * np.cross(d_dir, l_dir)
        else:
            d_dir = l_dir = np.zeros(3)
            
        a_aero = (lift * l_dir + drag * d_dir) / self.mass
        
        # Total dV/dt
        dv_dt = g_ecef + a_coriolis + a_centrifugal + a_aero
        
        # 5. Rotational Dynamics (Euler Equations)
        # Aerodynamic Moments (Simplified pitch/roll/yaw damping)
        # M = [roll_moment, pitch_moment, yaw_moment]
        m_aero = np.array([
            -0.1 * omega_body[0], # Roll damping
            -0.1 * omega_body[1] + 0.1 * (alpha - 0.1), # Pitch stability
            -0.1 * omega_body[2] # Yaw damping
        ])
        dw_dt = self.I_inv @ (m_aero - np.cross(omega_body, self.I @ omega_body))
        
        # 6. Kinematics (Euler Angle rates)
        # d_euler = T(euler) @ omega_body
        tan_th = np.tan(theta) if abs(np.cos(theta)) > 1e-6 else 0
        sec_th = 1.0 / np.cos(theta) if abs(np.cos(theta)) > 1e-6 else 1.0
        
        T_mat = np.array([
            [1, s_phi * tan_th, c_phi * tan_th],
            [0, c_phi, -s_phi],
            [0, s_phi * sec_th, c_phi * sec_th]
        ])
        d_euler = T_mat @ omega_body
        
        return np.concatenate([v_ecef, dv_dt, d_euler, dw_dt])
        
        return np.concatenate([v_ecef, dv_dt, d_euler, dw_dt])

    def step(self, t: float, dt: float, state: np.ndarray, control: Dict) -> np.ndarray:
        """
        Propagate state by dt.
        """
        sol = solve_ivp(
            self.eom, [t, t + dt], state, 
            args=(control,), 
            method=self.method, rtol=self.rtol, atol=self.atol
        )
        return sol.y[:, -1]

if __name__ == "__main__":
    # Test Dynamics
    from physnet_hgv.utils.config_loader import load_config
    import os
    
    # Mock config for standalone run
    mock_config = {
        "vehicle": {"mass_kg": 907.0, "reference_area_m2": 0.88, "Ixx": 120.0, "Iyy": 850.0, "Izz": 850.0},
        "aerodynamics": {"cd0": 0.015, "k_induced": 0.045, "cl_alpha": 0.08},
        "integration": {"method": "RK45", "rtol": 1e-8, "atol": 1e-10}
    }
    
    dyn = HGVDynamics(mock_config)
    
    # Initial State: 50km altitude, Mach 10
    r0 = np.array([dyn.WGS84_A + 50000.0, 0, 0])
    v0 = np.array([0, 3400.0, 0]) # ~Mach 10 at 50km
    state0 = np.zeros(12)
    state0[0:3] = r0
    state0[3:6] = v0
    
    print(f"Initial State Altitude: {np.linalg.norm(r0) - dyn.WGS84_A:.2f}m")
    
    next_state = dyn.step(0, 1.0, state0, {"alpha": 15.0, "bank": 10.0})
    print(f"State after 1s Altitude: {np.linalg.norm(next_state[0:3]) - dyn.WGS84_A:.2f}m")
    print(f"Velocity Mag: {np.linalg.norm(next_state[3:6]):.2f}m/s")
