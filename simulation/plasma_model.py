"""
Physics-Informed Plasma Blackout Model for Hypersonic Glide Vehicles.

Implements the Saha ionization equation and wave attenuation models to 
predict electron density and signal degradation during high-mach flight.
"""

import numpy as np
from typing import Dict, Tuple, Union

class PlasmaModel:
    """
    Advanced Plasma Model implementing the Saha equation and electromagnetic
    wave attenuation for hypersonic flow conditions.
    """

    # Physical Constants
    K_BOLTZMANN = 1.380649e-23  # J/K
    H_PLANCK = 6.62607015e-34    # J·s
    M_ELECTRON = 9.10938356e-31  # kg
    E_CHARGE = 1.60217663e-19    # C
    EPSILON_0 = 8.85418781e-12   # F/m
    C_LIGHT = 299792458.0        # m/s
    
    # Air Properties (Nitrogen Dominated)
    E_IONIZATION_N2 = 15.6 * 1.60217663e-19  # eV to Joules (approx 15.6 eV)
    
    def __init__(self, config: Dict):
        """
        Initialize the plasma model with configuration parameters.
        
        Parameters
        ----------
        config : Dict
            Dictionary containing 'Ne_threshold' and other plasma parameters.
        """
        self.ne_threshold = config.get("Ne_threshold", 1e18)
        self.freq_band_ghz = config.get("radar_frequency_ghz", 10.0)
        self.omega = 2 * np.pi * self.freq_band_ghz * 1e9

    def calculate_electron_density(self, temperature: float, pressure: float) -> float:
        """
        Calculate electron density (Ne) using the Saha Ionization Equation.
        
        Ne = sqrt( (2 * P / kT) * (2 * pi * me * kT / h^2)^(3/2) * exp(-Ei / kT) )
        
        Parameters
        ----------
        temperature : float
            Stagnation temperature in Kelvin (T).
        pressure : float
            Stagnation pressure in Pascals (P).
            
        Returns
        -------
        float
            Electron density in m^-3.
        """
        # Thermal energy
        kT = self.K_BOLTZMANN * temperature
        
        # Saha part 1: (2 * pi * m_e * k * T / h^2)^(3/2)
        de_broglie_term = (2 * np.pi * self.M_ELECTRON * kT / (self.H_PLANCK**2))**1.5
        
        # Saha part 2: exp(-Ei / kT)
        exp_term = np.exp(-self.E_IONIZATION_N2 / kT)
        
        # Saha part 3: Particle density (approximate for weakly ionized gas)
        # n = P / (k * T)
        n_total = pressure / kT
        
        # Ne ≈ sqrt(n_total * 2 * de_broglie_term * exp_term)
        # Note: This is an approximation for single ionization of a single species (N2)
        ne = np.sqrt(2 * n_total * de_broglie_term * exp_term)
        
        return float(ne)

    def calculate_attenuation(self, ne: float) -> float:
        """
        Calculate attenuation (alpha_dB) in dB/m.
        
        alpha_dB = (Ne * e^2) / (2 * epsilon_0 * me * c * omega) * (8.686)
        
        Parameters
        ----------
        ne : float
            Electron density in m^-3.
            
        Returns
        -------
        float
            Attenuation in dB/m.
        """
        # Physical attenuation coefficient (alpha)
        # alpha = (ne * e^2) / (2 * epsilon_0 * m_e * c * omega)
        numerator = ne * (self.E_CHARGE**2)
        denominator = 2 * self.EPSILON_0 * self.M_ELECTRON * self.C_LIGHT * self.omega
        
        alpha = numerator / denominator
        
        # Convert to dB/meter (1 Np = 8.686 dB)
        alpha_db = alpha * 8.68588
        
        return float(alpha_db)

    def is_blackout(self, ne: float) -> bool:
        """
        Check if the blackout condition is met.
        
        Parameters
        ----------
        ne : float
            Electron density in m^-3.
            
        Returns
        -------
        bool
            True if ne > threshold.
        """
        return ne > self.ne_threshold

    def get_plasma_state(self, temperature: float, pressure: float) -> Dict[str, Union[float, bool]]:
        """
        Compute full plasma state given local atmospheric conditions.
        """
        ne = self.calculate_electron_density(temperature, pressure)
        att = self.calculate_attenuation(ne)
        blackout = self.is_blackout(ne)
        
        return {
            "electron_density": ne,
            "attenuation_db_m": att,
            "is_blackout": blackout
        }

if __name__ == "__main__":
    # Verification Test
    config = {"Ne_threshold": 1e18, "radar_frequency_ghz": 10.0}
    model = PlasmaModel(config)
    
    # Sample Hypersonic Conditions (Stagnation Point)
    # T ~ 4000K, P ~ 100kPar
    test_t = 4000.0
    test_p = 101325.0
    
    state = model.get_plasma_state(test_t, test_p)
    print(f"Plasma State at {test_t}K, {test_p}Pa:")
    for k, v in state.items():
        print(f"  {k}: {v:.4e}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Check threshold
    ne_high = 2e18
    print(f"Is blackout at {ne_high:.1e}?: {model.is_blackout(ne_high)}")
