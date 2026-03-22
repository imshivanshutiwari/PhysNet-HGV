"""
Plasma Blackout and Electron Density Visualization.

Generates temporal heatmaps and profile plots for electron density (Ne) 
and signal attenuation across the HGV flight profile.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List

class BlackoutVisualizer:
    """
    Visual analytics for plasma blackout analysis.
    """
    
    def plot_ne_profile(self, time: np.ndarray, ne_data: np.ndarray, threshold: float = 1e18):
        """
        Plots electron density profile with blackout threshold.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(time, ne_data, color='purple', linewidth=2, label="Electron Density (Ne)")
        plt.axhline(y=threshold, color='red', linestyle='--', label="Blackout Threshold")
        
        # Shade blackout regions
        mask = ne_data > threshold
        plt.fill_between(time, 0, ne_data, where=mask, color='red', alpha=0.2, label="Blackout Zone")
        
        plt.yscale('log')
        plt.xlabel("Time (s)")
        plt.ylabel("Ne (m^-3)")
        plt.title("HGV Plasma Profile")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.show()

if __name__ == "__main__":
    viz = BlackoutVisualizer()
    t = np.linspace(0, 100, 1000)
    ne = 1e17 + 1e18 * np.exp(-(t-50)**2 / 100)
    # viz.plot_ne_profile(t, ne)
    print("BlackoutVisualizer initialized.")
