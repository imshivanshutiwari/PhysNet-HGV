"""
Radar Detection and CFAR Visualization.

Visualizes radar signal strength, noise floor estimation, and 
CA-CFAR detection events throughout the simulation.
"""

import numpy as np
import matplotlib.pyplot as plt

class RadarVisualizer:
    """
    Visual analytics for radar detection performance.
    """
    
    def plot_detections(self, signal: np.ndarray, thresholds: np.ndarray, detections: np.ndarray):
        """
        Plots radar signal and CA-CFAR threshold.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(signal, label="Received Signal (Power)")
        plt.plot(thresholds, 'r--', label="CA-CFAR Threshold")
        
        # Mark detections
        det_indices = np.where(detections)[0]
        plt.scatter(det_indices, signal[det_indices], color='green', marker='x', label="Detections")
        
        plt.xlabel("Bin / Sample")
        plt.ylabel("Power")
        plt.title("Radar Detection Profile")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    print("RadarVisualizer initialized.")
