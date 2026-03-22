"""
Cell Averaging Constant False Alarm Rate (CA-CFAR) Detector.

Implements adaptive thresholding to detect targets in varying 
background noise/clutter environments for radar signals.
"""

import numpy as np
from typing import Tuple, List

class CFARDetector:
    """
    1D CA-CFAR Detector for radar signal processing.
    """
    
    def __init__(self, guard_cells: int = 2, training_cells: int = 10, pfa: float = 1e-4):
        """
        Initialize the CFAR detector.
        
        Parameters:
            guard_cells: Number of cells around the CUT (Cell Under Test) to ignore.
            training_cells: Number of cells to average for noise estimation.
            pfa: Probability of False Alarm (sets the threshold factor).
        """
        self.guard = guard_cells
        self.train = training_cells
        self.pfa = pfa
        
        # Scaling factor for CA-CFAR: N * (Pfa^(-1/N) - 1)
        # For N training cells
        n_total = 2 * training_cells
        self.alpha = n_total * (pfa**(-1.0 / n_total) - 1)

    def detect(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform CA-CFAR detection on a 1D signal (e.g., range-doppler bin).
        
        Returns:
            detections: Boolean mask of detected targets
            thresholds: The adaptive threshold values
        """
        n = len(signal)
        detections = np.zeros(n, dtype=bool)
        thresholds = np.zeros(n)
        
        # Sliding window
        # Window size = 2 * (train + guard) + 1 (for CUT)
        for i in range(self.train + self.guard, n - (self.train + self.guard)):
            # Define training regions
            left_train = signal[i - self.train - self.guard : i - self.guard]
            right_train = signal[i + self.guard + 1 : i + self.guard + 1 + self.train]
            
            # Average noise level (Power)
            noise_level = (np.mean(left_train) + np.mean(right_train)) / 2.0
            
            # Adaptive Threshold
            threshold = noise_level * self.alpha
            thresholds[i] = threshold
            
            if signal[i] > threshold:
                detections[i] = True
                
        return detections, thresholds

if __name__ == "__main__":
    # Test CFAR
    detector = CFARDetector(guard_cells=2, training_cells=10, pfa=1e-3)
    
    # Generate noisy signal with 2 targets
    signal = np.random.exponential(1.0, 200) # Exponential noise (Power)
    signal[50] = 50.0 # Clear target
    signal[120] = 15.0 # Weak target
    
    detections, thresholds = detector.detect(signal)
    
    print(f"Detected targets at indices: {np.where(detections)[0]}")
    print(f"Target at 50 detected: {detections[50]}")
    print(f"Target at 120 detected: {detections[120]}")
