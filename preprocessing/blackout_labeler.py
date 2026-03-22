"""
Sliding Window Blackout Sequence Labeler.

Identifies segments of severe signal degradation (blackout) in 
trajectory time-series data using electron density and SNR thresholds.
"""

import numpy as np
from typing import List, Tuple, Dict

class BlackoutLabeler:
    """
    Labels sequences as 'Clear', 'Blackout', or 'Re-Entry/Transition'.
    """
    
    def __init__(self, ne_threshold: float = 1e18, window_size: int = 5):
        """
        Initialize the labeler.
        
        Parameters:
            ne_threshold: Critical electron density.
            window_size: Window for smoothing blackout transitions.
        """
        self.ne_threshold = ne_threshold
        self.window = window_size

    def label_sequence(self, ne_data: np.ndarray) -> np.ndarray:
        """
        Generates binary labels (0 = clear, 1 = blackout).
        """
        # Basic binary mapping
        raw_labels = (ne_data > self.ne_threshold).astype(int)
        
        # Smooth peaks (prevent flickering)
        # We'll use a simple sliding window bitwise OR to ensure a robust blackout detection
        labels = np.copy(raw_labels)
        for i in range(len(labels)):
            w_start = max(0, i - self.window // 2)
            w_end = min(len(labels), i + self.window // 2 + 1)
            # If any in window is blackout, we consider it a 'risky' or blackout zone
            if np.any(raw_labels[w_start:w_end]):
                labels[i] = 1
                
        return labels

    def find_segments(self, labels: np.ndarray) -> List[Dict[str, int]]:
        """
        Identifies start and end indices of blackout segments.
        """
        segments = []
        in_segment = False
        start_idx = 0
        
        for i, val in enumerate(labels):
            if val == 1 and not in_segment:
                start_idx = i
                in_segment = True
            elif val == 0 and in_segment:
                segments.append({"start": start_idx, "end": i - 1})
                in_segment = False
                
        if in_segment:
            segments.append({"start": start_idx, "end": len(labels) - 1})
            
        return segments

if __name__ == "__main__":
    # Test Labeler
    labeler = BlackoutLabeler(ne_threshold=1.0)
    test_ne = np.array([0, 0, 1.1, 1.2, 1.1, 0, 0, 0.5, 1.1, 0.2, 0])
    
    labels = labeler.label_sequence(test_ne)
    segments = labeler.find_segments(labels)
    
    print(f"Labels: {labels}")
    print(f"Segments found: {segments}")
