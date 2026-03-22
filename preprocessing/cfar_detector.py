import numpy as np


class CFARDetector:
    def __init__(self, num_guard_cells=2, num_training_cells=10, pfa=1e-6):
        self.num_guard_cells = num_guard_cells
        self.num_training_cells = num_training_cells
        self.pfa = pfa
        N = 2 * self.num_training_cells
        self.alpha = N * (self.pfa ** (-1.0 / N) - 1)

    def detect(self, signal_power):
        N = len(signal_power)
        detections = np.zeros(N, dtype=bool)
        thresholds = np.zeros(N)

        for i in range(N):
            left_train_start = max(0, i - self.num_guard_cells - self.num_training_cells)
            left_train_end = max(0, i - self.num_guard_cells)
            right_train_start = min(N, i + self.num_guard_cells + 1)
            right_train_end = min(N, i + self.num_guard_cells + self.num_training_cells + 1)

            train_cells = []
            if left_train_end > left_train_start:
                train_cells.extend(signal_power[left_train_start:left_train_end])
            if right_train_end > right_train_start:
                train_cells.extend(signal_power[right_train_start:right_train_end])

            if len(train_cells) > 0:
                noise_estimate = np.mean(train_cells)
                thresholds[i] = noise_estimate * self.alpha
                if signal_power[i] > thresholds[i]:
                    detections[i] = True

        return detections, thresholds
