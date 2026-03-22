import numpy as np


class BlackoutLabeler:
    def __init__(self, blackout_threshold=1e18):
        self.blackout_threshold = blackout_threshold

    def label_trajectory(self, ne_profile):
        return np.array(ne_profile) >= self.blackout_threshold

    def label_batch(self, ne_profiles):
        return [self.label_trajectory(profile) for profile in ne_profiles]
