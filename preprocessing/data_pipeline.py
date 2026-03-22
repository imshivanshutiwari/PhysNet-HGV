import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class HGVTrajectoryDataset(Dataset):
    def __init__(self, trajectories, normalizer=None, seq_len=10, transform=None):
        self.data = []
        self.labels = []
        self.masks = []
        self.ne_profiles = []
        self.seq_len = seq_len
        self.normalizer = normalizer
        self.transform = transform

        for traj_dict in trajectories:
            true_traj = traj_dict["trajectory"]
            ne_profile = traj_dict["plasma_profile"]
            blackout_mask = traj_dict["blackout_mask"]

            if self.normalizer:
                normalized_traj = self.normalizer.transform(true_traj)
            else:
                normalized_traj = true_traj

            n_steps = len(normalized_traj)
            for i in range(0, n_steps - seq_len + 1):
                input_seq = normalized_traj[i : i + seq_len - 1]
                target_seq = normalized_traj[i + 1 : i + seq_len]
                mask_seq = blackout_mask[i : i + seq_len - 1]
                ne_seq = ne_profile[i : i + seq_len - 1]

                self.data.append(input_seq)
                self.labels.append(target_seq)
                self.masks.append(mask_seq)
                self.ne_profiles.append(ne_seq)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_seq = torch.tensor(self.data[idx], dtype=torch.float32)
        target_seq = torch.tensor(self.labels[idx], dtype=torch.float32)
        mask_seq = torch.tensor(self.masks[idx], dtype=torch.bool)
        ne_seq = torch.tensor(self.ne_profiles[idx], dtype=torch.float32)

        if self.transform:
            input_seq = self.transform(input_seq)

        return input_seq, target_seq, mask_seq, ne_seq


def get_dataloaders(
    train_trajectories, val_trajectories, batch_size=32, seq_len=10, normalizer=None
):
    train_dataset = HGVTrajectoryDataset(train_trajectories, normalizer, seq_len)
    val_dataset = HGVTrajectoryDataset(val_trajectories, normalizer, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
