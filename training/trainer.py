import torch
import wandb
from tqdm import tqdm
from evaluation.metrics import compute_metrics_batch


class PINNTrainer:
    def __init__(self, model, optimizer, criterion, device="cuda"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        logs = {}
        for inputs, targets, masks, ne_profiles in tqdm(dataloader, desc="Training"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            masks, ne_profiles = masks.to(self.device), ne_profiles.to(self.device)

            self.optimizer.zero_grad()
            state_pred, ne_pred = self.model(inputs)

            loss, metrics = self.criterion(state_pred, targets, ne_pred, ne_profiles, masks)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            for k, v in metrics.items():
                logs[k] = logs.get(k, 0) + v * inputs.size(0)

        num_samples = len(dataloader.dataset)
        return {k: v / num_samples for k, v in logs.items()}

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets, masks, ne_profiles in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                masks, ne_profiles = masks.to(self.device), ne_profiles.to(self.device)

                state_pred, ne_pred = self.model(inputs)
                loss, _ = self.criterion(state_pred, targets, ne_pred, ne_profiles, masks)
                total_loss += loss.item() * inputs.size(0)

        return total_loss / len(dataloader.dataset)
