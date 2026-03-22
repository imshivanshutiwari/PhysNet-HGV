import torch
import torch.nn as nn
import torch.optim as optim


class DDPMTrainer:
    def __init__(self, model, lr=2e-4, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_step(self, x0, condition):
        self.model.train()
        x0, condition = x0.to(self.device), condition.to(self.device)

        self.optimizer.zero_grad()
        noise_pred, noise_true = self.model(x0, condition)
        loss = self.criterion(noise_pred, noise_true)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x0, condition in dataloader:
                x0, condition = x0.to(self.device), condition.to(self.device)
                noise_pred, noise_true = self.model(x0, condition)
                loss = self.criterion(noise_pred, noise_true)
                total_loss += loss.item() * x0.size(0)

        return total_loss / len(dataloader.dataset)
