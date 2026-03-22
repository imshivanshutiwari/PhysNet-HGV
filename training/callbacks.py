import os
import torch


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class ModelCheckpoint:
    def __init__(self, filepath, monitor="val_loss", mode="min"):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else -float("inf")

    def __call__(self, current_value, model, optimizer, epoch, metrics):
        if self.mode == "min" and current_value < self.best_value:
            self.best_value = current_value
            self._save(model, optimizer, epoch, metrics)
        elif self.mode == "max" and current_value > self.best_value:
            self.best_value = current_value
            self._save(model, optimizer, epoch, metrics)

    def _save(self, model, optimizer, epoch, metrics):
        from utils.checkpoint import save_checkpoint

        save_checkpoint(model, optimizer, epoch, metrics, self.filepath)
