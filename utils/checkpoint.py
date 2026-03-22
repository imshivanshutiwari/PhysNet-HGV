import torch
import os


def save_checkpoint(model, optimizer, epoch, metrics, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict() if hasattr(model, "state_dict") else model,
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "metrics": metrics,
    }
    torch.save(checkpoint, filename)


def load_checkpoint(model, optimizer, filename, device="cpu"):
    if not os.path.exists(filename):
        return None, None, None
    checkpoint = torch.load(filename, map_location=device)
    if hasattr(model, "load_state_dict"):
        model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("metrics", {})
