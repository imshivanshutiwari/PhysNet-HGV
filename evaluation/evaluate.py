import numpy as np
import torch
from .metrics import (
    position_rmse,
    velocity_rmse,
    nees,
    track_continuity_pct,
    divergence_rate,
    ospa_distance,
    gospa_metric,
    pd_pfa_curve,
    mape,
    frechet_distance,
    reacquisition_time,
)
from utils.logger import get_logger

logger = get_logger("evaluate")


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for inputs, targets, masks, _ in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            state_pred, _ = model(inputs)

            all_preds.append(state_pred.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_masks.append(masks.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    masks = np.concatenate(all_masks, axis=0)

    B, S, D = preds.shape
    preds_flat = preds.reshape(B * S, D)
    targets_flat = targets.reshape(B * S, D)
    masks_flat = masks.reshape(B * S)

    metrics = {
        "pos_rmse": position_rmse(preds_flat, targets_flat),
        "vel_rmse": velocity_rmse(preds_flat, targets_flat),
        "track_continuity": track_continuity_pct(preds_flat, targets_flat, masks_flat),
        "divergence_rate": divergence_rate(preds_flat, targets_flat),
        "mape": mape(preds_flat, targets_flat),
    }

    logger.info(f"Evaluation Metrics: {metrics}")
    return metrics


if __name__ == "__main__":
    logger.info("Running evaluation")
