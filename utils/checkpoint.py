"""
Model checkpoint save / load utilities.

Persists model weights, optimiser state, epoch, and optional metadata
into a single ``.pt`` file.  Supports best-model tracking via a metric.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str,
    metric: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Serialise training state to disk.

    Parameters
    ----------
    model : torch.nn.Module
        The model whose ``state_dict`` is saved.
    optimizer : torch.optim.Optimizer
        The optimiser whose ``state_dict`` is saved.
    epoch : int
        Current training epoch (0-indexed).
    path : str
        Destination file path (e.g. ``checkpoints/best.pt``).
    metric : float or None
        Optional validation metric for bookkeeping.
    extra : dict or None
        Arbitrary extra data to persist.

    Returns
    -------
    pathlib.Path
        Resolved path to the saved checkpoint.
    """
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if metric is not None:
        payload["metric"] = metric
    if extra is not None:
        payload.update(extra)

    torch.save(payload, ckpt_path)
    return ckpt_path


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Restore training state from a checkpoint file.

    Parameters
    ----------
    path : str
        Path to the ``.pt`` checkpoint.
    model : torch.nn.Module
        Model to load weights into.
    optimizer : torch.optim.Optimizer or None
        Optimiser to restore state into (skipped if *None*).
    device : str
        Map location for ``torch.load``.

    Returns
    -------
    dict
        The full checkpoint dictionary (contains ``epoch``, ``metric``, etc.).
    """
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    return payload
