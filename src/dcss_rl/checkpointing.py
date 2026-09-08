"""Deterministic paths for checkpoint series produced during online training."""

from pathlib import Path

from dcss_rl.units import UpdateCount


def update_checkpoint_path(directory: Path, update: UpdateCount) -> Path:
    """Return a naturally sorted immutable checkpoint path for one update."""
    if update < 1:
        raise ValueError("checkpoint update must be positive")
    return Path(directory) / f"update-{update:04d}.pt"
