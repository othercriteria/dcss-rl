"""Time-limit-aware return estimation for online reinforcement learning."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float32]
type BoolArray = NDArray[np.bool_]


def generalized_advantage_estimate(
    rewards: FloatArray,
    values: FloatArray,
    episode_ends: BoolArray,
    time_limit_bootstraps: FloatArray,
    final_values: FloatArray,
    *,
    discount: float,
    gae_lambda: float,
) -> tuple[FloatArray, FloatArray]:
    """Estimate advantages without treating administrative limits as terminals."""
    advantages = np.zeros_like(rewards)
    future_advantage = np.zeros(rewards.shape[1], dtype=np.float32)
    next_values = final_values
    for step in range(len(rewards) - 1, -1, -1):
        ended = episode_ends[step].astype(np.float32)
        alive = 1.0 - ended
        bootstrap = next_values * alive + time_limit_bootstraps[step] * ended
        delta = rewards[step] + discount * bootstrap - values[step]
        future_advantage = delta + discount * gae_lambda * alive * future_advantage
        advantages[step] = future_advantage
        next_values = values[step]
    return advantages, advantages + values
