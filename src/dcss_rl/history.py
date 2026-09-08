"""Policy-side action history, kept separate from semantic game observations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from dcss_rl.policy import ActionHistory
from dcss_rl.units import ActionHistoryLength

type ActionHistoryVector = NDArray[np.float32]


def encode_action_history(
    actions: ActionHistory,
    *,
    action_count: int,
    length: ActionHistoryLength,
) -> ActionHistoryVector:
    """One-hot encode newest-first actions with all-zero missing-history slots."""
    result = np.zeros(action_count * length, dtype=np.float32)
    if not length:
        return result
    for offset, action in enumerate(reversed(actions[-length:])):
        result[offset * action_count + action] = 1.0
    return result
