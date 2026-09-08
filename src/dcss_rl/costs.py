"""Training-only costs for decisions and short semantic state cycles."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass, field

from dcss_rl.schema import ObservationData, PlayerView
from dcss_rl.units import (
    CycleStateDigest,
    DecisionCost,
    DecisionWindow,
    ShortCycleCost,
)


def training_reward(
    environment_reward: float,
    *,
    decision_cost: DecisionCost,
    short_cycle_cost: ShortCycleCost,
    repeated_state: bool,
) -> float:
    """Apply training-only costs without changing reported environment returns."""
    return float(
        environment_reward
        - decision_cost
        - (short_cycle_cost if repeated_state else 0.0)
    )


def semantic_cycle_digest(observation: ObservationData) -> CycleStateDigest:
    """Fingerprint stable player-visible state while ignoring transient clocks/text."""
    player = PlayerView(observation["player"])
    player.pop("turn", None)
    player.pop("time", None)
    stable: ObservationData = {
        "player": player,
        "cells": observation["cells"],
        "messages": [],
        "menu": observation["menu"],
        "input_mode": observation["input_mode"],
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    return CycleStateDigest(hashlib.sha256(encoded).hexdigest())


@dataclass(slots=True)
class SemanticCycleTracker:
    """Detect recurrence within a bounded history for one live episode."""

    window: DecisionWindow
    _recent: deque[CycleStateDigest] = field(init=False)

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("cycle decision window must be positive")
        self._recent = deque(maxlen=self.window)

    def reset(self, observation: ObservationData) -> None:
        self._recent.clear()
        self._recent.append(semantic_cycle_digest(observation))

    def observe(self, observation: ObservationData) -> bool:
        digest = semantic_cycle_digest(observation)
        repeated = digest in self._recent
        self._recent.append(digest)
        return repeated
