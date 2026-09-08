"""Training-only costs for decisions, state cycles, and UI interaction bursts."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass, field

from dcss_rl.actions import Action, ActionKind
from dcss_rl.schema import ObservationData, PlayerView
from dcss_rl.units import (
    CycleStateDigest,
    DecisionCost,
    DecisionWindow,
    GameTurn,
    GameTurnDelta,
    ShortCycleCost,
    UiInteractionCost,
    UiInteractionRefillPerTurn,
    UiInteractionTokenBalance,
    UiInteractionTokenCapacity,
)

_DEFAULT_UI_INTERACTION_CAPACITY = UiInteractionTokenCapacity(2.0)
_DEFAULT_UI_INTERACTION_REFILL = UiInteractionRefillPerTurn(0.25)
_ZERO_UI_INTERACTION_COST = UiInteractionCost(0.0)

# Keep the policy explicit and extensible: future player-visible inspection commands
# can join this set without changing token-bucket accounting or action masks.
UI_INTERACTION_ACTION_KINDS = frozenset(
    {ActionKind.ABILITIES, ActionKind.MENU_SELECT, ActionKind.CANCEL}
)


@dataclass(frozen=True, slots=True)
class UiInteractionBudgetConfig:
    """A small free UI burst, replenished by actual player game turns."""

    capacity: UiInteractionTokenCapacity = _DEFAULT_UI_INTERACTION_CAPACITY
    refill_per_turn: UiInteractionRefillPerTurn = _DEFAULT_UI_INTERACTION_REFILL
    overflow_cost: UiInteractionCost = _ZERO_UI_INTERACTION_COST

    def __post_init__(self) -> None:
        if self.capacity < 0:
            raise ValueError("UI-interaction capacity cannot be negative")
        if self.refill_per_turn < 0:
            raise ValueError("UI-interaction refill cannot be negative")
        if self.overflow_cost < 0:
            raise ValueError("UI-interaction overflow cost cannot be negative")

    @property
    def enabled(self) -> bool:
        """Return whether overflow can affect the training objective."""
        return self.overflow_cost > 0


@dataclass(slots=True)
class UiInteractionBudget:
    """Track free UI interactions independently for one worker episode."""

    config: UiInteractionBudgetConfig
    _tokens: UiInteractionTokenBalance = field(
        init=False, default=UiInteractionTokenBalance(0.0)
    )
    _last_turn: GameTurn = field(init=False, default=GameTurn(0))

    def reset(self, observation: ObservationData) -> None:
        self._tokens = UiInteractionTokenBalance(self.config.capacity)
        self._last_turn = _player_turn(observation)

    def observe(self, action: Action, observation: ObservationData) -> bool:
        """Consume a token and report a cost-bearing zero-turn overflow."""
        current_turn = _player_turn(observation)
        turn_advance = GameTurnDelta(max(0, current_turn - self._last_turn))
        self._last_turn = current_turn
        self._tokens = UiInteractionTokenBalance(
            min(
                self.config.capacity,
                self._tokens + turn_advance * self.config.refill_per_turn,
            )
        )
        if not self.config.enabled or action.kind not in UI_INTERACTION_ACTION_KINDS:
            return False

        has_free_token = self._tokens >= 1.0
        if has_free_token:
            self._tokens = UiInteractionTokenBalance(self._tokens - 1.0)
        return turn_advance == 0 and not has_free_token


def _player_turn(observation: ObservationData) -> GameTurn:
    turn = observation["player"].get("turn", 0)
    return GameTurn(turn if isinstance(turn, int) else 0)


def training_reward(
    environment_reward: float,
    *,
    decision_cost: DecisionCost,
    short_cycle_cost: ShortCycleCost,
    repeated_state: bool,
    ui_interaction_cost: UiInteractionCost,
    ui_interaction_overflow: bool,
) -> float:
    """Apply training-only costs without changing reported environment returns."""
    return float(
        environment_reward
        - decision_cost
        - (short_cycle_cost if repeated_state else 0.0)
        - (ui_interaction_cost if ui_interaction_overflow else 0.0)
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
