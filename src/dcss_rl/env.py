"""Gymnasium environment over an unmodified local DCSS process."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dcss_rl.actions import Action, ActionKind, encode_action, legal_actions
from dcss_rl.observation import ObservationReducer, SemanticObservation
from dcss_rl.schema import GymMetadata, ObservationData
from dcss_rl.webtiles import GameConfig, ManagedGame, ObservationBatch

_COMMAND_ACTIONS = tuple(
    Action(kind) for kind in ActionKind if kind is not ActionKind.MENU_SELECT
)
_MENU_OFFSET = len(_COMMAND_ACTIONS)
_KEYCODE_COUNT = 256


class SemanticObservationSpace(gym.Space[ObservationData]):
    """Validation space for the variable-sized semantic observation schema."""

    def __init__(self) -> None:
        super().__init__(shape=None, dtype=None)

    def sample(
        self, mask: Any | None = None, probability: Any | None = None
    ) -> ObservationData:
        del mask, probability
        return {
            "player": {},
            "cells": [],
            "messages": [],
            "menu": None,
            "input_mode": None,
        }

    def contains(self, x: Any) -> bool:
        if not isinstance(x, Mapping):
            return False
        return all(
            key in x for key in ("player", "cells", "messages", "menu", "input_mode")
        )


def action_to_index(action: Action) -> int:
    """Encode a structured action into the fixed Gym action catalog."""
    if action.kind is ActionKind.MENU_SELECT:
        if action.keycode is None or not 0 <= action.keycode < _KEYCODE_COUNT:
            raise ValueError("menu keycode must be in [0, 255]")
        return _MENU_OFFSET + action.keycode
    try:
        return _COMMAND_ACTIONS.index(action)
    except ValueError as error:
        raise ValueError(f"action is not in the fixed catalog: {action}") from error


def index_to_action(index: int) -> Action:
    """Decode a fixed Gym action index into its structured representation."""
    if not 0 <= index < _MENU_OFFSET + _KEYCODE_COUNT:
        raise ValueError(f"action index out of range: {index}")
    if index < _MENU_OFFSET:
        return _COMMAND_ACTIONS[index]
    return Action.menu_select(index - _MENU_OFFSET)


def action_mask(observation: SemanticObservation) -> np.ndarray:
    """Return a boolean mask aligned with :class:`DcssEnv.action_space`."""
    result = np.zeros(_MENU_OFFSET + _KEYCODE_COUNT, dtype=np.bool_)
    for action in legal_actions(observation):
        result[action_to_index(action)] = True
    return result


class DcssEnv(gym.Env[ObservationData, int]):
    """Single-process Gym environment with structured, masked actions."""

    metadata: ClassVar[GymMetadata] = {"render_modes": []}

    def __init__(
        self,
        binary: Path,
        *,
        game_config: GameConfig | None = None,
        starting_weapon_key: str = "c",
        max_steps: int | None = None,
    ) -> None:
        super().__init__()
        if len(starting_weapon_key) != 1:
            raise ValueError("starting_weapon_key must be one character")
        self.binary = Path(binary)
        self.game_config = game_config or GameConfig()
        self.starting_weapon_key = starting_weapon_key
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(_MENU_OFFSET + _KEYCODE_COUNT)
        self.observation_space = SemanticObservationSpace()
        self.game: ManagedGame | None = None
        self.reducer: ObservationReducer | None = None
        self.current: SemanticObservation | None = None
        self.last_batch: ObservationBatch | None = None
        self.last_exchange: tuple[ObservationBatch, ...] = ()
        self.last_keycodes: tuple[int, ...] = ()
        self.steps = 0
        self._max_depth = 0
        self._max_xl = 1

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObservationData, dict[str, Any]]:
        super().reset(seed=seed)
        self.close()
        options = options or {}
        game_seed = options.get("game_seed", self.game_config.seed)
        if game_seed is not None and not isinstance(game_seed, int):
            raise ValueError("game_seed must be an integer or None")
        config = GameConfig(
            name=self.game_config.name,
            species=self.game_config.species,
            background=self.game_config.background,
            seed=game_seed,
        )
        self.game = ManagedGame(self.binary, config=config)
        self.reducer = ObservationReducer()
        self.last_batch = self.game.start()
        exchange = [self.last_batch]
        keycodes: list[int] = []
        self.current = self.reducer.apply(self.last_batch)
        self.steps = 0
        self._max_depth = 0
        self._max_xl = 1

        if self.current.menu_type == "newgame-choice":
            requested = Action.menu_select(ord(self.starting_weapon_key))
            keycode = encode_action(requested, self.current)
            self.last_batch = self.game.send_key(keycode)
            exchange.append(self.last_batch)
            keycodes.append(ord(keycode) if isinstance(keycode, str) else keycode)
            self.current = self.reducer.apply(self.last_batch)

        self.last_exchange = tuple(exchange)
        self.last_keycodes = tuple(keycodes)

        self._update_maxima(self.current)
        return self.current.to_dict(), self._info(None)

    def step(
        self, action: int
    ) -> tuple[ObservationData, float, bool, bool, dict[str, Any]]:
        if self.game is None or self.reducer is None or self.current is None:
            raise RuntimeError("reset must be called before step")
        structured_action = index_to_action(int(action))
        keycode = encode_action(structured_action, self.current)
        previous_depth = self._max_depth
        previous_xl = self._max_xl

        self.last_batch = self.game.send_key(keycode)
        self.last_exchange = (self.last_batch,)
        self.last_keycodes = (ord(keycode) if isinstance(keycode, str) else keycode,)
        self.current = self.reducer.apply(self.last_batch)
        self.steps += 1
        terminated, outcome = self._terminal_outcome(self.last_batch)
        self._update_maxima(self.current)
        reward = float(
            3 * (self._max_depth - previous_depth)
            + 10 * (self._max_xl - previous_xl)
            + (1000 if outcome == "won" else 0)
            - (10 if outcome == "dead" else 0)
        )
        truncated = self.max_steps is not None and self.steps >= self.max_steps
        return (
            self.current.to_dict(),
            reward,
            terminated,
            truncated,
            self._info(structured_action, outcome=outcome),
        )

    def _update_maxima(self, observation: SemanticObservation) -> None:
        depth = observation.player.get("depth")
        xl = observation.player.get("xl")
        if isinstance(depth, int):
            self._max_depth = max(self._max_depth, depth)
        if isinstance(xl, int):
            self._max_xl = max(self._max_xl, xl)

    @staticmethod
    def _terminal_outcome(batch: ObservationBatch) -> tuple[bool, str | None]:
        for message in batch.controls:
            if message.kind != "exit_reason":
                continue
            outcome = message.payload.get("type")
            if isinstance(outcome, str) and outcome != "unknown":
                return True, outcome
        return False, None

    def _info(
        self, action: Action | None, *, outcome: str | None = None
    ) -> dict[str, Any]:
        if self.current is None:
            raise RuntimeError("environment has no current observation")
        return {
            "action_mask": action_mask(self.current),
            "structured_action": action.to_dict() if action is not None else None,
            "emitted_keycodes": self.last_keycodes,
            "outcome": outcome,
            "steps": self.steps,
            "max_depth": self._max_depth,
            "max_xl": self._max_xl,
        }

    def close(self) -> None:
        if self.game is not None:
            self.game.close()
            self.game = None
        self.reducer = None
        self.current = None
        self.last_batch = None
        self.last_exchange = ()
        self.last_keycodes = ()
