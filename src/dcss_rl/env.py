"""Gymnasium environment over an unmodified local DCSS process."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dcss_rl.actions import Action, ActionKind, encode_action, legal_actions
from dcss_rl.observation import ObservationReducer, SemanticObservation
from dcss_rl.schema import EnvironmentInfo, GymMetadata, ObservationData, ResetOptions
from dcss_rl.units import (
    ActionCount,
    ActionIndex,
    GameSeed,
    Keycode,
    RewardWeight,
    StepLimit,
)
from dcss_rl.webtiles import GameConfig, ManagedGame, ObservationBatch
from dcss_rl.webtiles.cache import StaticDataCache

_LEGACY_COMMAND_ACTIONS = tuple(
    Action(kind)
    for kind in ActionKind
    if kind not in {ActionKind.MENU_SELECT, ActionKind.ABILITIES}
)
_MENU_OFFSET = len(_LEGACY_COMMAND_ACTIONS)
_KEYCODE_COUNT = 256
_ABILITY_INDEX = ActionIndex(_MENU_OFFSET + _KEYCODE_COUNT)
ACTION_COUNT = ActionCount(_ABILITY_INDEX + 1)
_ZERO_REWARD_WEIGHT = RewardWeight(0.0)


@dataclass(frozen=True, slots=True)
class RewardShaping:
    """Potential-style training rewards derived only from player-visible state."""

    explored_cell: RewardWeight = _ZERO_REWARD_WEIGHT
    depth_progress: RewardWeight = _ZERO_REWARD_WEIGHT
    experience_progress: RewardWeight = _ZERO_REWARD_WEIGHT
    hp_fraction: RewardWeight = _ZERO_REWARD_WEIGHT

    def __post_init__(self) -> None:
        if any(
            weight < 0
            for weight in (
                self.explored_cell,
                self.depth_progress,
                self.experience_progress,
                self.hp_fraction,
            )
        ):
            raise ValueError("reward-shaping weights cannot be negative")

    @property
    def enabled(self) -> bool:
        """Report whether computing player-visible shaping can affect reward."""
        return any(
            weight != _ZERO_REWARD_WEIGHT
            for weight in (
                self.explored_cell,
                self.depth_progress,
                self.experience_progress,
                self.hp_fraction,
            )
        )


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


def action_to_index(action: Action) -> ActionIndex:
    """Encode a structured action into the fixed Gym action catalog."""
    if action.kind is ActionKind.MENU_SELECT:
        if action.keycode is None or not 0 <= action.keycode < _KEYCODE_COUNT:
            raise ValueError("menu keycode must be in [0, 255]")
        return ActionIndex(_MENU_OFFSET + action.keycode)
    if action.kind is ActionKind.ABILITIES:
        return _ABILITY_INDEX
    try:
        return ActionIndex(_LEGACY_COMMAND_ACTIONS.index(action))
    except ValueError as error:
        raise ValueError(f"action is not in the fixed catalog: {action}") from error


def index_to_action(index: ActionIndex) -> Action:
    """Decode a fixed Gym action index into its structured representation."""
    if not 0 <= index < ACTION_COUNT:
        raise ValueError(f"action index out of range: {index}")
    if index == _ABILITY_INDEX:
        return Action(ActionKind.ABILITIES)
    if index < _MENU_OFFSET:
        return _LEGACY_COMMAND_ACTIONS[index]
    return Action.menu_select(Keycode(index - _MENU_OFFSET))


def action_mask(observation: SemanticObservation) -> np.ndarray:
    """Return a boolean mask aligned with :class:`DcssEnv.action_space`."""
    result = np.zeros(ACTION_COUNT, dtype=np.bool_)
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
        max_steps: StepLimit | None = None,
        run_root: Path | None = None,
        reward_shaping: RewardShaping | None = None,
        static_cache: StaticDataCache | None = None,
        collect_static_cache_timing: bool = False,
    ) -> None:
        super().__init__()
        if len(starting_weapon_key) != 1:
            raise ValueError("starting_weapon_key must be one character")
        self.binary = Path(binary)
        self.game_config = game_config or GameConfig()
        self.starting_weapon_key = starting_weapon_key
        self.max_steps = max_steps
        self.run_root = run_root
        self.reward_shaping = reward_shaping or RewardShaping()
        self.static_cache = static_cache
        self.collect_static_cache_timing = collect_static_cache_timing
        self.action_space = spaces.Discrete(ACTION_COUNT)
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
        """Implement Gym's raw info boundary; project code uses ``reset_typed``."""
        observation, info = self.reset_typed(
            seed=seed, options=cast(ResetOptions | None, options)
        )
        return observation, cast(dict[str, Any], info)

    def reset_typed(
        self,
        *,
        seed: int | None = None,
        options: ResetOptions | None = None,
    ) -> tuple[ObservationData, EnvironmentInfo]:
        """Reset DCSS while retaining semantic types past the Gym boundary."""
        super().reset(seed=seed)
        self.close()
        reset_options = options if options is not None else ResetOptions()
        raw_game_seed = reset_options.get("game_seed", self.game_config.seed)
        game_seed = GameSeed(raw_game_seed) if raw_game_seed is not None else None
        config = GameConfig(
            name=self.game_config.name,
            species=self.game_config.species,
            background=self.game_config.background,
            seed=game_seed,
        )
        self.game = ManagedGame(
            self.binary,
            config=config,
            run_root=self.run_root,
            static_cache=self.static_cache,
            collect_static_cache_timing=self.collect_static_cache_timing,
        )
        self.reducer = ObservationReducer()
        setup_keycode = Keycode(ord(self.starting_weapon_key))
        self.last_batch = self.game.start(initial_keycode=setup_keycode)
        exchange = [self.last_batch]
        keycodes: list[int] = [setup_keycode]
        self.current = self.reducer.apply(self.last_batch)
        self.steps = 0
        self._max_depth = 0
        self._max_xl = 1

        self.last_exchange = tuple(exchange)
        self.last_keycodes = tuple(keycodes)

        self._update_maxima(self.current)
        return self.current.to_dict(), self._info(None)

    def step(
        self, action: int
    ) -> tuple[ObservationData, float, bool, bool, dict[str, Any]]:
        """Implement Gym's raw info boundary; project code uses ``step_typed``."""
        observation, reward, terminated, truncated, info = self.step_typed(
            ActionIndex(action)
        )
        return (
            observation,
            reward,
            terminated,
            truncated,
            cast(dict[str, Any], info),
        )

    def step_typed(
        self, action: ActionIndex
    ) -> tuple[ObservationData, float, bool, bool, EnvironmentInfo]:
        """Advance DCSS while retaining semantic types past the Gym boundary."""
        if self.game is None or self.reducer is None or self.current is None:
            raise RuntimeError("reset must be called before step")
        structured_action = index_to_action(action)
        keycode = encode_action(structured_action, self.current)
        previous_observation = (
            self.current.to_dict() if self.reward_shaping.enabled else None
        )
        previous_depth = self._max_depth
        previous_xl = self._max_xl

        self.last_batch = (
            self.game.send_key(keycode, level_transition=True)
            if structured_action.kind in {ActionKind.STAIRS_UP, ActionKind.STAIRS_DOWN}
            else self.game.send_key(keycode)
        )
        self.last_exchange = (self.last_batch,)
        self.last_keycodes = (ord(keycode) if isinstance(keycode, str) else keycode,)
        self.current = self.reducer.apply(self.last_batch)
        current_observation = self.current.to_dict()
        self.steps += 1
        terminated, outcome = self._terminal_outcome(self.last_batch)
        self._update_maxima(self.current)
        reward = self._sparse_reward(
            previous_depth,
            previous_xl,
            self._max_depth,
            self._max_xl,
            outcome,
        )
        if previous_observation is not None:
            reward += shaped_reward(
                previous_observation,
                current_observation,
                shaping=self.reward_shaping,
            )
        truncated = self.max_steps is not None and self.steps >= self.max_steps
        return (
            current_observation,
            reward,
            terminated,
            truncated,
            self._info(structured_action, outcome=outcome),
        )

    @staticmethod
    def _sparse_reward(
        previous_depth: int,
        previous_xl: int,
        current_depth: int,
        current_xl: int,
        outcome: str | None,
    ) -> float:
        """Reward milestones without making delayed death cheaper under discounting."""
        return float(
            3 * (current_depth - previous_depth)
            + 10 * (current_xl - previous_xl)
            + (1000 if outcome == "won" else 0)
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
        for message in batch.messages:
            if message.control and message.kind == "exit_reason":
                outcome = message.payload.get("type")
                if isinstance(outcome, str) and outcome != "unknown":
                    return True, outcome
            if message.kind == "milestone" and message.payload.get("status") == "dead":
                return True, "dead"
            if message.kind == "player" and message.payload.get("hp") == 0:
                return True, "dead"
            if message.kind == "msgs":
                raw_messages = message.payload.get("messages")
                if isinstance(raw_messages, list) and any(
                    isinstance(item, dict)
                    and isinstance(item.get("text"), str)
                    and "you die" in item["text"].casefold()
                    for item in raw_messages
                ):
                    return True, "dead"
        return False, None

    def _info(
        self, action: Action | None, *, outcome: str | None = None
    ) -> EnvironmentInfo:
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


def shaped_reward(
    previous: ObservationData,
    current: ObservationData,
    *,
    shaping: RewardShaping,
) -> float:
    """Return dense potential deltas without changing headline sparse rewards."""
    previous_cells = {(cell["x"], cell["y"]) for cell in previous["cells"]}
    current_cells = {(cell["x"], cell["y"]) for cell in current["cells"]}
    newly_explored = len(current_cells - previous_cells)
    experience_delta = _experience_potential(current) - _experience_potential(previous)
    depth_delta = _depth_potential(current) - _depth_potential(previous)
    hp_delta = _hp_fraction(current) - _hp_fraction(previous)
    return float(
        shaping.explored_cell * newly_explored
        + shaping.depth_progress * depth_delta
        + shaping.experience_progress * experience_delta
        + shaping.hp_fraction * hp_delta
    )


def _experience_potential(observation: ObservationData) -> float:
    player = observation["player"]
    xl = player.get("xl", 1)
    progress = player.get("progress", 0)
    return float(xl) + float(progress) / 100.0


def _depth_potential(observation: ObservationData) -> float:
    return float(observation["player"].get("depth", 1))


def _hp_fraction(observation: ObservationData) -> float:
    player = observation["player"]
    hp = player.get("hp", 0)
    hp_max = max(player.get("hp_max", 1), 1)
    return hp / hp_max
