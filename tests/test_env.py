from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from dcss_rl.actions import Action, ActionKind
from dcss_rl.env import (
    DcssEnv,
    RewardShaping,
    action_to_index,
    index_to_action,
    shaped_reward,
)
from dcss_rl.observation import ObservationReducer, SemanticObservation
from dcss_rl.schema import ObservationData
from dcss_rl.units import GameSeed, Keycode, RewardWeight, StepLimit
from dcss_rl.webtiles import GameConfig, ManagedGame, Message, ObservationBatch

_DCSS_BINARY = Path("vendor/crawl/crawl-ref/source/crawl")


@pytest.mark.parametrize("collect_timing", [False, True])
def test_reset_passes_cache_timing_option_to_private_game(
    monkeypatch: pytest.MonkeyPatch, collect_timing: bool
) -> None:
    game = MagicMock(spec=ManagedGame)
    game.start.return_value = ObservationBatch(
        (
            Message({"msg": "player", "hp": 20, "hp_max": 20, "depth": 1, "xl": 1}),
            Message({"msg": "input_mode", "mode": 1}),
        ),
        (),
    )
    constructor = MagicMock(return_value=game)
    monkeypatch.setattr("dcss_rl.env.ManagedGame", constructor)
    env = DcssEnv(_DCSS_BINARY, collect_static_cache_timing=collect_timing)
    env.reset_typed()
    assert constructor.call_args.kwargs["collect_static_cache_timing"] is collect_timing
    env.close()


@pytest.mark.parametrize(
    "kind,key,level_transition",
    [
        (ActionKind.STAIRS_DOWN, ">", True),
        (ActionKind.STAIRS_UP, "<", True),
        (ActionKind.MENU_SELECT, ">", False),
        (ActionKind.MENU_SELECT, "<", False),
    ],
)
def test_level_readiness_is_selected_by_action_not_menu_hotkey(
    kind: ActionKind, key: str, level_transition: bool
) -> None:
    game = MagicMock(spec=ManagedGame)
    game.send_key.return_value = ObservationBatch(
        (Message({"msg": "input_mode", "mode": 1}),), ()
    )
    env = DcssEnv(_DCSS_BINARY)
    env.game = cast(ManagedGame, game)
    env.reducer = ObservationReducer()
    state: ObservationData = {
        "player": {"hp": 20, "hp_max": 20, "depth": 1, "xl": 1},
        "cells": [],
        "messages": [
            "There is a staircase leading down here.",
            "There is a staircase leading up here.",
        ],
        "menu": None
        if level_transition
        else {
            "type": "menu",
            "prompt": "Page",
            "choices": [{"keycode": ord(key), "text": "Page"}],
        },
        "input_mode": 1,
    }
    env.current = SemanticObservation.from_dict(state)
    action = Action(kind) if level_transition else Action.menu_select(Keycode(ord(key)))
    env.step_typed(action_to_index(action))
    if level_transition:
        game.send_key.assert_called_once_with(key, level_transition=True)
    else:
        game.send_key.assert_called_once_with(ord(key))


@pytest.mark.parametrize("kind", list(ActionKind))
def test_fixed_action_catalog_round_trips(kind: ActionKind) -> None:
    action = (
        Action.menu_select(Keycode(42))
        if kind is ActionKind.MENU_SELECT
        else Action(kind)
    )
    assert index_to_action(action_to_index(action)) == action


@pytest.mark.parametrize("keycode", [32, 13, 27])
@pytest.mark.parametrize("menu_type,mode", [("more", 5), ("menu", 5), ("more", 1)])
def test_more_continuation_requires_structured_more_state(
    keycode: int, menu_type: str, mode: int
) -> None:
    game = MagicMock(spec=ManagedGame)
    game.send_key.return_value = ObservationBatch(
        (Message({"msg": "input_mode", "mode": 1}),), ()
    )
    env = DcssEnv(_DCSS_BINARY)
    env.game = cast(ManagedGame, game)
    env.reducer = ObservationReducer()
    env.current = SemanticObservation.from_dict(
        {
            "player": {"hp": 20, "hp_max": 20, "depth": 1, "xl": 1},
            "cells": [],
            "messages": [],
            "menu": {
                "type": menu_type,
                "prompt": "--more--",
                "choices": [{"keycode": keycode, "text": "Continue"}],
            },
            "input_mode": mode,
        }
    )
    action = (
        Action(ActionKind.CANCEL)
        if keycode == 27
        else Action.menu_select(Keycode(keycode))
    )
    env.step_typed(action_to_index(action))
    if menu_type == "more" and mode == 5:
        game.send_key.assert_called_once_with(keycode, ui_continuation=True)
    else:
        game.send_key.assert_called_once_with(keycode)


def test_player_visible_death_ends_episode_before_post_game_ui() -> None:
    batch = ObservationBatch(
        observations=(
            Message({"msg": "player", "hp": 0}),
            Message(
                {
                    "msg": "msgs",
                    "messages": [{"text": "<lightgrey>You die...", "turn": 7}],
                }
            ),
        ),
        controls=(),
    )

    assert DcssEnv._terminal_outcome(batch) == (True, "dead")


def test_sparse_reward_does_not_pay_agent_to_postpone_death() -> None:
    # Death is terminal feedback. An explicit negative reward would be discounted by
    # every delaying action and could be escaped entirely at a finite time limit.
    assert DcssEnv._sparse_reward(1, 1, 1, 1, "dead") == 0.0


def test_dense_reward_uses_visible_potential_deltas() -> None:
    previous: ObservationData = {
        "player": {
            "depth": 1,
            "xl": 1,
            "progress": 50,
            "hp": 10,
            "hp_max": 20,
        },
        "cells": [{"x": 0, "y": 0, "g": "@"}],
        "messages": [],
        "menu": None,
        "input_mode": 1,
    }
    current: ObservationData = {
        "player": {
            "depth": 2,
            "xl": 2,
            "progress": 10,
            "hp": 15,
            "hp_max": 20,
        },
        "cells": [
            {"x": 0, "y": 0, "g": "."},
            {"x": 1, "y": 0, "g": "@"},
            {"x": 2, "y": 0, "g": "."},
        ],
        "messages": [],
        "menu": None,
        "input_mode": 1,
    }
    shaping = RewardShaping(
        explored_cell=RewardWeight(0.1),
        depth_progress=RewardWeight(5.0),
        experience_progress=RewardWeight(2.0),
        hp_fraction=RewardWeight(4.0),
    )

    # 2 new cells * .1 + 1 depth * 5 + .6 levels * 2 + .25 HP fraction * 4
    assert shaped_reward(previous, current, shaping=shaping) == pytest.approx(7.4)


def test_dense_reward_is_zero_by_default() -> None:
    observation: ObservationData = {
        "player": {"xl": 1, "progress": 0, "hp": 10, "hp_max": 20},
        "cells": [{"x": 0, "y": 0, "g": "@"}],
        "messages": [],
        "menu": None,
        "input_mode": 1,
    }

    assert shaped_reward(observation, observation, shaping=RewardShaping()) == 0.0
    assert not RewardShaping().enabled
    assert RewardShaping(explored_cell=RewardWeight(0.1)).enabled


@pytest.mark.integration
@pytest.mark.skipif(
    not _DCSS_BINARY.is_file(), reason="local DCSS binary has not been built"
)
def test_gym_environment_resets_and_steps_real_trunk() -> None:
    env = DcssEnv(
        _DCSS_BINARY,
        game_config=GameConfig(seed=GameSeed(7)),
        max_steps=StepLimit(1),
    )
    try:
        observation, info = env.reset()
        assert env.observation_space.contains(observation)
        assert observation["player"]["species"] == "Minotaur"
        assert info["action_mask"].dtype == np.bool_
        assert info["action_mask"][action_to_index(Action(ActionKind.WAIT))]

        next_observation, reward, terminated, truncated, step_info = env.step(
            action_to_index(Action(ActionKind.WAIT))
        )
        assert env.observation_space.contains(next_observation)
        assert isinstance(reward, float)
        assert not terminated
        assert truncated
        assert step_info["structured_action"] == {
            "kind": "wait",
            "keycode": None,
        }
    finally:
        env.close()
