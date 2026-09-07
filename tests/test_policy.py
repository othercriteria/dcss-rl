import numpy as np

from dcss_rl.actions import Action, ActionKind
from dcss_rl.env import action_to_index
from dcss_rl.policy import ScriptedMibePolicy
from dcss_rl.schema import CellView, ObservationData
from dcss_rl.units import Keycode


def observation(
    *,
    hp: int = 20,
    hp_max: int = 20,
    cells: list[CellView] | None = None,
    messages: list[str] | None = None,
) -> ObservationData:
    default_cells: list[CellView] = [{"x": 0, "y": 0, "g": "@"}]
    return {
        "player": {"hp": hp, "hp_max": hp_max, "pos": {"x": 0, "y": 0}},
        "cells": cells if cells is not None else default_cells,
        "messages": messages or [],
        "menu": None,
        "input_mode": 1,
    }


def test_attacks_adjacent_monster_before_resting() -> None:
    policy = ScriptedMibePolicy()
    state = observation(
        hp=10,
        cells=[
            {"x": 0, "y": 0, "g": "@"},
            {"x": 1, "y": 0, "g": "g", "mon": {"name": "goblin"}},
        ],
    )

    assert policy.decide(state).action == Action(ActionKind.MOVE_E)


def test_ignores_stationary_flora() -> None:
    state = observation(
        cells=[
            {"x": 0, "y": 0, "g": "@"},
            {"x": 1, "y": 0, "g": "P", "mon": {"name": "bush"}},
        ]
    )

    assert ScriptedMibePolicy().decide(state).action == Action(ActionKind.EXPLORE)


def test_treats_sparse_monster_delta_as_tactical() -> None:
    state = observation(
        cells=[
            {"x": 0, "y": 0, "g": "@"},
            {"x": 1, "y": 0, "g": "g", "mon": {}},
        ]
    )

    assert ScriptedMibePolicy().decide(state).action == Action(ActionKind.MOVE_E)


def test_rests_when_injured_and_no_monster_is_visible() -> None:
    assert ScriptedMibePolicy().decide(observation(hp=10)).action == Action(
        ActionKind.REST
    )


def test_routes_to_known_stairs_after_exploration() -> None:
    state = observation(
        cells=[
            {"x": 0, "y": 0, "g": "@"},
            {"x": 1, "y": 0, "g": "."},
            {"x": 2, "y": 0, "g": ">"},
        ],
        messages=["Done exploring."],
    )

    assert ScriptedMibePolicy().decide(state).action == Action(ActionKind.MOVE_E)


def test_routes_to_known_stairs_without_transient_exploration_message() -> None:
    state = observation(
        cells=[
            {"x": 0, "y": 0, "g": "@"},
            {"x": 1, "y": 0, "g": "."},
            {"x": 2, "y": 0, "g": ">", "mf": 44},
        ]
    )

    assert ScriptedMibePolicy().decide(state).action == Action(ActionKind.MOVE_E)


def test_descends_when_message_reveals_stairs_under_player() -> None:
    state = observation(
        messages=["Done exploring.", "There is a stone staircase leading down here."]
    )

    assert ScriptedMibePolicy().decide(state).action == Action(ActionKind.STAIRS_DOWN)


def test_waits_for_announced_but_not_yet_visible_monster() -> None:
    state = observation(messages=["A hobgoblin is nearby!"])

    assert ScriptedMibePolicy().decide(state).action == Action(ActionKind.WAIT)


def test_select_falls_back_to_legal_cancel() -> None:
    size = action_to_index(Action.menu_select(Keycode(255))) + 1
    mask = np.zeros(size, dtype=np.bool_)
    cancel = action_to_index(Action(ActionKind.CANCEL))
    mask[cancel] = True

    assert ScriptedMibePolicy().select(observation(), mask) == cancel
