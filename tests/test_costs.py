from copy import deepcopy

from dcss_rl.costs import SemanticCycleTracker, semantic_cycle_digest, training_reward
from dcss_rl.schema import ObservationData
from dcss_rl.units import DecisionCost, DecisionWindow, ShortCycleCost


def state(
    *, x: int = 0, turn: int = 1, messages: list[str] | None = None
) -> ObservationData:
    return {
        "player": {
            "pos": {"x": x, "y": 0},
            "hp": 20,
            "hp_max": 20,
            "xl": 2,
            "depth": 1,
            "turn": turn,
            "time": turn * 10,
        },
        "cells": [{"x": x, "y": 0, "g": "@"}],
        "messages": messages or [],
        "menu": None,
        "input_mode": 1,
    }


def test_cycle_digest_ignores_transient_clock_and_messages() -> None:
    first = state(turn=1, messages=["You wait."])
    later = state(turn=99, messages=["Done waiting."])

    assert semantic_cycle_digest(first) == semantic_cycle_digest(later)


def test_cycle_digest_preserves_absolute_position_and_menu_state() -> None:
    base = state()
    moved = state(x=1)
    menu = deepcopy(base)
    menu["menu"] = {
        "type": "prompt",
        "prompt": "Really renounce your faith?",
        "choices": [{"keycode": ord("y"), "text": "yes"}],
    }

    assert semantic_cycle_digest(base) != semantic_cycle_digest(moved)
    assert semantic_cycle_digest(base) != semantic_cycle_digest(menu)


def test_cycle_tracker_detects_short_recurrence_and_resets_between_episodes() -> None:
    tracker = SemanticCycleTracker(DecisionWindow(3))
    first = state()
    second = state(x=1)
    tracker.reset(first)

    assert not tracker.observe(second)
    assert tracker.observe(first)

    tracker.reset(first)
    assert not tracker.observe(second)


def test_training_reward_applies_independent_typed_costs() -> None:
    assert (
        training_reward(
            3.0,
            decision_cost=DecisionCost(0.1),
            short_cycle_cost=ShortCycleCost(0.5),
            repeated_state=False,
        )
        == 2.9
    )
    assert (
        training_reward(
            3.0,
            decision_cost=DecisionCost(0.1),
            short_cycle_cost=ShortCycleCost(0.5),
            repeated_state=True,
        )
        == 2.4
    )
