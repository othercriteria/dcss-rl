from copy import deepcopy

import pytest

from dcss_rl.actions import Action, ActionKind
from dcss_rl.costs import (
    SemanticCycleTracker,
    UiInteractionBudget,
    UiInteractionBudgetConfig,
    semantic_cycle_digest,
    training_reward,
)
from dcss_rl.schema import ObservationData
from dcss_rl.units import (
    DecisionCost,
    DecisionWindow,
    Keycode,
    ShortCycleCost,
    UiInteractionCost,
    UiInteractionRefillPerTurn,
    UiInteractionTokenCapacity,
)


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
            ui_interaction_cost=UiInteractionCost(0.25),
            ui_interaction_overflow=False,
        )
        == 2.9
    )
    assert (
        training_reward(
            3.0,
            decision_cost=DecisionCost(0.1),
            short_cycle_cost=ShortCycleCost(0.5),
            repeated_state=True,
            ui_interaction_cost=UiInteractionCost(0.25),
            ui_interaction_overflow=True,
        )
        == 2.15
    )


def interaction_budget(
    *, capacity: float = 2.0, refill: float = 0.25, cost: float = 0.1
) -> UiInteractionBudget:
    return UiInteractionBudget(
        UiInteractionBudgetConfig(
            capacity=UiInteractionTokenCapacity(capacity),
            refill_per_turn=UiInteractionRefillPerTurn(refill),
            overflow_cost=UiInteractionCost(cost),
        )
    )


def test_sparse_ui_interactions_stay_within_refilled_budget() -> None:
    budget = interaction_budget()
    budget.reset(state(turn=1))

    assert not budget.observe(Action(ActionKind.ABILITIES), state(turn=1))
    assert not budget.observe(Action.menu_select(Keycode(ord("a"))), state(turn=1))
    assert not budget.observe(Action(ActionKind.WAIT), state(turn=5))
    assert not budget.observe(Action(ActionKind.ABILITIES), state(turn=5))


def test_zero_turn_ui_burst_reports_overflow_after_free_capacity() -> None:
    budget = interaction_budget()
    unchanged = state(turn=1)
    budget.reset(unchanged)

    assert not budget.observe(Action(ActionKind.ABILITIES), unchanged)
    assert not budget.observe(Action.menu_select(Keycode(ord("a"))), unchanged)
    assert budget.observe(Action(ActionKind.CANCEL), unchanged)


def test_game_turns_refill_fractional_ui_budget() -> None:
    budget = interaction_budget(capacity=1.0, refill=0.5)
    budget.reset(state(turn=1))

    assert not budget.observe(Action(ActionKind.ABILITIES), state(turn=1))
    assert not budget.observe(Action(ActionKind.WAIT), state(turn=2))
    assert budget.observe(Action.menu_select(Keycode(ord("a"))), state(turn=2))
    assert not budget.observe(Action(ActionKind.WAIT), state(turn=3))
    assert not budget.observe(Action(ActionKind.CANCEL), state(turn=3))


def test_turn_advancing_ui_interaction_is_never_costed() -> None:
    budget = interaction_budget(capacity=0.0, refill=0.0)
    budget.reset(state(turn=1))

    assert not budget.observe(Action(ActionKind.ABILITIES), state(turn=2))
    assert budget.observe(Action(ActionKind.ABILITIES), state(turn=2))


def test_ui_budget_reset_restores_episode_capacity() -> None:
    budget = interaction_budget(capacity=1.0, refill=0.0)
    unchanged = state(turn=1)
    budget.reset(unchanged)
    assert not budget.observe(Action(ActionKind.CANCEL), unchanged)
    assert budget.observe(Action(ActionKind.CANCEL), unchanged)

    budget.reset(state(turn=1))
    assert not budget.observe(Action(ActionKind.CANCEL), unchanged)


def test_default_zero_cost_preserves_overflow_telemetry() -> None:
    config = UiInteractionBudgetConfig()
    budget = UiInteractionBudget(config)
    unchanged = state(turn=1)
    budget.reset(unchanged)

    assert not config.enabled
    assert not budget.observe(Action(ActionKind.ABILITIES), unchanged)
    assert not budget.observe(Action(ActionKind.ABILITIES), unchanged)
    assert budget.observe(Action(ActionKind.ABILITIES), unchanged)


def test_ui_budget_validates_knobs() -> None:
    with pytest.raises(ValueError, match="capacity"):
        UiInteractionBudgetConfig(capacity=UiInteractionTokenCapacity(-1.0))
