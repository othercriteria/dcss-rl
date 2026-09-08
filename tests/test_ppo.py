import numpy as np
import pytest
import torch

from dcss_rl.actions import Action
from dcss_rl.env import ACTION_COUNT, action_to_index
from dcss_rl.features import feature_count
from dcss_rl.learned import ModelConfig, SemanticActorCritic, align_feature_spec
from dcss_rl.ppo import (
    _restrict_warmup_gradients,
    _uses_reset_bootstrap,
    _warmup_action_indices,
    _WorkerStep,
)
from dcss_rl.returns import ReturnBoundaryMode, generalized_advantage_estimate
from dcss_rl.schedule import TrainingSeedSchedule
from dcss_rl.units import (
    ActionIndex,
    CaseCount,
    EpisodeIndex,
    Keycode,
    TerminalOutcome,
    WorkerCount,
    WorkerIndex,
)


def test_training_seed_schedule_allocates_disjoint_worker_blocks() -> None:
    schedule = TrainingSeedSchedule(CaseCount(64), WorkerCount(20))

    rounds = [
        {
            schedule.case_index(WorkerIndex(worker), EpisodeIndex(episode))
            for worker in range(20)
        }
        for episode in range(4)
    ]

    assert rounds[0] == set(range(0, 20))
    assert rounds[1] == set(range(20, 40))
    assert rounds[2] == set(range(40, 60))
    assert rounds[3] == {*range(60, 64), *range(0, 16)}


def test_training_seed_schedule_rejects_empty_suite() -> None:
    with pytest.raises(ValueError, match="counts must be positive"):
        TrainingSeedSchedule(CaseCount(0), WorkerCount(20))


def test_gae_bootstraps_time_limit_without_crossing_episode_boundary() -> None:
    rewards = np.asarray([[0.0], [0.0], [100.0]], dtype=np.float32)
    values = np.asarray([[2.0], [3.0], [50.0]], dtype=np.float32)
    episode_ends = np.asarray([[False], [True], [False]], dtype=np.bool_)
    boundary_bootstraps = np.asarray([[0.0], [7.0], [0.0]], dtype=np.float32)

    _, returns = generalized_advantage_estimate(
        rewards,
        values,
        episode_ends,
        boundary_bootstraps,
        np.asarray([60.0], dtype=np.float32),
        discount=1.0,
        gae_lambda=1.0,
    )

    # The time-limited episode bootstraps V=7, while the next episode's reward 100
    # cannot leak backward across the reset boundary.
    np.testing.assert_array_equal(returns[:, 0], np.asarray([7.0, 7.0, 160.0]))


def test_gae_can_bootstrap_reset_value_without_crossing_death_boundary() -> None:
    rewards = np.asarray([[1.0], [0.0], [100.0]], dtype=np.float32)
    values = np.asarray([[2.0], [3.0], [50.0]], dtype=np.float32)
    episode_ends = np.asarray([[False], [True], [False]], dtype=np.bool_)
    boundary_bootstraps = np.asarray([[0.0], [-4.0], [0.0]], dtype=np.float32)

    _, returns = generalized_advantage_estimate(
        rewards,
        values,
        episode_ends,
        boundary_bootstraps,
        np.asarray([60.0], dtype=np.float32),
        discount=1.0,
        gae_lambda=1.0,
    )

    # Death inherits the fresh-reset value -4, while the next rollout reward 100
    # cannot leak backward across the reset boundary.
    np.testing.assert_array_equal(returns[:, 0], np.asarray([-3.0, -4.0, 160.0]))


def test_continuing_reset_bootstraps_death_but_not_win() -> None:
    def terminal_step(outcome: str) -> _WorkerStep:
        return _WorkerStep(
            observation={
                "player": {},
                "cells": [],
                "messages": [],
                "menu": None,
                "input_mode": 1,
            },
            action_mask=np.ones(1, dtype=np.bool_),
            reward=0.0,
            terminated=True,
            truncated=False,
            completed_return=0.0,
            action_history=(),
            terminal_outcome=TerminalOutcome(outcome),
            short_cycle=False,
        )

    assert _uses_reset_bootstrap(
        terminal_step("dead"), ReturnBoundaryMode.CONTINUING_RESET
    )
    assert not _uses_reset_bootstrap(
        terminal_step("won"), ReturnBoundaryMode.CONTINUING_RESET
    )
    assert not _uses_reset_bootstrap(terminal_step("dead"), ReturnBoundaryMode.EPISODIC)


def test_action_warmup_includes_appended_and_companion_menu_rows() -> None:
    menu_a = action_to_index(Action.menu_select(Keycode(ord("a"))))

    indices = _warmup_action_indices(270, int(ACTION_COUNT), (Keycode(ord("a")),))

    assert set(indices) == {ActionIndex(270), menu_a}


def test_action_warmup_zeros_every_unrelated_gradient() -> None:
    model = SemanticActorCritic(ModelConfig(action_count=4, hidden_size=2))
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    frozen_rows, frozen_features = _restrict_warmup_gradients(
        model, (ActionIndex(1), ActionIndex(3))
    )

    assert frozen_rows.tolist() == [True, False, True, False]
    assert frozen_features is None
    assert model.policy_head.weight.grad is not None
    assert model.policy_head.bias.grad is not None
    assert model.policy_head.weight.grad[:, 0].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert model.policy_head.bias.grad.tolist() == [0.0, 1.0, 0.0, 1.0]
    assert model.input_layer.weight.grad is None


def test_action_warmup_can_train_only_new_semantic_columns() -> None:
    model = SemanticActorCritic(ModelConfig(action_count=4, hidden_size=2))
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    _, frozen_features = _restrict_warmup_gradients(
        model,
        (ActionIndex(3),),
        trainable_feature_indices=(model.input_layer.weight.shape[1] - 1,),
    )

    assert frozen_features is not None
    assert frozen_features[-2:].tolist() == [True, False]
    assert model.input_layer.weight.grad is not None
    assert model.input_layer.weight.grad[0, -2:].tolist() == [0.0, 1.0]
    assert model.input_layer.bias.grad is None


def test_feature_migration_preserves_outputs_before_new_inputs_are_trained() -> None:
    model = SemanticActorCritic(
        ModelConfig(action_count=4, hidden_size=2, feature_spec_version=3)
    )
    old_features = torch.randn(3, feature_count(3))
    old_outputs = model(old_features)

    migrated = align_feature_spec(model, 4)
    new_features = torch.cat((old_features, torch.zeros(3, 2)), dim=1)
    new_outputs = migrated(new_features)

    torch.testing.assert_close(old_outputs[0], new_outputs[0])
    torch.testing.assert_close(old_outputs[1], new_outputs[1])
    torch.testing.assert_close(old_outputs[2], new_outputs[2][:, : feature_count(3)])
    torch.testing.assert_close(new_outputs[2][:, feature_count(3) :], torch.zeros(3, 2))
