from concurrent.futures import Future
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from dcss_rl.actions import Action, ActionKind
from dcss_rl.costs import UiInteractionBudgetConfig
from dcss_rl.env import ACTION_COUNT, action_to_index
from dcss_rl.features import feature_count
from dcss_rl.learned import ModelConfig, SemanticActorCritic, align_feature_spec
from dcss_rl.policy import ScriptedMibePolicy
from dcss_rl.ppo import (
    PpoConfig,
    _checkpoint_metadata,
    _collect_worker_rollout,
    _fixed_inference_inputs,
    _InferenceBatcher,
    _InferenceRequest,
    _InferenceResult,
    _restrict_warmup_gradients,
    _uses_reset_bootstrap,
    _warmup_action_indices,
    _Worker,
    _worker_run_root,
    _WorkerStep,
)
from dcss_rl.returns import ReturnBoundaryMode, generalized_advantage_estimate
from dcss_rl.schedule import TrainingSeedSchedule
from dcss_rl.schema import ObservationData
from dcss_rl.units import (
    ActionHistoryLength,
    ActionIndex,
    CaseCount,
    EpisodeIndex,
    FeatureSpecVersion,
    Keycode,
    RolloutLength,
    StartupAttemptIndex,
    TerminalOutcome,
    UiInteractionCost,
    UiInteractionOverflowCount,
    UiInteractionRefillPerTurn,
    UiInteractionTokenCapacity,
    UpdateCount,
    WorkerCount,
    WorkerIndex,
)


def _rollout_observation(turn: int) -> ObservationData:
    return {
        "player": {"pos": {"x": 0, "y": 0}, "turn": turn},
        "cells": [{"x": 0, "y": 0, "g": "@"}],
        "messages": [],
        "menu": None,
        "input_mode": 1,
    }


class _FakeWorker:
    def __init__(self) -> None:
        self.worker_index = WorkerIndex(0)
        self.observation = _rollout_observation(0)
        self.mask = np.ones(int(ACTION_COUNT), dtype=np.bool_)
        self.rng = np.random.default_rng(1)

    def ready(self) -> tuple[ObservationData, np.ndarray]:
        return self.observation, self.mask

    def history(self) -> tuple[ActionIndex, ...]:
        return ()

    def step(self, _action: ActionIndex) -> _WorkerStep:
        turn = int(self.observation["player"].get("turn", 0)) + 1
        self.observation = _rollout_observation(turn)
        return _WorkerStep(
            self.observation,
            self.mask,
            0.0,
            False,
            False,
            None,
            (),
            None,
            False,
        )


class _FakeBatcher:
    feature_spec_version = 4
    model_config = ModelConfig(action_count=int(ACTION_COUNT))

    def infer(
        self,
        _slot: WorkerIndex,
        _feature: np.ndarray,
        _history: np.ndarray,
        _mask: np.ndarray,
    ) -> _InferenceResult:
        probabilities = np.zeros(int(ACTION_COUNT), dtype=np.float32)
        probabilities[0] = 1.0
        return _InferenceResult(probabilities, 0.0)


def test_rollout_reuses_next_feature_at_following_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encode_calls = 0

    def counting_encode(
        _observation: ObservationData, *, spec_version: FeatureSpecVersion
    ) -> np.ndarray:
        nonlocal encode_calls
        encode_calls += 1
        return np.full(feature_count(spec_version), encode_calls, dtype=np.float32)

    monkeypatch.setattr("dcss_rl.ppo.encode_observation", counting_encode)
    rollout = _collect_worker_rollout(
        cast(_Worker, _FakeWorker()),
        ScriptedMibePolicy(),
        batcher=cast(_InferenceBatcher, _FakeBatcher()),
        config=PpoConfig(rollout_length=RolloutLength(3), device="cpu"),
    )

    assert encode_calls == 4
    assert rollout.features[:, 0].tolist() == [1.0, 2.0, 3.0]
    assert rollout.next_deltas[:, 0].tolist() == [1.0, 1.0, 1.0]


def test_worker_run_root_does_not_include_unbounded_case_label() -> None:
    root = _worker_run_root(
        Path("/tmp/run"),
        WorkerIndex(47),
        EpisodeIndex(123),
        StartupAttemptIndex(2),
    )

    assert root == Path("/tmp/run/worker-47/episode-123-attempt-2")


def test_checkpoint_metadata_records_ui_budget_and_cumulative_overflows() -> None:
    config = PpoConfig(
        ui_interaction_budget=UiInteractionBudgetConfig(
            capacity=UiInteractionTokenCapacity(3.0),
            refill_per_turn=UiInteractionRefillPerTurn(0.5),
            overflow_cost=UiInteractionCost(0.125),
        ),
        device="cpu",
    )

    metadata = _checkpoint_metadata(
        config,
        updates=UpdateCount(2),
        mean_episode_return=4.0,
        action_history_length=ActionHistoryLength(1),
        ui_interaction_overflows=UiInteractionOverflowCount(7),
    )

    assert metadata.ui_interaction_capacity == 3.0
    assert metadata.ui_interaction_refill_per_turn == 0.5
    assert metadata.ui_interaction_cost == 0.125
    assert metadata.ui_interaction_overflows == 7


def test_inference_requests_are_padded_to_reproducible_fixed_shape() -> None:
    request = _InferenceRequest(
        WorkerIndex(1),
        np.asarray([1.0, 2.0], dtype=np.float32),
        np.asarray([3.0], dtype=np.float32),
        np.asarray([False, True, False], dtype=np.bool_),
        Future(),
    )

    batch = _fixed_inference_inputs([request], WorkerCount(4))

    assert batch.features.shape == (4, 2)
    assert batch.action_histories.shape == (4, 1)
    assert batch.masks.shape == (4, 3)
    np.testing.assert_array_equal(batch.features[1], request.feature)
    np.testing.assert_array_equal(batch.masks[1], request.mask)
    np.testing.assert_array_equal(batch.masks[0], [True, False, False])
    np.testing.assert_array_equal(batch.masks[2:], [[True, False, False]] * 2)


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


def test_action_warmup_can_select_companion_row_without_action_expansion() -> None:
    menu_a = action_to_index(Action.menu_select(Keycode(ord("a"))))

    indices = _warmup_action_indices(
        int(ACTION_COUNT), int(ACTION_COUNT), (Keycode(ord("a")),)
    )

    assert indices == (menu_a,)


def test_action_warmup_can_select_existing_structured_rows() -> None:
    indices = _warmup_action_indices(
        int(ACTION_COUNT),
        int(ACTION_COUNT),
        (),
        (ActionKind.ABILITIES, ActionKind.CANCEL),
    )

    assert set(indices) == {
        action_to_index(Action(ActionKind.ABILITIES)),
        action_to_index(Action(ActionKind.CANCEL)),
    }


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
        ModelConfig(
            action_count=4,
            hidden_size=2,
            feature_spec_version=FeatureSpecVersion(3),
        )
    )
    old_features = torch.randn(3, feature_count(FeatureSpecVersion(3)))
    old_outputs = model(old_features)

    migrated = align_feature_spec(model, FeatureSpecVersion(4))
    new_features = torch.cat((old_features, torch.zeros(3, 2)), dim=1)
    new_outputs = migrated(new_features)

    torch.testing.assert_close(old_outputs[0], new_outputs[0])
    torch.testing.assert_close(old_outputs[1], new_outputs[1])
    old_feature_count = feature_count(FeatureSpecVersion(3))
    torch.testing.assert_close(old_outputs[2], new_outputs[2][:, :old_feature_count])
    torch.testing.assert_close(new_outputs[2][:, old_feature_count:], torch.zeros(3, 2))
