import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from dcss_rl.actions import Action, ActionKind
from dcss_rl.env import ACTION_COUNT, action_to_index
from dcss_rl.learned import ModelConfig, SemanticActorCritic
from dcss_rl.schema import JsonObject, ObservationData
from dcss_rl.training_rollout_compare import (
    RolloutComparison,
    _episodes,
    compare_training_rollouts,
)
from dcss_rl.trajectory import observation_delta
from dcss_rl.units import StepLimit, WorkerCount


def _state(hp: int, colour: int = 7) -> ObservationData:
    return {
        "player": {"hp": hp, "hp_max": 20},
        "cells": [{"x": 0, "y": 0, "g": "@", "col": colour}],
        "messages": [],
        "menu": None,
        "input_mode": 1,
    }


def _run(
    root: Path,
    *,
    raw_tag: str = "same",
    bootstrap_hp: int = 20,
    final_hp: int = 0,
    weight_change: bool = False,
    colour: int = 7,
    sampling: bool = False,
) -> None:
    root.mkdir()
    checkpoints = root / "collector-checkpoints"
    checkpoints.mkdir()
    torch.manual_seed(0)
    model = SemanticActorCritic(ModelConfig(ACTION_COUNT, hidden_size=4))
    if weight_change:
        with torch.no_grad():
            model.value_head.bias.add_(1)
    checkpoint = checkpoints / "update-0000.pt"
    torch.save(
        {
            "schema_version": 1,
            "feature_spec_version": model.config.feature_spec_version,
            "model_config": asdict(model.config),
            "model_state": model.state_dict(),
            "training_metadata": {
                "return_boundary": "continuing-reset",
                "worker_count": 2,
                "rollout_steps": 2,
            },
            "policy_id": root.name,
        },
        checkpoint,
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    for worker in range(2):
        for episode in range(2):
            directory = root / f"worker-{worker}" / f"episode-{episode}-attempt-0"
            directory.mkdir(parents=True)
            initial = _state(20 if episode == 0 else bootstrap_hp, colour)
            header = {
                "type": "episode",
                "schema_version": 2,
                "metadata": {"seed": 3000 + worker, "agent_id": root.name},
                "initial": {
                    "observation": initial,
                    "raw_messages": [{"text": raw_tag}],
                    "setup_keycodes": [99],
                },
            }
            lines = [json.dumps(header)]
            previous = initial
            if episode == 0:
                for step in range(2):
                    current = _state(19 if step == 0 else final_hp, colour)
                    action = Action(ActionKind.WAIT)
                    lines.append(
                        json.dumps(
                            {
                                "type": "transition",
                                "step": step,
                                "action_index": action_to_index(action),
                                "action": action.to_dict(),
                                "emitted_keycodes": [46],
                                "raw_messages": [{"text": raw_tag}],
                                "terminated": step == 1,
                                "truncated": False,
                                "reward": 0.0,
                                "outcome": "dead" if step == 1 else None,
                                "collection_provenance": {
                                    "update": 1,
                                    "checkpoint_sha256": digest,
                                },
                                "observation_delta": observation_delta(
                                    previous, current
                                ),
                            }
                        )
                    )
                    previous = current
            if sampling:
                probabilities = [0.0] * ACTION_COUNT
                probabilities[action_to_index(Action(ActionKind.WAIT))] = 1.0
                for index in range(1, len(lines)):
                    row = json.loads(lines[index])
                    row["sampling_evidence"] = {
                        "probabilities": probabilities,
                        "rng_state_sha256": "a" * 64,
                    }
                    lines[index] = json.dumps(row)
            (directory / "trajectory.jsonl").write_text("\n".join(lines) + "\n")


def _compare(left: Path, right: Path) -> RolloutComparison:
    return compare_training_rollouts(
        left, right, workers=WorkerCount(2), steps=StepLimit(2)
    )


def test_serialization_provenance_differs_but_models_and_rollouts_match(
    tmp_path: Path,
) -> None:
    left, right = tmp_path / "left", tmp_path / "right"
    _run(left)
    _run(right)
    result = _compare(left, right)
    assert result.reference_checkpoint_sha256 != result.candidate_checkpoint_sha256
    assert result.initial_model_equal and result.matched and result.raw_identical
    assert result.cases[0].bootstrap_present == (True, True)
    assert result.cases[0].reset_metadata_differences == ("agent_id",)
    assert result.sampling_identical is None
    assert result.policy_inputs_identical
    assert not result.policy_rollout_matched


@pytest.mark.parametrize(
    "change,field",
    [("raw", "raw"), ("bootstrap", "bootstrap_semantic"), ("poststate", "after")],
)
def test_raw_terminal_poststate_and_bootstrap_are_distinct(
    tmp_path: Path, change: str, field: str
) -> None:
    left, right = tmp_path / "left", tmp_path / "right"
    _run(left)
    _run(
        right,
        raw_tag="changed" if change == "raw" else "same",
        bootstrap_hp=12 if change == "bootstrap" else 20,
        final_hp=1 if change == "poststate" else 0,
    )
    result = _compare(left, right)
    assert any(
        item.field == field and item.count > 0 for item in result.cases[0].differences
    )
    assert result.matched is (change == "raw")
    assert result.raw_identical is (change != "raw")


def test_different_initial_weights_fail_even_if_recorded_states_match(
    tmp_path: Path,
) -> None:
    left, right = tmp_path / "left", tmp_path / "right"
    _run(left)
    _run(right, weight_change=True)
    result = _compare(left, right)
    assert not result.initial_model_equal and not result.matched
    assert result.cases[0].matched


@pytest.mark.parametrize(
    "failure",
    ["missing_worker", "retry", "missing_bootstrap", "incomplete", "wrong_provenance"],
)
def test_incomplete_or_ambiguous_recordings_fail_closed(
    tmp_path: Path, failure: str
) -> None:
    left, right = tmp_path / "left", tmp_path / "right"
    _run(left)
    _run(right)
    if failure == "missing_worker":
        (right / "worker-1").rename(right / "not-a-worker")
    elif failure == "retry":
        (right / "worker-0/episode-0-attempt-1").mkdir()
    elif failure == "missing_bootstrap":
        (right / "worker-0/episode-1-attempt-0/trajectory.jsonl").unlink()
    else:
        path = right / "worker-0/episode-0-attempt-0/trajectory.jsonl"
        lines = path.read_text().splitlines()
        if failure == "incomplete":
            lines.pop()
        else:
            lines[1] = lines[1].replace(
                '"checkpoint_sha256": "', '"checkpoint_sha256": "incorrect-'
            )
        path.write_text("\n".join(lines) + "\n")
    with pytest.raises((ValueError, OSError)):
        _compare(left, right)


@pytest.mark.parametrize("key,value", [("reward", 3.0), ("seed", 9999)])
def test_reward_or_seed_change_invalidates_matching(
    tmp_path: Path, key: str, value: float
) -> None:
    left, right = tmp_path / "left", tmp_path / "right"
    _run(left)
    _run(right)
    path = right / "worker-0/episode-0-attempt-0/trajectory.jsonl"
    lines = path.read_text().splitlines()
    if key == "reward":
        row: JsonObject = json.loads(lines[1])
        row[key] = value
        lines[1] = json.dumps(row)
    else:
        lines[0] = lines[0].replace('"seed": 3000', '"seed": 9999')
    path.write_text("\n".join(lines) + "\n")
    assert not _compare(left, right).matched


def test_episode_order_is_numeric(tmp_path: Path) -> None:
    for episode in reversed(range(12)):
        (tmp_path / f"episode-{episode}-attempt-0").mkdir()
    assert [path.parent.name for path in _episodes(tmp_path)] == [
        f"episode-{episode}-attempt-0" for episode in range(12)
    ]


def test_one_sided_sampling_evidence_is_not_reported_equal(tmp_path: Path) -> None:
    left, right = tmp_path / "left", tmp_path / "right"
    _run(left)
    _run(right)
    path = right / "worker-0/episode-0-attempt-0/trajectory.jsonl"
    lines = path.read_text().splitlines()
    row = json.loads(lines[1])
    row["sampling_evidence"] = {
        "probabilities": [0.1, 0.9],
        "rng_state_sha256": "a" * 64,
    }
    lines[1] = json.dumps(row)
    path.write_text("\n".join(lines) + "\n")
    result = _compare(left, right)
    assert result.matched
    assert result.sampling_identical is False
    assert any(
        item.field == "probabilities" and item.count == 1
        for item in result.cases[0].differences
    )


@pytest.mark.parametrize("colour,final_hp,inputs_equal", [(3, 0, True), (7, 1, False)])
def test_strict_semantics_and_policy_inputs_remain_separate(
    tmp_path: Path, colour: int, final_hp: int, inputs_equal: bool
) -> None:
    left, right = tmp_path / "left", tmp_path / "right"
    _run(left, sampling=True)
    _run(right, colour=colour, final_hp=final_hp, sampling=True)
    result = _compare(left, right)
    assert not result.matched
    assert result.policy_inputs_identical is inputs_equal
    assert result.policy_rollout_matched is inputs_equal
    assert result.sampling_identical is True
    assert result.reference_feature_version == result.candidate_feature_version


def test_history_models_are_explicitly_unsupported(tmp_path: Path) -> None:
    left, right = tmp_path / "left", tmp_path / "right"
    _run(left)
    _run(right)
    path = right / "collector-checkpoints/update-0000.pt"
    payload = torch.load(path, weights_only=True)
    payload["model_config"]["action_history_length"] = 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="action-history"):
        _compare(left, right)
