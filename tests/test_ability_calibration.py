import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from dcss_rl.ability_calibration import (
    audit_ability_calibration,
    context_calibration,
    split_training_paths,
)
from dcss_rl.actions import Action, ActionKind
from dcss_rl.env import ACTION_COUNT, action_to_index
from dcss_rl.features import FeatureVector, encode_observation
from dcss_rl.learned import ModelConfig, SemanticActorCritic
from dcss_rl.observation import MenuChoiceApplicability
from dcss_rl.schema import ObservationData
from dcss_rl.units import (
    ActionHistoryLength,
    FeatureSpecVersion,
    Keycode,
    TrajectoryCount,
)

_A = action_to_index(Action.menu_select(Keycode(ord("a"))))
_X = action_to_index(Action.menu_select(Keycode(ord("X"))))
_CANCEL = action_to_index(Action(ActionKind.CANCEL))


def test_masking_probability_partition_quantiles_and_berserk_maximum() -> None:
    logits = torch.full((2, ACTION_COUNT), 1000.0)
    masks = torch.zeros_like(logits, dtype=torch.bool)
    masks[:, [_A, _X, _CANCEL]] = True
    logits[:, [_A, _X, _CANCEL]] = torch.tensor(
        [[0.6, 0.3, 0.1], [0.2, 0.05, 0.75]]
    ).log()
    report = context_calibration(
        logits,
        masks,
        torch.tensor([_CANCEL, _CANCEL]),
        MenuChoiceApplicability.INAPPLICABLE,
    )
    assert report.target_probability is not None
    assert report.renounce_x_probability is not None
    assert report.other_wrong_legal_mass is not None
    assert report.berserk_a_probability is not None
    assert report.target_probability.mean == pytest.approx(0.425)
    assert report.target_probability.minimum == pytest.approx(0.1)
    assert report.target_probability.p05 == pytest.approx(0.1325)
    assert report.renounce_x_probability.mean == pytest.approx(0.175)
    assert report.other_wrong_legal_mass.mean == pytest.approx(0.4)
    assert report.berserk_a_probability.maximum == pytest.approx(0.6)
    assert report.argmax_accuracy == 0.5
    assert (
        report.target_probability.mean
        + report.renounce_x_probability.mean
        + report.other_wrong_legal_mass.mean
    ) == pytest.approx(1)
    with pytest.raises(ValueError, match="legal"):
        context_calibration(
            logits,
            masks,
            torch.zeros(2, dtype=torch.int64),
            MenuChoiceApplicability.INAPPLICABLE,
        )


def _state(context: str) -> ObservationData:
    state: ObservationData = {
        "player": {"hp": 20, "hp_max": 20},
        "cells": [],
        "messages": [],
        "menu": None,
        "input_mode": 1,
    }
    if context == "none":
        return state
    state["menu"] = {
        "type": "inventory" if context == "other" else "ability",
        "prompt": None,
        "choices": [],
    }
    if context != "missing":
        state["menu"]["choices"].append(
            {
                "keycode": 97,
                "text": "a - Berserk",
                "applicability": MenuChoiceApplicability(context).value
                if context in ("applicable", "inapplicable")
                else "unknown",
            }
        )
    state["menu"]["choices"].append({"keycode": 88, "text": "X - Renounce Religion"})
    return state


def _trajectory(root: Path, seed: int) -> Path:
    root.mkdir(parents=True)
    path = root / "trajectory.jsonl"
    states = [
        _state(context)
        for context in (
            "applicable",
            "inapplicable",
            "missing",
            "unknown",
            "other",
            "none",
            "applicable",
        )
    ]
    lines = [
        json.dumps({"metadata": {"seed": seed}, "initial": {"observation": states[0]}})
    ]
    lines.extend(
        json.dumps({"step": index, "action": {"kind": "cancel"}, "observation": state})
        for index, state in enumerate(states[1:])
    )
    path.write_text("\n".join(lines) + "\n")
    return path


def _checkpoint(
    path: Path, *, action_count: int = ACTION_COUNT, history: int = 0
) -> None:
    model = SemanticActorCritic(
        ModelConfig(
            action_count,
            hidden_size=4,
            feature_spec_version=FeatureSpecVersion(4),
            action_history_length=ActionHistoryLength(history),
        )
    )
    torch.save(
        {
            "schema_version": 1,
            "feature_spec_version": 4,
            "model_config": asdict(model.config),
            "model_state": model.state_dict(),
            "policy_id": "calibration-test",
            "training_metadata": {},
        },
        path,
    )


def test_contexts_seed_split_and_checkpoint_feature_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = tmp_path / "replay"
    for seed in range(3001, 3005):
        _trajectory(replay / str(seed), seed)
    checkpoint = tmp_path / "source.pt"
    _checkpoint(checkpoint)
    versions: set[FeatureSpecVersion] = set()

    def encode(
        observation: ObservationData, *, spec_version: FeatureSpecVersion
    ) -> FeatureVector:
        versions.add(spec_version)
        return encode_observation(observation, spec_version=spec_version)

    monkeypatch.setattr("dcss_rl.ability_calibration.encode_observation", encode)
    before = checkpoint.read_bytes()
    report = audit_ability_calibration(
        (checkpoint, checkpoint), replay, validation_episodes=TrajectoryCount(1)
    )
    result = report.checkpoints[0]
    assert versions == {4}
    assert result.config.feature_spec_version == 4
    assert {episode.seed for episode in result.training.episodes} == {3001, 3002, 3003}
    assert {episode.seed for episode in result.validation.episodes} == {3004}
    assert result.training.observations == 18
    assert result.training.missing_berserk == result.training.unknown_applicability == 3
    assert [context.samples for context in result.training.contexts] == [3, 3]
    assert [context.samples for context in result.validation.contexts] == [1, 1]
    assert result.preservation is not None
    assert result.preservation.non_menu.observations == 4
    assert result.preservation.other_menu.observations == 4
    assert result.preservation.non_menu.masked_probabilities_exact
    assert result.preservation.other_menu.masked_probabilities_exact
    assert checkpoint.read_bytes() == before


def test_split_rejects_duplicate_or_nontraining_seeds(tmp_path: Path) -> None:
    _trajectory(tmp_path / "a", 3001)
    second = _trajectory(tmp_path / "b", 3001)
    with pytest.raises(ValueError, match="unique episode seeds"):
        split_training_paths(tmp_path, TrajectoryCount(1))
    second.write_text(second.read_text().replace('"seed": 3001', '"seed": -1'))
    with pytest.raises(ValueError, match="outside current training suite"):
        split_training_paths(tmp_path, TrajectoryCount(1))


def test_absent_context_reports_missing_statistics() -> None:
    report = context_calibration(
        torch.empty((0, ACTION_COUNT)),
        torch.empty((0, ACTION_COUNT), dtype=torch.bool),
        torch.empty(0, dtype=torch.int64),
        MenuChoiceApplicability.APPLICABLE,
    )
    assert report.samples == 0
    assert report.target_probability is report.argmax_accuracy is None


@pytest.mark.parametrize(
    "history,action_count,message",
    [(1, ACTION_COUNT, "action-history"), (0, ACTION_COUNT - 1, "action catalog")],
)
def test_incompatible_checkpoints_fail_explicitly(
    tmp_path: Path,
    history: int,
    action_count: int,
    message: str,
) -> None:
    replay = tmp_path / "replay"
    _trajectory(replay / "a", 3001)
    _trajectory(replay / "b", 3002)
    checkpoint = tmp_path / "source.pt"
    _checkpoint(checkpoint, action_count=action_count, history=history)
    with pytest.raises(ValueError, match=message):
        audit_ability_calibration(
            (checkpoint,), replay, validation_episodes=TrajectoryCount(1)
        )
