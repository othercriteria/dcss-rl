import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from dcss_rl.ability_probe import (
    _A,
    _CANCEL,
    _ROWS,
    _V4_WIDTH,
    _X,
    ProbeData,
    ProbeSnapshot,
    ProbeStep,
    _first_step_gate_passes,
    _probe_columns,
    measure,
    run_probe,
    selective_step,
    unowned_parameters_exact,
    validate_training_paths,
)
from dcss_rl.coverage import replay_coverage
from dcss_rl.env import ACTION_COUNT
from dcss_rl.features import FEATURE_COUNT, FEATURE_SPEC_VERSION
from dcss_rl.learned import (
    CheckpointTrainingMetadata,
    LearnedPolicy,
    ModelConfig,
    SemanticActorCritic,
    save_checkpoint,
)
from dcss_rl.units import ActionHistoryLength, FeatureSpecVersion


def _data() -> ProbeData:
    features = torch.ones((3, FEATURE_COUNT))
    masks = torch.zeros((3, ACTION_COUNT), dtype=torch.bool)
    masks[:, list(_ROWS)] = True
    targets = torch.tensor([_A, _CANCEL, _CANCEL])
    return ProbeData(
        features,
        masks,
        targets,
        torch.tensor([True, False, False]),
        torch.tensor([False, True, False]),
        torch.tensor([True, True, False]),
        replay_coverage(np.asarray(targets, dtype=np.int64), masks.numpy()),
        (),
        (),
    )


def test_decision_margin_includes_third_legal_competitor() -> None:
    model = SemanticActorCritic(ModelConfig(ACTION_COUNT, hidden_size=4))
    with torch.no_grad():
        model.policy_head.weight.zero_()
        model.policy_head.bias.zero_()
        model.policy_head.bias[_CANCEL] = 8
        model.policy_head.bias[_X] = 12
    data = _data()
    metrics = measure(model, data, model(data.features)[0].detach())
    assert metrics.inapplicable_margin == 8
    assert metrics.inapplicable_decision_margin == -4
    assert metrics.inapplicable_accuracy == 0
    assert metrics.inapplicable_target_probability < 0.02
    assert metrics.inapplicable_cross_entropy > 4


def test_selective_step_preserves_weights_but_can_change_unowned_outputs() -> None:
    torch.manual_seed(0)
    model = SemanticActorCritic(ModelConfig(ACTION_COUNT, hidden_size=4))
    data = _data()
    baseline = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    with torch.no_grad():
        baseline_logits = model(data.features)[0]
    columns = tuple(range(_V4_WIDTH, FEATURE_COUNT))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
    selective_step(model, data, optimizer, baseline, columns)
    assert unowned_parameters_exact(model, baseline, columns)
    assert measure(model, data, baseline_logits).nonmenu_max_unowned_logit_change > 0
    with torch.no_grad():
        model.value_head.bias.add_(0.001)
    assert not unowned_parameters_exact(model, baseline, columns)


def test_probe_refuses_existing_output_before_loading_inputs(tmp_path: Path) -> None:
    with pytest.raises(FileExistsError):
        run_probe(tmp_path / "missing.pt", tmp_path / "missing", tmp_path)


def test_row_only_mode_requires_current_stateless_features() -> None:
    assert _probe_columns(ModelConfig(ACTION_COUNT), row_only_calibration=True) == ()
    with pytest.raises(ValueError, match="current feature spec"):
        _probe_columns(
            ModelConfig(ACTION_COUNT, feature_spec_version=FeatureSpecVersion(5)),
            row_only_calibration=True,
        )
    with pytest.raises(ValueError, match="current action count"):
        _probe_columns(ModelConfig(action_count=270), row_only_calibration=True)
    with pytest.raises(ValueError, match="stateless"):
        _probe_columns(
            ModelConfig(ACTION_COUNT, action_history_length=ActionHistoryLength(1)),
            row_only_calibration=True,
        )
    assert _probe_columns(
        ModelConfig(ACTION_COUNT, feature_spec_version=FeatureSpecVersion(4)),
        row_only_calibration=False,
    ) == tuple(range(_V4_WIDTH, FEATURE_COUNT))
    assert FEATURE_SPEC_VERSION == 6


def test_calibration_allows_first_step_class_tradeoff_but_legacy_gate_does_not() -> (
    None
):
    model = SemanticActorCritic(ModelConfig(ACTION_COUNT, hidden_size=4))
    data = _data()
    metrics = measure(model, data, model(data.features)[0].detach())
    initial = ProbeSnapshot(ProbeStep(0), metrics, metrics, True, 0.0)
    traded = replace(
        metrics,
        applicable_decision_margin=metrics.applicable_decision_margin + 1,
        inapplicable_decision_margin=metrics.inapplicable_decision_margin - 1,
    )
    current = ProbeSnapshot(ProbeStep(1), traded, traded, True, 0.0)
    assert not _first_step_gate_passes(initial, current, row_only_calibration=False)
    assert _first_step_gate_passes(initial, current, row_only_calibration=True)


def test_row_only_optimizer_preserves_encoder_and_unowned_logits() -> None:
    model = SemanticActorCritic(ModelConfig(ACTION_COUNT, hidden_size=4))
    data = _data()
    baseline = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    with torch.no_grad():
        logits = model(data.features)[0]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
    selective_step(model, data, optimizer, baseline, ())
    assert unowned_parameters_exact(model, baseline, ())
    assert measure(model, data, logits).nonmenu_max_unowned_logit_change == 0


@pytest.mark.parametrize("corrupt", [False, True])
def test_calibration_fixed_budget_and_immediate_ownership_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corrupt: bool,
) -> None:
    trajectories = tmp_path / "input"
    for index in range(4):
        attempt = trajectories / str(index) / "attempt-0"
        attempt.mkdir(parents=True)
        (attempt / "trajectory.jsonl").touch()
    model = SemanticActorCritic(ModelConfig(ACTION_COUNT, hidden_size=4))
    source = MagicMock()
    source.model = model
    source.checkpoint_action_count = int(ACTION_COUNT)
    source.checkpoint_id = "a" * 64
    monkeypatch.setattr("dcss_rl.ability_probe.LearnedPolicy", lambda *args: source)
    monkeypatch.setattr("dcss_rl.ability_probe.load_probe_data", lambda *args: _data())
    monkeypatch.setattr(
        "dcss_rl.ability_probe.validate_training_paths", lambda *args: None
    )
    steps = 0
    data = _data()
    metrics = measure(model, data, model(data.features)[0].detach())

    def traded_measure(*args: object) -> object:
        if not steps:
            return metrics
        return replace(
            metrics,
            applicable_decision_margin=metrics.applicable_decision_margin + 1,
            inapplicable_decision_margin=metrics.inapplicable_decision_margin - 1,
        )

    monkeypatch.setattr("dcss_rl.ability_probe.measure", traded_measure)

    def corrupt_second_step(*args: object) -> None:
        nonlocal steps
        steps += 1
        if corrupt and steps == 2:
            with torch.no_grad():
                model.value_head.bias.add_(1)

    monkeypatch.setattr("dcss_rl.ability_probe.selective_step", corrupt_second_step)
    report_path = run_probe(
        tmp_path / "source.pt",
        trajectories,
        tmp_path / "result",
        row_only_calibration=True,
    )
    report = json.loads(report_path.read_text())
    assert steps == (2 if corrupt else 256)
    assert [snapshot["step"] for snapshot in report["snapshots"]] == (
        [0, 1, 2] if corrupt else [0, 1, 16, 64, 256]
    )
    assert report["predeclared_steps"] == [1, 16, 64, 256]
    assert report["row_only_calibration"] is True
    assert report["owned_feature_columns"] == []
    assert (
        "preservation requirement failed"
        if corrupt
        else "no automatic live-rollout approval"
    ) in report["decision"]
    assert "probability" in report["gate_definition"]
    checkpoint = torch.load(
        report_path.parent / f"step-{steps:04d}.pt", weights_only=True
    )
    assert checkpoint["training_metadata"]["row_only_calibration"] is True
    assert (
        checkpoint["training_metadata"]["gate_definition"] == report["gate_definition"]
    )


def test_probe_rejects_nontraining_and_duplicate_seeds(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(json.dumps({"metadata": {"seed": -1}}) + "\n")
    with pytest.raises(ValueError, match="outside current training suite"):
        validate_training_paths((first,))
    for path in (first, second):
        path.write_text(json.dumps({"metadata": {"seed": 3001}}) + "\n")
    with pytest.raises(ValueError, match="unique episode seeds"):
        validate_training_paths((first, second))


def test_row_only_mode_rejects_saved_legacy_catalog_despite_loader_expansion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "legacy.pt"
    save_checkpoint(
        checkpoint,
        model=SemanticActorCritic(ModelConfig(action_count=270, hidden_size=4)),
        policy_id="legacy",
        training_metadata=CheckpointTrainingMetadata(
            "test",
            0,
            0,
            0,
            0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ),
    )
    loaded = LearnedPolicy(checkpoint)
    assert loaded.checkpoint_action_count == 270
    assert loaded.model.config.action_count == ACTION_COUNT
    trajectories = tmp_path / "input"
    for index in range(4):
        attempt = trajectories / str(index) / "attempt-0"
        attempt.mkdir(parents=True)
        (attempt / "trajectory.jsonl").touch()
    monkeypatch.setattr("dcss_rl.ability_probe.load_probe_data", lambda *args: _data())
    monkeypatch.setattr(
        "dcss_rl.ability_probe.validate_training_paths", lambda *args: None
    )
    with pytest.raises(ValueError, match="recorded current action count"):
        run_probe(
            checkpoint, trajectories, tmp_path / "result", row_only_calibration=True
        )
    assert not list((tmp_path / "result").glob("*.pt"))
