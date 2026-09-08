import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import msgspec
import numpy as np
import pytest
import torch

from dcss_rl.ability_calibration import _source
from dcss_rl.ability_residual_probe import (
    ContextMetrics,
    OptimizerStep,
    PreservationMetrics,
    ResidualMetadata,
    ResidualObjective,
    ResidualReport,
    ResidualSnapshot,
    SplitMetrics,
)
from dcss_rl.coverage import replay_coverage
from dcss_rl.env import ACTION_COUNT
from dcss_rl.learned import (
    CheckpointTrainingMetadata,
    ModelConfig,
    SemanticActorCritic,
    enable_ability_residual,
    save_checkpoint,
)
from dcss_rl.residual_gate import audit_residual_gate, confidence_failures
from dcss_rl.units import ActionCount, Probability, Seconds


def _snapshot() -> ResidualSnapshot:
    contexts = tuple(
        ContextMetrics(
            context,
            ActionCount(1),
            Probability(1),
            Probability(0.995),
            Probability(0.99),
            Probability(1e-5),
            Probability(0.001),
        )
        for context in ("applicable", "inapplicable", "missing")
    )
    split = SplitMetrics(
        contexts,
        ActionCount(0),
        (
            PreservationMetrics("non_menu", ActionCount(1), True, True, True),
            PreservationMetrics("other_menu", ActionCount(1), True, True, True),
        ),
    )
    return ResidualSnapshot(OptimizerStep(1), split, split, True, Seconds(0.1))


@dataclass(frozen=True)
class GateFixture:
    path: Path
    report: ResidualReport
    model: SemanticActorCritic
    metadata: ResidualMetadata


def _write_report(path: Path, report: ResidualReport) -> None:
    path.write_text(json.dumps(asdict(report)) + "\n")


def _fixture(root: Path) -> GateFixture:
    source = SemanticActorCritic(ModelConfig(ACTION_COUNT, hidden_size=4))
    source_path = root / "source.pt"
    save_checkpoint(
        source_path,
        model=source,
        policy_id="source",
        training_metadata=CheckpointTrainingMetadata(
            "test", 0, 0, 0, 0, 0.1, 0, 0, 1, 0
        ),
    )
    model = enable_ability_residual(source)
    start = root / "start.pt"
    save_checkpoint(
        start,
        model=model,
        policy_id="start",
        training_metadata=CheckpointTrainingMetadata(
            "test", 0, 0, 0, 0, 0.1, 0, 0, 1, 0
        ),
    )
    coverage = replay_coverage(
        np.array([0], dtype=np.int64), np.ones((1, ACTION_COUNT), dtype=np.bool_)
    )
    metadata = ResidualMetadata(
        "test",
        0,
        1,
        1,
        0,
        0.1,
        0,
        0,
        1,
        1,
        _source(source_path),
        (),
        (),
        (),
        coverage,
        coverage,
        True,
        _snapshot(),
        ("ability_residual_head.weight", "ability_residual_head.bias"),
        (OptimizerStep(1),),
    )
    assert model.ability_residual_head is not None
    with torch.no_grad():
        model.ability_residual_head.bias.add_(0.25)
    candidate = root / "step-0001.pt"
    save_checkpoint(
        candidate, model=model, policy_id="candidate", training_metadata=metadata
    )
    report = ResidualReport(
        _source(source_path),
        _source(start),
        (),
        (),
        (),
        coverage,
        coverage,
        True,
        (_snapshot(),),
        (_source(candidate),),
        (OptimizerStep(1),),
        0.1,
        18,
        "test",
    )
    path = root / "report.json"
    _write_report(path, report)
    return GateFixture(path, report, model, metadata)


def test_confidence_thresholds_are_inclusive() -> None:
    assert confidence_failures(_snapshot()) == ()


@pytest.mark.parametrize(
    "change",
    (
        "missing",
        "unknown",
        "nan",
        "infinity",
        "mean",
        "minimum",
        "renounce",
        "berserk",
        "accuracy",
        "base",
    ),
)
def test_confidence_fails_closed(change: str) -> None:
    snapshot = _snapshot()
    split = snapshot.training
    context = split.contexts[1]
    if change == "missing":
        split = replace(split, contexts=split.contexts[:2])
    elif change == "unknown":
        split = replace(split, unknown_excluded=ActionCount(1))
    elif change == "base":
        snapshot = replace(snapshot, base_tensors_exact=False)
    else:
        fields = {
            "nan": ("target_probability_mean", float("nan")),
            "infinity": ("renounce_x_probability_maximum", float("inf")),
            "mean": ("target_probability_mean", 0.994999),
            "minimum": ("target_probability_minimum", 0.989999),
            "renounce": ("renounce_x_probability_maximum", 0.000010001),
            "berserk": ("berserk_a_probability_maximum", 0.001001),
            "accuracy": ("accuracy", 0.999),
        }
        field, value = fields[change]
        context = replace(context, **{field: Probability(value)})
        split = replace(split, contexts=(split.contexts[0], context, split.contexts[2]))
    assert confidence_failures(replace(snapshot, training=split))


def test_other_menu_requires_observed_exact_preservation() -> None:
    snapshot = _snapshot()
    empty = PreservationMetrics("other_menu", ActionCount(0), None, None, None)
    split = replace(
        snapshot.training, preservation=(snapshot.training.preservation[0], empty)
    )
    assert confidence_failures(replace(snapshot, training=split, validation=split))
    assert confidence_failures(replace(snapshot, validation=split)) == ()
    drift = replace(snapshot.training.preservation[1], probabilities_exact=False)
    split = replace(
        snapshot.training, preservation=(snapshot.training.preservation[0], drift)
    )
    assert confidence_failures(replace(snapshot, training=split))


def test_actual_residual_only_checkpoint_qualifies(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    gate = audit_residual_gate(fixture.path)
    assert gate.source_verified
    assert gate.first_qualifying_checkpoint == fixture.report.checkpoints[0].path
    assert gate.snapshots[0].failures == ()


def test_legacy_ce_records_decode_defaults(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    for record, domain in (
        (fixture.report, ResidualReport),
        (fixture.metadata, ResidualMetadata),
    ):
        payload = asdict(record)
        del payload["objective"], payload["competitor_margins"]
        restored = msgspec.json.decode(json.dumps(payload), type=domain)
        assert restored.objective == ResidualObjective.CE
        assert restored.competitor_margins == ()


def test_report_metrics_must_match_checkpoint(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    snapshot = _snapshot()
    first = replace(
        snapshot.training.contexts[0], target_probability_mean=Probability(1)
    )
    split = replace(
        snapshot.training, contexts=(first, *snapshot.training.contexts[1:])
    )
    snapshot = replace(snapshot, training=split)
    _write_report(fixture.path, replace(fixture.report, snapshots=(snapshot,)))
    with pytest.raises(ValueError, match="metadata"):
        audit_residual_gate(fixture.path)


def test_checkpoint_tampering_fails_hash_check(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    path = Path(fixture.report.checkpoints[0].path)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash"):
        audit_residual_gate(fixture.path)


@pytest.mark.parametrize("target", ("source", "start", "start-residual", "candidate"))
def test_base_mutation_cannot_pass_even_with_refreshed_file_hash(
    tmp_path: Path, target: str
) -> None:
    fixture = _fixture(tmp_path)
    if target == "candidate":
        with torch.no_grad():
            fixture.model.value_head.bias.add_(1)
        path = Path(fixture.report.checkpoints[0].path)
        save_checkpoint(
            path,
            model=fixture.model,
            policy_id="mutated",
            training_metadata=fixture.metadata,
        )
        report = replace(fixture.report, checkpoints=(_source(path),))
    else:
        path = Path(
            fixture.report.source.path
            if target == "source"
            else fixture.report.zero_enabled_source.path
        )
        from dcss_rl.learned import LearnedPolicy

        model = LearnedPolicy(path, device="cpu").model
        with torch.no_grad():
            if target == "start-residual":
                assert model.ability_residual_head is not None
                model.ability_residual_head.bias.add_(1)
            else:
                model.value_head.bias.add_(1)
        save_checkpoint(
            path, model=model, policy_id="mutated", training_metadata=fixture.metadata
        )
        report = replace(
            fixture.report,
            **{
                ("source" if target == "source" else "zero_enabled_source"): _source(
                    path
                )
            },
        )
    _write_report(fixture.path, report)
    if target == "candidate":
        gate = audit_residual_gate(fixture.path)
        assert gate.first_qualifying_checkpoint is None
        assert any("ownership" in failure for failure in gate.snapshots[0].failures)
    else:
        with pytest.raises(ValueError, match="source identity"):
            audit_residual_gate(fixture.path)
