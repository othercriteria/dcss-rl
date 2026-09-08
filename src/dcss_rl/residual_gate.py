"""Fail-closed confidence and checkpoint-ownership gate for residual probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import msgspec
import torch

from dcss_rl.ability_residual_probe import (
    OptimizerStep,
    ResidualMetadata,
    ResidualReport,
    ResidualSnapshot,
)
from dcss_rl.checkpoint_audit import (
    _bit_exact,
    _object,
    audit_checkpoint,
    load_checkpoint_contents,
)
from dcss_rl.learned import AbilityResidualVersion
from dcss_rl.units import Probability

_CONTEXTS = {"applicable", "inapplicable", "missing"}
_RESIDUAL = {"ability_residual_head.weight", "ability_residual_head.bias"}


@dataclass(frozen=True)
class SnapshotGate:
    step: OptimizerStep
    checkpoint: str
    failures: tuple[str, ...]


@dataclass(frozen=True)
class ResidualGateReport:
    report: str
    source_verified: bool
    snapshots: tuple[SnapshotGate, ...]
    first_qualifying_checkpoint: str | None
    scope: str = "Offline recorded-context gate only; no live evaluation or promotion."


def _probability(value: Probability | None) -> bool:
    return value is not None and math.isfinite(value) and 0 <= value <= 1


def confidence_failures(snapshot: ResidualSnapshot) -> tuple[str, ...]:
    failures: list[str] = []
    if not snapshot.base_tensors_exact:
        failures.append("base tensors changed")
    observations = {"non_menu": 0, "other_menu": 0}
    for label, split in (
        ("training", snapshot.training),
        ("validation", snapshot.validation),
    ):
        if {item.context for item in split.contexts} != _CONTEXTS or len(
            split.contexts
        ) != 3:
            failures.append(f"{label}: incomplete or duplicate target contexts")
        if split.unknown_excluded:
            failures.append(f"{label}: unknown ability states remain unaudited")
        for item in split.contexts:
            prefix = f"{label}/{item.context}"
            if item.count <= 0 or item.accuracy != 1:
                failures.append(f"{prefix}: missing examples or imperfect accuracy")
            for name, value, threshold, minimum in (
                ("mean target", item.target_probability_mean, 0.995, True),
                ("minimum target", item.target_probability_minimum, 0.99, True),
                ("maximum Renounce", item.renounce_x_probability_maximum, 1e-5, False),
            ):
                if (
                    not _probability(value)
                    or value is None
                    or (value < threshold if minimum else value > threshold)
                ):
                    failures.append(f"{prefix}: {name} gate failed")
            if item.context == "inapplicable":
                value = item.berserk_a_probability_maximum
                if not _probability(value) or value is None or value > 0.001:
                    failures.append(f"{prefix}: maximum Berserk gate failed")
        if {item.context for item in split.preservation} != set(observations) or len(
            split.preservation
        ) != 2:
            failures.append(f"{label}: incomplete preservation groups")
        for item in split.preservation:
            if item.context not in observations or item.count < 0:
                failures.append(f"{label}: invalid preservation group")
                continue
            observations[item.context] += item.count
            if item.count and not (
                item.logits_exact is True
                and item.probabilities_exact is True
                and item.argmax_exact is True
            ):
                failures.append(f"{label}/{item.context}: outputs changed")
    for context, count in observations.items():
        if not count:
            failures.append(f"{context}: no empirical preservation examples")
    return tuple(failures)


def audit_residual_gate(path: Path) -> ResidualGateReport:
    report = msgspec.json.decode(path.read_bytes(), type=ResidualReport)
    for evidence in (report.source, report.zero_enabled_source, *report.checkpoints):
        if (
            hashlib.sha256(Path(evidence.path).read_bytes()).hexdigest()
            != evidence.sha256
        ):
            raise ValueError(f"checkpoint hash changed: {evidence.path}")
    source = load_checkpoint_contents(Path(report.source.path))
    start = load_checkpoint_contents(Path(report.zero_enabled_source.path))
    if (
        not report.zero_enabled_outputs_exact
        or source.config.ability_residual_version != 0
        or replace(source.config, ability_residual_version=AbilityResidualVersion(1))
        != start.config
        or set(start.state) != set(source.state) | _RESIDUAL
        or any(
            not _bit_exact(tensor, start.state[name])
            for name, tensor in source.state.items()
        )
        or any(torch.count_nonzero(start.state[name]).item() for name in _RESIDUAL)
    ):
        raise ValueError("zero-enabled source identity was not established")
    snapshots = tuple(item for item in report.snapshots if item.step)
    if tuple(item.step for item in snapshots) != report.predeclared_steps or len(
        snapshots
    ) != len(report.checkpoints):
        raise ValueError(
            "snapshot/checkpoint schedule differs from the declared protocol"
        )
    results: list[SnapshotGate] = []
    for snapshot, checkpoint in zip(snapshots, report.checkpoints, strict=True):
        failures = list(confidence_failures(snapshot))
        payload = _object(
            torch.load(checkpoint.path, map_location="cpu", weights_only=True),
            "checkpoint",
        )
        metadata = msgspec.convert(
            payload.get("training_metadata"), type=ResidualMetadata
        )
        if (
            metadata.snapshot != snapshot
            or metadata.source != report.source
            or metadata.training_episodes != report.training_episodes
            or metadata.validation_episodes != report.validation_episodes
            or metadata.code != report.code
            or metadata.zero_enabled_outputs_exact != report.zero_enabled_outputs_exact
            or set(metadata.owned_parameters) != _RESIDUAL
            or metadata.training_coverage != report.training_coverage
            or metadata.validation_coverage != report.validation_coverage
            or metadata.predeclared_steps != report.predeclared_steps
            or metadata.objective != report.objective
            or metadata.competitor_margins != report.competitor_margins
        ):
            raise ValueError("report evidence disagrees with checkpoint metadata")
        audit = audit_checkpoint(
            Path(checkpoint.path),
            Path(report.zero_enabled_source.path),
            allow_ability_residual=True,
        )
        if audit.comparison is None or audit.comparison.ownership_passed is not True:
            failures.append("independent checkpoint ownership audit failed")
        results.append(SnapshotGate(snapshot.step, checkpoint.path, tuple(failures)))
    return ResidualGateReport(
        str(path),
        True,
        tuple(results),
        next((item.checkpoint for item in results if not item.failures), None),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    report = audit_residual_gate(args.report)
    encoded = json.dumps(asdict(report), indent=2) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(encoded)
    print(encoded, end="")
    if args.require_pass and report.first_qualifying_checkpoint is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
