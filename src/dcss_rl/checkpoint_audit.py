"""Read-only checkpoint coverage and exact selective-ownership audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict, cast

import torch
from torch import Tensor

from dcss_rl.actions import Action, ActionKind
from dcss_rl.env import ACTION_COUNT, action_to_index, index_to_action
from dcss_rl.learned import AbilityResidualVersion, ModelConfig
from dcss_rl.units import (
    ActionCount,
    ActionHistoryLength,
    ActionIndex,
    CheckpointId,
    FeatureSpecVersion,
    Keycode,
)

type TensorState = dict[str, Tensor]
type BoundaryObject = dict[str, object]


class CoverageData(TypedDict):
    samples: int
    targets: list[int] | tuple[int, ...]
    legal_exposures: list[int] | tuple[int, ...]


@dataclass(frozen=True, slots=True)
class NamedActionCount:
    action: str
    index: ActionIndex
    targets: ActionCount
    legal_exposures: ActionCount


@dataclass(frozen=True, slots=True)
class CoverageAudit:
    name: str
    samples: ActionCount
    actions: tuple[NamedActionCount, ...]


@dataclass(frozen=True, slots=True)
class CheckpointContents:
    sha256: CheckpointId
    config: ModelConfig
    state: TensorState
    coverage: tuple[CoverageAudit, ...]


@dataclass(frozen=True, slots=True)
class TensorComparison:
    changed_policy_rows: tuple[ActionIndex, ...]
    changed_policy_actions: tuple[str, ...]
    changed_non_policy_tensors: tuple[str, ...]
    incompatibilities: tuple[str, ...]
    allowed_policy_rows: tuple[ActionIndex, ...] | None
    ownership_passed: bool | None
    allowed_value_head: bool = False
    allowed_ability_residual: bool = False


@dataclass(frozen=True, slots=True)
class CheckpointAudit:
    path: str
    sha256: CheckpointId
    config: ModelConfig
    coverage: tuple[CoverageAudit, ...]
    reference_sha256: CheckpointId | None
    comparison: TensorComparison | None


def _object(value: object, label: str) -> BoundaryObject:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object with string keys")
    return cast(BoundaryObject, value)


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def action_name(index: ActionIndex) -> str:
    if index >= ACTION_COUNT:
        return f"unknown_catalog_index:{index}"
    action = index_to_action(index)
    if action.kind is ActionKind.MENU_SELECT:
        return f"menu_select:{chr(action.keycode or 0)!r}"
    return action.kind.value


def _coverage(name: str, value: object) -> CoverageAudit:
    raw = cast(CoverageData, _object(value, name))
    samples = _integer(raw.get("samples"), f"{name}.samples")
    targets, exposures = raw.get("targets"), raw.get("legal_exposures")
    if not isinstance(targets, (list, tuple)) or not isinstance(
        exposures, (list, tuple)
    ):
        raise ValueError(f"{name} needs target and legal-exposure vectors")
    if len(targets) != len(exposures) or samples < 0:
        raise ValueError(f"{name} has invalid coverage dimensions")
    target_counts = tuple(_integer(item, name) for item in targets)
    legal_counts = tuple(_integer(item, name) for item in exposures)
    if sum(target_counts) != samples or any(
        not 0 <= target <= legal <= samples
        for target, legal in zip(target_counts, legal_counts, strict=True)
    ):
        raise ValueError(f"{name} has inconsistent target/legal counts")
    named_indices = {
        action_to_index(Action(kind))
        for kind in ActionKind
        if kind is not ActionKind.MENU_SELECT
    } | {action_to_index(Action.menu_select(Keycode(ord(key)))) for key in ("a", "X")}
    # Include every targeted class alongside named important absent classes.
    named_indices.update(
        ActionIndex(i) for i, count in enumerate(target_counts) if count
    )
    return CoverageAudit(
        name,
        ActionCount(samples),
        tuple(
            NamedActionCount(
                action_name(index),
                index,
                ActionCount(target_counts[index]),
                ActionCount(legal_counts[index]),
            )
            for index in sorted(named_indices)
            if index < len(target_counts)
        ),
    )


def load_checkpoint_contents(path: Path) -> CheckpointContents:
    payload: object = torch.load(path, map_location="cpu", weights_only=True)
    raw = _object(payload, "checkpoint")
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported checkpoint schema")
    config = _object(raw.get("model_config"), "model_config")
    version = _integer(raw.get("feature_spec_version"), "feature_spec_version")
    if config.get("feature_spec_version", version) != version:
        raise ValueError("checkpoint feature specification disagrees with config")
    model_config = ModelConfig(
        action_count=_integer(config.get("action_count"), "action_count"),
        hidden_size=_integer(config.get("hidden_size"), "hidden_size"),
        feature_spec_version=FeatureSpecVersion(version),
        action_history_length=ActionHistoryLength(
            _integer(config.get("action_history_length", 0), "action_history_length")
        ),
        ability_residual_version=AbilityResidualVersion(
            _integer(
                config.get("ability_residual_version", 0), "ability_residual_version"
            )
        ),
    )
    state: TensorState = {}
    for name, tensor in _object(raw.get("model_state"), "model_state").items():
        if not isinstance(tensor, Tensor):
            raise ValueError(f"model_state entry {name} is not a tensor")
        state[name] = tensor
    if not state:
        raise ValueError("model_state is empty")
    metadata = _object(raw.get("training_metadata", {}), "training_metadata")
    coverage = tuple(
        _coverage(name, metadata[name])
        for name in (
            "anchor_coverage",
            "imitation_coverage",
            "training_coverage",
            "validation_coverage",
        )
        if metadata.get(name) is not None
    )
    return CheckpointContents(
        CheckpointId(hashlib.sha256(path.read_bytes()).hexdigest()),
        model_config,
        state,
        coverage,
    )


def _bit_exact(left: Tensor, right: Tensor) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and torch.equal(
            left.contiguous().reshape(-1).view(torch.uint8),
            right.contiguous().reshape(-1).view(torch.uint8),
        )
    )


def compare_checkpoints(
    current: CheckpointContents,
    reference: CheckpointContents,
    allowed_rows: tuple[ActionIndex, ...] | None = None,
    *,
    allow_value_head: bool = False,
    allow_ability_residual: bool = False,
) -> TensorComparison:
    incompatible: list[str] = []
    if current.config != reference.config:
        incompatible.append("model configuration differs; no migration is performed")
    if allow_ability_residual:
        if current.config.ability_residual_version != 1:
            incompatible.append(
                "ability residual ownership requires residual version 1"
            )
        for checkpoint in (current, reference):
            for name, shape in (
                ("ability_residual_head.weight", (3, 5)),
                ("ability_residual_head.bias", (3,)),
            ):
                tensor = checkpoint.state.get(name)
                if tensor is None or tuple(tensor.shape) != shape:
                    incompatible.append(f"{name}: missing or invalid residual tensor")
    changed_rows: set[ActionIndex] = set()
    changed_non_policy: list[str] = []
    for name in sorted(current.state.keys() | reference.state.keys()):
        left, right = current.state.get(name), reference.state.get(name)
        if left is None or right is None:
            incompatible.append(f"{name}: tensor is missing from one checkpoint")
            continue
        if left.shape != right.shape or left.dtype != right.dtype:
            incompatible.append(f"{name}: shape or dtype differs")
            continue
        if _bit_exact(left, right):
            continue
        if name in {"policy_head.weight", "policy_head.bias"}:
            changed_rows.update(
                ActionIndex(row)
                for row in range(left.shape[0])
                if not _bit_exact(left[row], right[row])
            )
        else:
            changed_non_policy.append(name)
    if allowed_rows is not None and any(
        not 0 <= row < current.config.action_count for row in allowed_rows
    ):
        incompatible.append("declared allowed policy row is outside checkpoint catalog")
    allowed_non_policy = (
        {"value_head.weight", "value_head.bias"} if allow_value_head else set()
    )
    if allow_ability_residual:
        allowed_non_policy.update(
            {"ability_residual_head.weight", "ability_residual_head.bias"}
        )
    passed = (
        None
        if allowed_rows is None
        else (
            not incompatible
            and set(changed_non_policy).issubset(allowed_non_policy)
            and changed_rows.issubset(allowed_rows)
        )
    )
    rows = tuple(sorted(changed_rows))
    return TensorComparison(
        rows,
        tuple(action_name(row) for row in rows),
        tuple(changed_non_policy),
        tuple(incompatible),
        allowed_rows,
        passed,
        allow_value_head,
        allow_ability_residual,
    )


def audit_checkpoint(
    checkpoint: Path,
    reference: Path | None = None,
    *,
    allowed_rows: tuple[ActionIndex, ...] | None = None,
    allow_value_head: bool = False,
    allow_ability_residual: bool = False,
) -> CheckpointAudit:
    if (allow_value_head or allow_ability_residual) and allowed_rows is None:
        allowed_rows = ()
    if allowed_rows is not None and reference is None:
        raise ValueError("an allowed-row ownership contract requires --reference")
    current = load_checkpoint_contents(checkpoint)
    baseline = load_checkpoint_contents(reference) if reference is not None else None
    return CheckpointAudit(
        str(checkpoint),
        current.sha256,
        current.config,
        current.coverage,
        baseline.sha256 if baseline is not None else None,
        compare_checkpoints(
            current,
            baseline,
            allowed_rows,
            allow_value_head=allow_value_head,
            allow_ability_residual=allow_ability_residual,
        )
        if baseline is not None
        else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument(
        "--allow-action-kind",
        action="append",
        type=ActionKind,
        choices=[kind for kind in ActionKind if kind is not ActionKind.MENU_SELECT],
    )
    parser.add_argument("--allow-menu-key", action="append")
    parser.add_argument("--allow-value-head", action="store_true")
    parser.add_argument("--allow-ability-residual", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    allowed: tuple[ActionIndex, ...] | None = None
    if args.allow_action_kind is not None or args.allow_menu_key is not None:
        keys: list[Keycode] = []
        for key in args.allow_menu_key or ():
            if len(key) != 1 or ord(key) > 255:
                parser.error("--allow-menu-key requires one catalog character")
            keys.append(Keycode(ord(key)))
        allowed = tuple(
            sorted(
                {
                    *(
                        action_to_index(Action(kind))
                        for kind in args.allow_action_kind or ()
                    ),
                    *(action_to_index(Action.menu_select(key)) for key in keys),
                }
            )
        )
    try:
        report = audit_checkpoint(
            args.checkpoint,
            args.reference,
            allowed_rows=allowed,
            allow_value_head=args.allow_value_head,
            allow_ability_residual=args.allow_ability_residual,
        )
    except ValueError as error:
        parser.error(str(error))
    result = json.dumps(asdict(report), indent=2) + "\n"
    if args.output is not None:
        with args.output.open("x") as stream:
            stream.write(result)
    print(result, end="")
    if report.comparison is not None and report.comparison.ownership_passed is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
