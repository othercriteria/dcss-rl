"""Bounded, episode-split ability-menu learning and preservation experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NewType

import numpy as np
import torch
from torch import Tensor

from dcss_rl.actions import Action, ActionKind
from dcss_rl.coverage import ReplayCoverage, replay_coverage
from dcss_rl.env import ACTION_COUNT, action_to_index
from dcss_rl.evaluation import load_suite
from dcss_rl.features import FEATURE_COUNT, FEATURE_SPEC_VERSION, feature_count
from dcss_rl.learned import (
    CheckpointTrainingMetadata,
    LearnedPolicy,
    ModelConfig,
    SemanticActorCritic,
    align_action_count,
    align_feature_spec,
    save_checkpoint,
)
from dcss_rl.policy import ScriptedMibePolicy
from dcss_rl.ppo import _restore_warmup_parameters, _restrict_warmup_gradients
from dcss_rl.training import load_imitation_replay_episode
from dcss_rl.units import FeatureSpecVersion, GameSeed, Keycode

ProbeStep = NewType("ProbeStep", int)
_STEPS = tuple(ProbeStep(value) for value in (1, 16, 64, 256))
_A = action_to_index(Action.menu_select(Keycode(ord("a"))))
_X = action_to_index(Action.menu_select(Keycode(ord("X"))))
_CANCEL = action_to_index(Action(ActionKind.CANCEL))
_ROWS = (_A, _X, _CANCEL)
_V4_WIDTH = feature_count(FeatureSpecVersion(4))
_GATE_DEFINITION = (
    "Both training-context target-minus-max-legal-competitor margins "
    "must improve at step 1, and all unowned parameters remain exact. "
    "Corrects initial pairwise cancel-a gate: X dominates both choices. "
    "Balanced masked cross-entropy objective is unchanged."
)
_CALIBRATION_GATE_DEFINITION = (
    "Fixed 1/16/64/256-step row-only calibration; all encoder columns and "
    "unowned parameters must remain exact at every step. Already learned "
    "classes may trade margins at step 1, so no both-margins-improve gate applies. "
    "A separate probability and other-menu preservation audit is required; "
    "completion grants no automatic live-rollout approval."
)


@dataclass(frozen=True)
class ProbeData:
    features: Tensor
    masks: Tensor
    targets: Tensor
    applicable: Tensor
    inapplicable: Tensor
    menu: Tensor
    coverage: ReplayCoverage
    paths: tuple[str, ...]
    hashes: tuple[str, ...]


@dataclass(frozen=True)
class ProbeMetrics:
    applicable_count: int
    inapplicable_count: int
    other_menu_count: int
    applicable_margin: float
    inapplicable_margin: float
    applicable_decision_margin: float
    inapplicable_decision_margin: float
    applicable_target_probability: float
    inapplicable_target_probability: float
    applicable_cross_entropy: float
    inapplicable_cross_entropy: float
    applicable_accuracy: float
    inapplicable_accuracy: float
    nonmenu_count: int
    nonmenu_action_changes: int
    nonmenu_max_logit_change: float
    nonmenu_max_unowned_logit_change: float


@dataclass(frozen=True)
class ProbeSnapshot:
    step: ProbeStep
    train: ProbeMetrics
    validation: ProbeMetrics
    unowned_parameters_exact: bool
    elapsed_seconds: float


@dataclass(frozen=True)
class CodeEvidence:
    module: str
    sha256: str


@dataclass(frozen=True)
class ProbeMetadata(CheckpointTrainingMetadata):
    source_checkpoint_sha256: str
    training_paths: tuple[str, ...]
    validation_paths: tuple[str, ...]
    trajectory_sha256: tuple[str, ...]
    training_coverage: ReplayCoverage
    validation_coverage: ReplayCoverage
    snapshot: ProbeSnapshot
    code_evidence: tuple[CodeEvidence, ...]
    gate_definition: str
    row_only_calibration: bool = False


def validate_training_paths(paths: tuple[Path, ...]) -> None:
    allowed = {
        case.seed for case in load_suite(Path("configs/training-suite.json")).cases
    }
    seen: set[GameSeed] = set()
    for path in paths:
        with path.open() as stream:
            header: object = json.loads(stream.readline())
        if not isinstance(header, dict):
            raise ValueError(f"missing trajectory metadata: {path}")
        metadata = header.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"missing trajectory metadata: {path}")
        seed = metadata.get("seed")
        if type(seed) is not int or GameSeed(seed) not in allowed:
            raise ValueError(
                f"trajectory seed is outside current training suite: {path}"
            )
        if GameSeed(seed) in seen:
            raise ValueError(
                "probe requires unique episode seeds for disjoint splitting"
            )
        seen.add(GameSeed(seed))


def load_probe_data(paths: tuple[Path, ...]) -> ProbeData:
    episodes = tuple(
        load_imitation_replay_episode(path, teacher=ScriptedMibePolicy())
        for path in paths
    )
    features = np.stack([row for episode in episodes for row in episode.features])
    masks = np.stack([row for episode in episodes for row in episode.masks])
    targets = np.asarray(
        [row for episode in episodes for row in episode.actions], dtype=np.int64
    )
    # These positions are part of the append-only feature-v5 contract.
    menu = features[:, _V4_WIDTH + 2] == 1
    applicable = menu & (features[:, _V4_WIDTH + 4] == 1)
    inapplicable = menu & (features[:, _V4_WIDTH + 5] == 1)
    selected = applicable | inapplicable
    if not applicable.any() or not inapplicable.any():
        raise ValueError("each episode split must cover both ability contexts")
    if not np.all(targets[applicable] == _A):
        raise ValueError("applicable teacher targets must select menu a")
    if not np.all(targets[inapplicable] == _CANCEL):
        raise ValueError("inapplicable teacher targets must cancel")
    return ProbeData(
        torch.from_numpy(features),
        torch.from_numpy(masks),
        torch.from_numpy(targets),
        torch.from_numpy(applicable),
        torch.from_numpy(inapplicable),
        torch.from_numpy(menu),
        replay_coverage(targets[selected], masks[selected]),
        tuple(str(path) for path in paths),
        tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in paths),
    )


def _logits(model: SemanticActorCritic, data: ProbeData) -> Tensor:
    return model(data.features)[0]


def measure(
    model: SemanticActorCritic, data: ProbeData, baseline: Tensor
) -> ProbeMetrics:
    with torch.no_grad():
        logits = _logits(model, data)
        legal_logits = logits.masked_fill(~data.masks, -torch.inf)
        target_logits = legal_logits.gather(1, data.targets[:, None]).squeeze(1)
        competitors = legal_logits.clone()
        competitors.scatter_(1, data.targets[:, None], -torch.inf)
        decision_margin = target_logits - competitors.max(-1).values
        cross_entropy = torch.nn.functional.cross_entropy(
            legal_logits, data.targets, reduction="none"
        )
        probability = (-cross_entropy).exp()
        predictions = logits.masked_fill(~data.masks, -torch.inf).argmax(-1)
        old_predictions = baseline.masked_fill(~data.masks, -torch.inf).argmax(-1)
        nonmenu = ~data.menu
        unowned = torch.ones(ACTION_COUNT, dtype=torch.bool)
        unowned[list(_ROWS)] = False
        change = (logits - baseline).abs()[nonmenu]
        return ProbeMetrics(
            int(data.applicable.sum()),
            int(data.inapplicable.sum()),
            int((data.menu & ~(data.applicable | data.inapplicable)).sum()),
            float((logits[data.applicable, _A] - logits[data.applicable, _X]).mean()),
            float(
                (
                    logits[data.inapplicable, _CANCEL] - logits[data.inapplicable, _A]
                ).mean()
            ),
            float(decision_margin[data.applicable].mean()),
            float(decision_margin[data.inapplicable].mean()),
            float(probability[data.applicable].mean()),
            float(probability[data.inapplicable].mean()),
            float(cross_entropy[data.applicable].mean()),
            float(cross_entropy[data.inapplicable].mean()),
            float((predictions[data.applicable] == _A).float().mean()),
            float((predictions[data.inapplicable] == _CANCEL).float().mean()),
            int(nonmenu.sum()),
            int((predictions[nonmenu] != old_predictions[nonmenu]).sum()),
            float(change.max()) if change.numel() else 0.0,
            float(change[:, unowned].max()) if change.numel() else 0.0,
        )


def unowned_parameters_exact(
    model: SemanticActorCritic, baseline: dict[str, Tensor], columns: tuple[int, ...]
) -> bool:
    for name, current in model.state_dict().items():
        previous = baseline[name]
        if name in {"policy_head.weight", "policy_head.bias"}:
            rows = torch.ones(current.shape[0], dtype=torch.bool)
            rows[list(_ROWS)] = False
            current, previous = current[rows], previous[rows]
        elif name == "encoder.0.weight":
            frozen = torch.ones(current.shape[1], dtype=torch.bool)
            frozen[list(columns)] = False
            current, previous = current[:, frozen], previous[:, frozen]
        if not torch.equal(current, previous):
            return False
    return True


def selective_step(
    model: SemanticActorCritic,
    data: ProbeData,
    optimizer: torch.optim.Optimizer,
    baseline: dict[str, Tensor],
    columns: tuple[int, ...],
) -> None:
    optimizer.zero_grad(set_to_none=True)
    selected = data.applicable | data.inapplicable
    logits = model(data.features[selected])[0].masked_fill(
        ~data.masks[selected], -torch.inf
    )
    losses = torch.nn.functional.cross_entropy(
        logits, data.targets[selected], reduction="none"
    )
    applicable = data.applicable[selected]
    # Equal context weight; full replay minibatch avoids stochastic absent-class steps.
    loss = (losses[applicable].mean() + losses[~applicable].mean()) / 2
    loss.backward()
    rows, frozen_columns = _restrict_warmup_gradients(
        model, _ROWS, trainable_feature_indices=columns
    )
    optimizer.step()
    _restore_warmup_parameters(
        model,
        frozen_rows=rows,
        frozen_policy_weight=baseline["policy_head.weight"],
        frozen_policy_bias=baseline["policy_head.bias"],
        frozen_feature_columns=frozen_columns,
        frozen_encoder_weight=baseline["encoder.0.weight"],
    )


def _probe_columns(
    config: ModelConfig, *, row_only_calibration: bool
) -> tuple[int, ...]:
    if config.action_history_length:
        raise ValueError(
            "this conditional-choice probe requires a stateless checkpoint"
        )
    if row_only_calibration:
        if config.feature_spec_version != FEATURE_SPEC_VERSION:
            raise ValueError("row-only calibration requires the current feature spec")
        if config.action_count != ACTION_COUNT:
            raise ValueError("row-only calibration requires the current action count")
        return ()
    return tuple(range(feature_count(config.feature_spec_version), FEATURE_COUNT))


def _first_step_gate_passes(
    initial: ProbeSnapshot, current: ProbeSnapshot, *, row_only_calibration: bool
) -> bool:
    return row_only_calibration or (
        current.train.applicable_decision_margin
        > initial.train.applicable_decision_margin
        and current.train.inapplicable_decision_margin
        > initial.train.inapplicable_decision_margin
    )


def run_probe(
    checkpoint: Path,
    trajectories: Path,
    output: Path,
    *,
    row_only_calibration: bool = False,
) -> Path:
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    paths = tuple(sorted(trajectories.glob("*/attempt-0/trajectory.jsonl")))
    if len(paths) < 4:
        raise ValueError("probe requires at least four training-only episodes")
    validate_training_paths(paths)
    code_evidence = tuple(
        CodeEvidence(
            name,
            hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest(),
        )
        for name in (
            "ability_probe.py",
            "ppo.py",
            "features.py",
            "learned.py",
            "training.py",
        )
    )
    train, validation = load_probe_data(paths[:-2]), load_probe_data(paths[-2:])
    source = LearnedPolicy(checkpoint)
    # LearnedPolicy expands legacy catalogs on load; inspect the recorded count
    # before allowing a mode that promises no model-contract migration.
    if row_only_calibration and source.checkpoint_action_count != ACTION_COUNT:
        raise ValueError(
            "row-only calibration requires the recorded current action count"
        )
    columns = _probe_columns(
        source.model.config, row_only_calibration=row_only_calibration
    )
    gate_definition = (
        _CALIBRATION_GATE_DEFINITION if row_only_calibration else _GATE_DEFINITION
    )
    model = align_feature_spec(
        align_action_count(source.model, ACTION_COUNT), FEATURE_SPEC_VERSION
    )
    baseline = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        train_logits, validation_logits = (
            _logits(model, train),
            _logits(model, validation),
        )
    snapshots = [
        ProbeSnapshot(
            ProbeStep(0),
            measure(model, train, train_logits),
            measure(model, validation, validation_logits),
            True,
            time.monotonic() - started,
        )
    ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    decision = "stop: one-minibatch margin or preservation requirement failed"
    for step in range(1, max(_STEPS) + 1):
        selective_step(model, train, optimizer, baseline, columns)
        exact = (
            unowned_parameters_exact(model, baseline, columns)
            if row_only_calibration or step in _STEPS
            else True
        )
        if step not in _STEPS and exact:
            continue
        snapshot = ProbeSnapshot(
            ProbeStep(step),
            measure(model, train, train_logits),
            measure(model, validation, validation_logits),
            exact,
            time.monotonic() - started,
        )
        snapshots.append(snapshot)
        metadata = ProbeMetadata(
            "ability-menu-row-only-calibration"
            if row_only_calibration
            else "ability-menu-selective-probe",
            0,
            step,
            int(train.coverage.samples),
            len(train.paths),
            0.001,
            0.0,
            0.0,
            1.0,
            (
                snapshot.validation.applicable_accuracy
                + snapshot.validation.inapplicable_accuracy
            )
            / 2,
            str(source.checkpoint_id),
            train.paths,
            validation.paths,
            train.hashes + validation.hashes,
            train.coverage,
            validation.coverage,
            snapshot,
            code_evidence,
            gate_definition,
            row_only_calibration=row_only_calibration,
        )
        save_checkpoint(
            output / f"step-{step:04d}.pt",
            model=model,
            policy_id=f"ability-menu-probe-step-{step}",
            training_metadata=metadata,
        )
        if not snapshot.unowned_parameters_exact:
            decision = "stop: unowned parameter preservation requirement failed"
            break
        if step == 1 and not _first_step_gate_passes(
            snapshots[0], snapshot, row_only_calibration=row_only_calibration
        ):
            break
        decision = (
            "offline row-only calibration only; probability and other-menu "
            "preservation audits required; no automatic live-rollout approval"
            if row_only_calibration
            else "conditional-choice learning only; inspect validation and "
            "non-menu drift before rollout"
        )
    report = output / "report.json"
    report.write_text(
        json.dumps(
            {
                "source_checkpoint_sha256": str(source.checkpoint_id),
                "training_paths": train.paths,
                "validation_paths": validation.paths,
                "trajectory_sha256": train.hashes + validation.hashes,
                "training_coverage": asdict(train.coverage),
                "validation_coverage": asdict(validation.coverage),
                "predeclared_steps": _STEPS,
                "learning_rate": 0.001,
                "gate_definition": gate_definition,
                "row_only_calibration": row_only_calibration,
                "code_evidence": [asdict(item) for item in code_evidence],
                "owned_action_indices": _ROWS,
                "owned_feature_columns": columns,
                "snapshots": [asdict(snapshot) for snapshot in snapshots],
                "decision": decision,
            },
            indent=2,
        )
        + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row-only-calibration", action="store_true")
    args = parser.parse_args()
    print(
        run_probe(
            args.checkpoint,
            args.trajectories,
            args.output,
            row_only_calibration=args.row_only_calibration,
        )
    )


if __name__ == "__main__":
    main()
