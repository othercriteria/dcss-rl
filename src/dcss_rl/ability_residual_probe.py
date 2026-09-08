"""Fixed-budget, training-only three-context ability residual experiment."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from enum import IntEnum, StrEnum
from itertools import pairwise
from pathlib import Path
from typing import NewType, cast

import numpy as np
import torch
from torch import Tensor

from dcss_rl.ability_calibration import EpisodeEvidence, SourceEvidence, _source
from dcss_rl.ability_probe import validate_training_paths
from dcss_rl.actions import Action, ActionKind
from dcss_rl.coverage import ReplayCoverage, replay_coverage
from dcss_rl.env import ACTION_COUNT, action_to_index
from dcss_rl.features import FEATURE_SPEC_VERSION, encode_observation
from dcss_rl.learned import (
    CheckpointTrainingMetadata,
    LearnedPolicy,
    SemanticActorCritic,
    enable_ability_residual,
    save_checkpoint,
)
from dcss_rl.replay import replay_frames
from dcss_rl.schema import JsonObject
from dcss_rl.training import _training_action_mask
from dcss_rl.units import (
    ActionCount,
    ActionIndex,
    GameSeed,
    Keycode,
    Probability,
    Seconds,
)

OptimizerStep = NewType("OptimizerStep", int)
_STEPS = tuple(OptimizerStep(step) for step in (1, 16, 64, 256))
_LEARNING_RATE = 0.1
_A = action_to_index(Action.menu_select(Keycode(ord("a"))))
_X = action_to_index(Action.menu_select(Keycode(ord("X"))))
_CANCEL = action_to_index(Action(ActionKind.CANCEL))
_OWNED = frozenset(("ability_residual_head.weight", "ability_residual_head.bias"))


class MenuContext(IntEnum):
    APPLICABLE = 0
    INAPPLICABLE = 1
    MISSING = 2
    UNKNOWN = 3
    NON_MENU = 4
    OTHER_MENU = 5


class ResidualObjective(StrEnum):
    CE = "ce"
    WORST_MARGIN = "worst-margin"


@dataclass(frozen=True)
class CompetitorMargin:
    context: str
    target: ActionIndex
    competitor: ActionIndex
    relative_odds_cap: float
    required_logit_gap: float


def objective_margins(objective: ResidualObjective) -> tuple[CompetitorMargin, ...]:
    if objective is ResidualObjective.CE:
        return ()
    if objective is not ResidualObjective.WORST_MARGIN:
        raise ValueError("unsupported residual objective")
    x_cap = 1e-5
    cancel_cap = (1 - 0.995) / 0.995 - x_cap
    return tuple(
        CompetitorMargin(context.name.lower(), target, competitor, cap, -math.log(cap))
        for context, target, competitor, cap in (
            (MenuContext.APPLICABLE, _A, _X, x_cap),
            (MenuContext.APPLICABLE, _A, _CANCEL, cancel_cap),
            (MenuContext.INAPPLICABLE, _CANCEL, _X, x_cap),
            (MenuContext.INAPPLICABLE, _CANCEL, _A, 1e-3),
            (MenuContext.MISSING, _CANCEL, _X, x_cap),
        )
    )


_TARGET_CONTEXTS = (
    MenuContext.APPLICABLE,
    MenuContext.INAPPLICABLE,
    MenuContext.MISSING,
)


@dataclass(frozen=True)
class ResidualData:
    features: Tensor
    masks: Tensor
    targets: Tensor
    contexts: Tensor
    episodes: tuple[EpisodeEvidence, ...]


@dataclass(frozen=True)
class ContextMetrics:
    context: str
    count: ActionCount
    accuracy: Probability | None
    target_probability_mean: Probability | None
    target_probability_minimum: Probability | None
    renounce_x_probability_maximum: Probability | None
    berserk_a_probability_maximum: Probability | None


@dataclass(frozen=True)
class PreservationMetrics:
    context: str
    count: ActionCount
    logits_exact: bool | None
    probabilities_exact: bool | None
    argmax_exact: bool | None


@dataclass(frozen=True)
class SplitMetrics:
    contexts: tuple[ContextMetrics, ...]
    unknown_excluded: ActionCount
    preservation: tuple[PreservationMetrics, ...]


@dataclass(frozen=True)
class ResidualSnapshot:
    step: OptimizerStep
    training: SplitMetrics
    validation: SplitMetrics
    base_tensors_exact: bool
    elapsed_seconds: Seconds


@dataclass(frozen=True)
class ResidualMetadata(CheckpointTrainingMetadata):
    source: SourceEvidence
    training_episodes: tuple[EpisodeEvidence, ...]
    validation_episodes: tuple[EpisodeEvidence, ...]
    code: tuple[SourceEvidence, ...]
    training_coverage: ReplayCoverage
    validation_coverage: ReplayCoverage
    zero_enabled_outputs_exact: bool
    snapshot: ResidualSnapshot
    owned_parameters: tuple[str, ...]
    predeclared_steps: tuple[OptimizerStep, ...]
    # Checkpoint metadata must contain primitives for torch weights_only loading.
    objective: str = ResidualObjective.CE.value
    competitor_margins: tuple[CompetitorMargin, ...] = ()


@dataclass(frozen=True)
class ResidualReport:
    source: SourceEvidence
    zero_enabled_source: SourceEvidence
    training_episodes: tuple[EpisodeEvidence, ...]
    validation_episodes: tuple[EpisodeEvidence, ...]
    code: tuple[SourceEvidence, ...]
    training_coverage: ReplayCoverage
    validation_coverage: ReplayCoverage
    zero_enabled_outputs_exact: bool
    snapshots: tuple[ResidualSnapshot, ...]
    checkpoints: tuple[SourceEvidence, ...]
    predeclared_steps: tuple[OptimizerStep, ...]
    learning_rate: float
    trainable_parameters: int
    protocol: str
    objective: ResidualObjective = ResidualObjective.CE
    competitor_margins: tuple[CompetitorMargin, ...] = ()


def load_residual_data(paths: tuple[Path, ...]) -> ResidualData:
    features, masks, targets, contexts = [], [], [], []
    episodes: list[EpisodeEvidence] = []
    for path in paths:
        with path.open() as stream:
            header = cast(JsonObject, json.loads(stream.readline()))
        metadata = header.get("metadata")
        if (
            not isinstance(metadata, dict)
            or type(seed := metadata.get("seed")) is not int
        ):
            raise ValueError("residual replay requires seeded metadata")
        episodes.append(EpisodeEvidence(_source(path), GameSeed(seed)))
        for before, _after in pairwise(replay_frames(path)):
            state = before.observation
            menu = state["menu"]
            target = _CANCEL
            if menu is None:
                context = MenuContext.NON_MENU
            elif menu["type"] != "ability":
                context = MenuContext.OTHER_MENU
            else:
                choice = next(
                    (
                        item
                        for item in menu["choices"]
                        if "berserk" in item["text"].casefold()
                    ),
                    None,
                )
                if choice is None:
                    context = MenuContext.MISSING
                elif choice.get("applicability") == "applicable":
                    if choice["keycode"] != ord("a"):
                        raise ValueError("v1 residual requires Berserk key a")
                    context, target = MenuContext.APPLICABLE, _A
                elif choice.get("applicability") == "inapplicable":
                    context = MenuContext.INAPPLICABLE
                else:
                    context = MenuContext.UNKNOWN
            features.append(
                encode_observation(state, spec_version=FEATURE_SPEC_VERSION)
            )
            masks.append(_training_action_mask(state))
            targets.append(target)
            contexts.append(context)
    if not features:
        raise ValueError("residual replay contains no preaction states")
    return ResidualData(
        torch.from_numpy(np.stack(features)),
        torch.from_numpy(np.stack(masks)),
        torch.tensor(targets, dtype=torch.int64),
        torch.tensor(contexts, dtype=torch.int64),
        tuple(episodes),
    )


def base_tensors_exact(model: SemanticActorCritic, baseline: dict[str, Tensor]) -> bool:
    current = model.state_dict()
    return set(current) - _OWNED == set(baseline) and all(
        torch.equal(
            current[name].contiguous().view(torch.uint8),
            tensor.contiguous().view(torch.uint8),
        )
        for name, tensor in baseline.items()
    )


def selected_coverage(data: ResidualData) -> ReplayCoverage:
    selected = data.contexts <= MenuContext.MISSING
    return replay_coverage(data.targets[selected].numpy(), data.masks[selected].numpy())


def outputs_exact(
    reference: SemanticActorCritic, candidate: SemanticActorCritic, data: ResidualData
) -> bool:
    with torch.inference_mode():
        return all(
            torch.equal(before, after)
            for before, after in zip(
                reference(data.features), candidate(data.features), strict=True
            )
        )


def measure_residual(
    model: SemanticActorCritic, data: ResidualData, baseline: Tensor
) -> SplitMetrics:
    with torch.inference_mode():
        logits = model(data.features)[0]
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite residual logits")
        probabilities = logits.masked_fill(~data.masks, -torch.inf).softmax(-1)
        original = baseline.masked_fill(~data.masks, -torch.inf).softmax(-1)
        metrics: list[ContextMetrics] = []
        for context in _TARGET_CONTEXTS:
            selected = data.contexts == context
            count = int(selected.sum())
            target = (
                probabilities[selected]
                .gather(1, data.targets[selected, None])
                .squeeze(1)
            )
            metrics.append(
                ContextMetrics(
                    context.name.lower(),
                    ActionCount(count),
                    Probability(
                        float(
                            (
                                probabilities[selected].argmax(-1)
                                == data.targets[selected]
                            )
                            .float()
                            .mean()
                        )
                    )
                    if count
                    else None,
                    Probability(float(target.mean())) if count else None,
                    Probability(float(target.min())) if count else None,
                    Probability(float(probabilities[selected, _X].max()))
                    if count
                    else None,
                    Probability(float(probabilities[selected, _A].max()))
                    if count
                    else None,
                )
            )
        preservation: list[PreservationMetrics] = []
        for context in (MenuContext.NON_MENU, MenuContext.OTHER_MENU):
            selected = data.contexts == context
            count = int(selected.sum())
            preservation.append(
                PreservationMetrics(
                    context.name.lower(),
                    ActionCount(count),
                    torch.equal(logits[selected], baseline[selected])
                    if count
                    else None,
                    torch.equal(probabilities[selected], original[selected])
                    if count
                    else None,
                    torch.equal(
                        probabilities[selected].argmax(-1),
                        original[selected].argmax(-1),
                    )
                    if count
                    else None,
                )
            )
    return SplitMetrics(
        tuple(metrics),
        ActionCount(int((data.contexts == MenuContext.UNKNOWN).sum())),
        tuple(preservation),
    )


def worst_margin_loss(
    logits: Tensor, masks: Tensor, targets: Tensor, contexts: Tensor
) -> Tensor:
    """Mean context-wise worst example sum of squared legal-margin violations."""
    if (
        logits.ndim != 2
        or logits.shape[1] != ACTION_COUNT
        or masks.shape != logits.shape
        or masks.dtype != torch.bool
        or targets.shape != (len(logits),)
        or contexts.shape != targets.shape
    ):
        raise ValueError("worst-margin requires matching catalog logits/masks/labels")
    if (
        targets.dtype != torch.int64
        or contexts.dtype != torch.int64
        or not torch.isfinite(logits).all()
    ):
        raise ValueError("worst-margin requires integer labels and finite logits")
    recognized = torch.zeros_like(contexts, dtype=torch.bool)
    losses: list[Tensor] = []
    margins = objective_margins(ResidualObjective.WORST_MARGIN)
    for context in _TARGET_CONTEXTS:
        selected = contexts == context
        recognized |= selected
        if not selected.any():
            raise ValueError("worst-margin requires all three target contexts")
        expected_target = _A if context is MenuContext.APPLICABLE else _CANCEL
        if not (targets[selected] == expected_target).all():
            raise ValueError("unexpected target for worst-margin context")
        allowed = torch.zeros(ACTION_COUNT, dtype=torch.bool, device=masks.device)
        allowed[[_CANCEL, _X]] = True
        if context is not MenuContext.MISSING:
            allowed[_A] = True
        if (masks[selected] & ~allowed).any():
            raise ValueError("unexpected legal competitor for worst-margin context")
        if not masks[selected, expected_target].all():
            raise ValueError("worst-margin target must be syntactically legal")
        example_loss = torch.zeros(
            int(selected.sum()), dtype=logits.dtype, device=logits.device
        )
        for margin in margins:
            if margin.context != context.name.lower():
                continue
            gap = (
                logits[selected, expected_target] - logits[selected, margin.competitor]
            )
            violation = torch.relu(margin.required_logit_gap - gap).square()
            example_loss = example_loss + violation.masked_fill(
                ~masks[selected, margin.competitor], 0
            )
        losses.append(example_loss.max())
    if not recognized.all():
        raise ValueError("unknown target context passed to worst-margin objective")
    return torch.stack(losses).mean()


def residual_step(
    model: SemanticActorCritic,
    data: ResidualData,
    optimizer: torch.optim.Optimizer,
    *,
    objective: ResidualObjective = ResidualObjective.CE,
) -> None:
    selected = data.contexts <= MenuContext.MISSING
    if not all((data.contexts == context).any() for context in _TARGET_CONTEXTS):
        raise ValueError("balanced residual fit requires all three training contexts")
    if not data.masks[selected].gather(1, data.targets[selected, None]).all():
        raise ValueError("residual targets must be syntactically legal")
    optimizer.zero_grad(set_to_none=True)
    logits = model(data.features[selected])[0]
    if objective is ResidualObjective.WORST_MARGIN:
        loss = worst_margin_loss(
            logits,
            data.masks[selected],
            data.targets[selected],
            data.contexts[selected],
        )
    elif objective is ResidualObjective.CE:
        losses = torch.nn.functional.cross_entropy(
            logits.masked_fill(~data.masks[selected], -torch.inf),
            data.targets[selected],
            reduction="none",
        )
        loss = torch.stack(
            [
                losses[data.contexts[selected] == context].mean()
                for context in _TARGET_CONTEXTS
            ]
        ).mean()
    else:
        raise ValueError("unsupported residual objective")
    if not torch.isfinite(loss):
        raise ValueError("nonfinite residual loss")
    loss.backward()
    optimizer.step()


def run_residual_probe(
    checkpoint: Path,
    trajectories: Path,
    output: Path,
    *,
    objective: ResidualObjective = ResidualObjective.CE,
) -> Path:
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    paths = tuple(sorted(trajectories.rglob("trajectory.jsonl")))
    validate_training_paths(paths)
    training, validation = (
        load_residual_data(paths[:36]),
        load_residual_data(paths[36:]),
    )
    if tuple(
        episode.seed for episode in training.episodes + validation.episodes
    ) != tuple(range(3049, 3097)):
        raise ValueError(
            "fixed ac2 protocol requires exactly seeds 3049..3096 in lexical order"
        )
    source = LearnedPolicy(checkpoint, device="cpu")
    config = source.model.config
    if (
        source.checkpoint_action_count != ACTION_COUNT
        or config.feature_spec_version != FEATURE_SPEC_VERSION
        or config.action_history_length
        or config.ability_residual_version
    ):
        raise ValueError(
            "source must be current-feature/current-catalog stateless base "
            "without residual"
        )
    baseline = {
        name: tensor.detach().clone()
        for name, tensor in source.model.state_dict().items()
    }
    with torch.inference_mode():
        train_logits, validation_logits = (
            source.model(training.features)[0],
            source.model(validation.features)[0],
        )
    model = enable_ability_residual(source.model)
    if not outputs_exact(source.model, model, training) or not outputs_exact(
        source.model, model, validation
    ):
        raise ValueError("zero-enabled residual changed source outputs")
    training_coverage, validation_coverage = (
        selected_coverage(training),
        selected_coverage(validation),
    )
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in _OWNED)
    trainable = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    if sum(parameter.numel() for parameter in trainable) != 18:
        raise ValueError("v1 residual must own exactly 18 parameters")
    optimizer = torch.optim.Adam(trainable, lr=_LEARNING_RATE)
    code = tuple(
        _source(Path(__file__).with_name(name))
        for name in (
            "ability_residual_probe.py",
            "ability_probe.py",
            "ability_calibration.py",
            "learned.py",
            "features.py",
            "training.py",
            "replay.py",
            "observation.py",
            "env.py",
            "actions.py",
            "terrain.py",
        )
    )
    source_evidence = _source(checkpoint)
    snapshots: list[ResidualSnapshot] = []
    checkpoints: list[SourceEvidence] = []
    for step in range(257):
        if step:
            residual_step(model, training, optimizer, objective=objective)
        if not base_tensors_exact(model, baseline):
            raise ValueError(f"base tensor ownership violated at step {step}")
        if step not in (0, *_STEPS):
            continue
        snapshot = ResidualSnapshot(
            OptimizerStep(step),
            measure_residual(model, training, train_logits),
            measure_residual(model, validation, validation_logits),
            True,
            Seconds(time.monotonic() - started),
        )
        if any(
            group.logits_exact is False or group.probabilities_exact is False
            for split in (snapshot.training, snapshot.validation)
            for group in split.preservation
        ):
            raise ValueError(f"non-ability preservation violated at step {step}")
        snapshots.append(snapshot)
        metadata = ResidualMetadata(
            f"ability-residual-fixed-{objective.value}",
            0,
            step,
            int((training.contexts <= MenuContext.MISSING).sum()),
            36,
            _LEARNING_RATE,
            0.0,
            0.0,
            1.0,
            sum(item.accuracy or 0.0 for item in snapshot.validation.contexts) / 3,
            source_evidence,
            training.episodes,
            validation.episodes,
            code,
            training_coverage,
            validation_coverage,
            True,
            snapshot,
            tuple(sorted(_OWNED)),
            _STEPS,
            objective.value,
            objective_margins(objective),
        )
        path = output / (f"step-{step:04d}.pt" if step else "start.pt")
        save_checkpoint(
            path,
            model=model,
            policy_id=f"ability-residual-step-{step}",
            training_metadata=metadata,
        )
        checkpoints.append(_source(path))
        if not step:
            restored = LearnedPolicy(path, device="cpu").model
            if not outputs_exact(source.model, restored, training) or not outputs_exact(
                source.model, restored, validation
            ):
                raise ValueError(
                    "restored zero-enabled checkpoint changed source outputs"
                )
    report = ResidualReport(
        source_evidence,
        checkpoints[0],
        training.episodes,
        validation.episodes,
        code,
        training_coverage,
        validation_coverage,
        True,
        tuple(snapshots),
        tuple(checkpoints[1:]),
        _STEPS,
        _LEARNING_RATE,
        18,
        f"Fixed CPU Adam lr0.1, 256 full-batch steps, objective {objective.value}: "
        "equal context weight, mean example CE or worst example summed squared "
        "positive legal-competitor margin violations, "
        "for applicable, present-inapplicable, missing-Berserk contexts. "
        "Unknown excluded. Seeds3049..3084 train,3085..3096 validation; "
        "preaction frames only. Base tensors audited every step; "
        "non-ability preservation measured on all train+validation states "
        "at declared snapshots. No checkpoint selection, continuation, "
        "live rollout, or promotion is authorized by completion.",
        objective,
        objective_margins(objective),
    )
    path = output / "report.json"
    with path.open("x") as stream:
        stream.write(json.dumps(asdict(report), indent=2) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--objective",
        choices=list(ResidualObjective),
        type=ResidualObjective,
        default=ResidualObjective.CE,
    )
    args = parser.parse_args()
    print(
        run_residual_probe(
            args.checkpoint, args.trajectories, args.output, objective=args.objective
        )
    )


if __name__ == "__main__":
    main()
