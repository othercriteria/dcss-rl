"""Read-only ability-choice confidence audit on seed-disjoint training replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor

from dcss_rl.ability_probe import validate_training_paths
from dcss_rl.actions import Action, ActionKind
from dcss_rl.env import ACTION_COUNT, action_to_index
from dcss_rl.features import encode_observation
from dcss_rl.learned import LearnedPolicy, ModelConfig, SemanticActorCritic
from dcss_rl.observation import MenuChoiceApplicability
from dcss_rl.replay import replay_frames
from dcss_rl.schema import JsonObject, ObservationData
from dcss_rl.training import _training_action_mask
from dcss_rl.units import (
    ActionCount,
    ActionIndex,
    GameSeed,
    Keycode,
    Probability,
    TrajectoryCount,
)

_VALIDATION_EPISODES = TrajectoryCount(2)
_RENOUNCE = action_to_index(Action.menu_select(Keycode(ord("X"))))
_CANCEL = action_to_index(Action(ActionKind.CANCEL))
_BERSERK = action_to_index(Action.menu_select(Keycode(ord("a"))))


@dataclass(frozen=True)
class SourceEvidence:
    path: str
    sha256: str


@dataclass(frozen=True)
class EpisodeEvidence:
    source: SourceEvidence
    seed: GameSeed


@dataclass(frozen=True)
class ProbabilitySummary:
    mean: Probability
    minimum: Probability
    p05: Probability
    p25: Probability
    median: Probability
    p75: Probability
    p95: Probability
    maximum: Probability


@dataclass(frozen=True)
class ContextCalibration:
    context: MenuChoiceApplicability
    samples: ActionCount
    target_probability: ProbabilitySummary | None
    renounce_x_probability: ProbabilitySummary | None
    other_wrong_legal_mass: ProbabilitySummary | None
    argmax_accuracy: Probability | None
    berserk_a_probability: ProbabilitySummary | None


@dataclass(frozen=True)
class PreservationGroup:
    observations: ActionCount
    masked_probabilities_exact: bool | None
    changed_argmax_actions: ActionCount
    maximum_probability_change: Probability | None


@dataclass(frozen=True)
class ProbabilityPreservation:
    reference: SourceEvidence
    non_menu: PreservationGroup
    other_menu: PreservationGroup


@dataclass(frozen=True)
class CalibrationSplit:
    episodes: tuple[EpisodeEvidence, ...]
    observations: ActionCount
    non_ability_observations: ActionCount
    missing_berserk: ActionCount
    unknown_applicability: ActionCount
    contexts: tuple[ContextCalibration, ...]


@dataclass(frozen=True)
class CheckpointCalibration:
    checkpoint: SourceEvidence
    config: ModelConfig
    training: CalibrationSplit
    validation: CalibrationSplit
    preservation: ProbabilityPreservation | None


@dataclass(frozen=True)
class AbilityCalibration:
    checkpoints: tuple[CheckpointCalibration, ...]
    code: tuple[SourceEvidence, ...]
    split_definition: str
    probability_definition: str


@dataclass(frozen=True)
class _MenuSample:
    observation: ObservationData
    context: MenuChoiceApplicability
    target: ActionIndex


@dataclass(frozen=True)
class _SplitData:
    episodes: tuple[EpisodeEvidence, ...]
    observations: ActionCount
    non_ability_observations: ActionCount
    missing_berserk: ActionCount
    unknown_applicability: ActionCount
    samples: tuple[_MenuSample, ...]
    non_ability_states: tuple[ObservationData, ...]


def _source(path: Path) -> SourceEvidence:
    return SourceEvidence(str(path), hashlib.sha256(path.read_bytes()).hexdigest())


def split_training_paths(
    root: Path,
    validation_episodes: TrajectoryCount = _VALIDATION_EPISODES,
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    paths = tuple(sorted(root.rglob("trajectory.jsonl")))
    if not 1 <= validation_episodes < len(paths):
        raise ValueError(
            "split requires at least one training and one validation episode"
        )
    validate_training_paths(paths)
    return paths[:-validation_episodes], paths[-validation_episodes:]


def _load(paths: tuple[Path, ...]) -> _SplitData:
    episodes: list[EpisodeEvidence] = []
    samples: list[_MenuSample] = []
    non_ability_states: list[ObservationData] = []
    observations = non_ability = missing = unknown = 0
    for path in paths:
        with path.open() as stream:
            header = cast(JsonObject, json.loads(stream.readline()))
        metadata = header["metadata"]
        if (
            not isinstance(metadata, dict)
            or type(seed := metadata.get("seed")) is not int
        ):
            raise ValueError("trajectory must have a seeded metadata header")
        episodes.append(EpisodeEvidence(_source(path), GameSeed(seed)))
        for before, _after in pairwise(replay_frames(path)):
            observations += 1
            observation = before.observation
            menu = observation["menu"]
            if menu is None or menu["type"] != "ability":
                non_ability += 1
                non_ability_states.append(observation)
                continue
            berserk = next(
                (
                    choice
                    for choice in menu["choices"]
                    if "berserk" in choice["text"].casefold()
                ),
                None,
            )
            if berserk is None:
                missing += 1
                continue
            context = MenuChoiceApplicability(berserk.get("applicability", "unknown"))
            if context is MenuChoiceApplicability.UNKNOWN:
                unknown += 1
                continue
            target = (
                action_to_index(Action.menu_select(Keycode(berserk["keycode"])))
                if context is MenuChoiceApplicability.APPLICABLE
                else _CANCEL
            )
            if target == _RENOUNCE:
                raise ValueError("Berserk target unexpectedly aliases Renounce X")
            samples.append(_MenuSample(observation, context, target))
    return _SplitData(
        tuple(episodes),
        ActionCount(observations),
        ActionCount(non_ability),
        ActionCount(missing),
        ActionCount(unknown),
        tuple(samples),
        tuple(non_ability_states),
    )


def probability_summary(values: Tensor) -> ProbabilitySummary | None:
    if not values.numel():
        return None
    quantiles = torch.quantile(
        values.double(),
        torch.tensor([0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0], dtype=torch.float64),
    )
    return ProbabilitySummary(
        Probability(float(values.double().mean())),
        *(Probability(float(value)) for value in quantiles),
    )


def context_calibration(
    logits: Tensor,
    masks: Tensor,
    targets: Tensor,
    context: MenuChoiceApplicability,
) -> ContextCalibration:
    if (
        logits.ndim != 2
        or masks.shape != logits.shape
        or targets.shape != (len(logits),)
    ):
        raise ValueError("logits, masks, and targets must have matching sample rows")
    if masks.dtype != torch.bool or logits.shape[1] <= _RENOUNCE:
        raise ValueError("calibration needs boolean catalog masks including menu X")
    if (
        targets.dtype != torch.int64
        or torch.any(targets < 0)
        or torch.any(targets >= logits.shape[1])
    ):
        raise ValueError("calibration targets must be valid integer catalog indices")
    if torch.any(targets == _RENOUNCE) or not masks.gather(1, targets[:, None]).all():
        raise ValueError(
            "calibration targets must be legal and distinct from Renounce X"
        )
    if not torch.isfinite(logits).all():
        raise ValueError("calibration requires finite model logits")
    probabilities = torch.softmax(logits.masked_fill(~masks, -torch.inf), dim=-1)
    target_probability = probabilities.gather(1, targets[:, None]).squeeze(1)
    renounce_probability = probabilities[:, _RENOUNCE]
    other = masks.clone()
    other.scatter_(1, targets[:, None], False)
    other[:, _RENOUNCE] = False
    other_mass = probabilities.masked_fill(~other, 0).sum(-1)
    accuracy = (
        Probability(float((probabilities.argmax(-1) == targets).float().mean()))
        if len(logits)
        else None
    )
    return ContextCalibration(
        context,
        ActionCount(len(logits)),
        probability_summary(target_probability),
        probability_summary(renounce_probability),
        probability_summary(other_mass),
        accuracy,
        probability_summary(probabilities[:, _BERSERK]),
    )


def _probabilities(
    model: SemanticActorCritic, states: tuple[ObservationData, ...]
) -> Tensor:
    features = torch.from_numpy(
        np.stack(
            [
                encode_observation(
                    state, spec_version=model.config.feature_spec_version
                )
                for state in states
            ]
        )
    )
    masks = torch.from_numpy(
        np.stack([_training_action_mask(state) for state in states])
    )
    with torch.inference_mode():
        logits = model(features)[0]
        return torch.softmax(logits.masked_fill(~masks, -torch.inf), dim=-1)


def _preservation_group(
    reference: SemanticActorCritic,
    candidate: SemanticActorCritic,
    states: tuple[ObservationData, ...],
) -> PreservationGroup:
    if not states:
        return PreservationGroup(ActionCount(0), None, ActionCount(0), None)
    before, after = _probabilities(reference, states), _probabilities(candidate, states)
    return PreservationGroup(
        ActionCount(len(states)),
        torch.equal(before, after),
        ActionCount(int((before.argmax(-1) != after.argmax(-1)).sum())),
        Probability(float((before - after).abs().max())),
    )


def _measure(model: SemanticActorCritic, data: _SplitData) -> CalibrationSplit:
    if model.config.action_history_length:
        raise ValueError(
            "ability calibration does not yet support action-history checkpoints"
        )
    if model.config.action_count != ACTION_COUNT:
        raise ValueError(
            "checkpoint action catalog is incompatible; no migration is performed"
        )
    contexts: list[ContextCalibration] = []
    for context in (
        MenuChoiceApplicability.APPLICABLE,
        MenuChoiceApplicability.INAPPLICABLE,
    ):
        selected = tuple(sample for sample in data.samples if sample.context is context)
        if selected:
            features = torch.from_numpy(
                np.stack(
                    [
                        encode_observation(
                            sample.observation,
                            spec_version=model.config.feature_spec_version,
                        )
                        for sample in selected
                    ]
                )
            )
            masks = torch.from_numpy(
                np.stack(
                    [_training_action_mask(sample.observation) for sample in selected]
                )
            )
            targets = torch.tensor(
                [sample.target for sample in selected], dtype=torch.int64
            )
            with torch.inference_mode():
                logits = model(features)[0]
        else:
            logits = torch.empty((0, ACTION_COUNT))
            masks = torch.empty((0, ACTION_COUNT), dtype=torch.bool)
            targets = torch.empty(0, dtype=torch.int64)
        contexts.append(context_calibration(logits, masks, targets, context))
    return CalibrationSplit(
        data.episodes,
        data.observations,
        data.non_ability_observations,
        data.missing_berserk,
        data.unknown_applicability,
        tuple(contexts),
    )


def audit_ability_calibration(
    checkpoints: tuple[Path, ...],
    trajectories: Path,
    *,
    validation_episodes: TrajectoryCount = _VALIDATION_EPISODES,
) -> AbilityCalibration:
    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    training_paths, validation_paths = split_training_paths(
        trajectories, validation_episodes
    )
    training, validation = _load(training_paths), _load(validation_paths)
    reports: list[CheckpointCalibration] = []
    reference = LearnedPolicy(checkpoints[0], device="cpu")
    non_ability_states = training.non_ability_states + validation.non_ability_states
    for path in checkpoints:
        policy = LearnedPolicy(path, device="cpu")
        if policy.checkpoint_action_count != ACTION_COUNT:
            raise ValueError(
                "checkpoint action catalog is incompatible; no migration is performed"
            )
        reports.append(
            CheckpointCalibration(
                SourceEvidence(str(path), str(policy.checkpoint_id)),
                policy.model.config,
                _measure(policy.model, training),
                _measure(policy.model, validation),
                ProbabilityPreservation(
                    SourceEvidence(str(checkpoints[0]), str(reference.checkpoint_id)),
                    _preservation_group(
                        reference.model,
                        policy.model,
                        tuple(
                            state
                            for state in non_ability_states
                            if state["menu"] is None
                        ),
                    ),
                    _preservation_group(
                        reference.model,
                        policy.model,
                        tuple(
                            state
                            for state in non_ability_states
                            if state["menu"] is not None
                        ),
                    ),
                )
                if len(checkpoints) > 1
                else None,
            )
        )
    code = tuple(
        _source(Path(__file__).with_name(name))
        for name in (
            "ability_calibration.py",
            "ability_probe.py",
            "features.py",
            "terrain.py",
            "actions.py",
            "env.py",
            "observation.py",
            "learned.py",
            "replay.py",
            "trajectory.py",
            "training.py",
        )
    )
    return AbilityCalibration(
        tuple(reports),
        code,
        "Lexically sorted trajectory paths; last N episodes reserved for validation. "
        "Every seed must belong to current training suite and occur in one episode. "
        "Only pre-action observations counted; final unacted states excluded.",
        "Syntactically masked softmax at the checkpoint's unchanged feature version. "
        "Target is visible Berserk choice when applicable, cancel when inapplicable. "
        "Target + Renounce X + all other wrong legal mass partitions probability one. "
        "Berserk-a is explicit and overlaps target or other-wrong mass. "
        "Preservation covers training AND validation non-ability states. "
        "Quantiles use linear interpolation; absent contexts have null statistics. "
        "This audits confidence on training replay, not rollout survival or promotion.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--validation-episodes", type=int, default=_VALIDATION_EPISODES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        report = audit_ability_calibration(
            tuple(args.checkpoint),
            args.trajectories,
            validation_episodes=TrajectoryCount(args.validation_episodes),
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    result = json.dumps(asdict(report), indent=2) + "\n"
    if args.output is not None:
        with args.output.open("x") as stream:
            stream.write(result)
    print(result, end="")


if __name__ == "__main__":
    main()
