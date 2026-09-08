"""Compare complete first-update worker recordings without conflating provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NewType, cast

import msgspec
import torch

from dcss_rl.checkpoint_audit import compare_checkpoints, load_checkpoint_contents
from dcss_rl.features import encode_observation
from dcss_rl.returns import ReturnBoundaryMode
from dcss_rl.schema import (
    ActionData,
    JsonObject,
    ObservationData,
    ObservationDeltaData,
)
from dcss_rl.training import _training_action_mask
from dcss_rl.trajectory import apply_observation_delta
from dcss_rl.units import (
    ActionCount,
    ActionIndex,
    EpisodeIndex,
    FeatureSpecVersion,
    Keycode,
    StepLimit,
    WorkerCount,
    WorkerIndex,
)

ContentDigest = NewType("ContentDigest", str)
RolloutStep = NewType("RolloutStep", int)
_EPISODE = re.compile(r"episode-(\d+)-attempt-(\d+)")
_WORKER = re.compile(r"worker-(\d+)")
_DEFAULT_WORKERS = WorkerCount(48)
_DEFAULT_STEPS = StepLimit(128)


@dataclass(frozen=True)
class _Initial:
    observation: msgspec.Raw
    raw_messages: msgspec.Raw
    setup_keycodes: tuple[Keycode, ...]


@dataclass(frozen=True)
class _Header:
    type: str
    schema_version: int
    metadata: msgspec.Raw
    initial: _Initial


@dataclass(frozen=True)
class _Provenance:
    update: int
    checkpoint_sha256: str


@dataclass(frozen=True)
class _Sampling:
    probabilities: tuple[float, ...]
    rng_state_sha256: str


@dataclass(frozen=True)
class _Transition:
    type: str
    step: int
    action_index: ActionIndex
    action: ActionData
    emitted_keycodes: tuple[Keycode, ...]
    raw_messages: msgspec.Raw
    terminated: bool
    truncated: bool
    reward: float
    collection_provenance: _Provenance
    observation_delta: msgspec.Raw
    outcome: str | None = None
    sampling_evidence: _Sampling | None = None


@dataclass(frozen=True)
class _Reset:
    episode: EpisodeIndex
    semantic: ContentDigest
    mask: ContentDigest
    raw: ContentDigest
    keys: tuple[Keycode, ...]
    metadata: tuple[tuple[str, ContentDigest], ...]
    features: ContentDigest


@dataclass(frozen=True)
class _Sample:
    episode: EpisodeIndex
    step: RolloutStep
    before: ContentDigest
    after: ContentDigest
    mask_before: ContentDigest
    mask_after: ContentDigest
    action: ContentDigest
    keys: tuple[Keycode, ...]
    terminal: tuple[bool, bool, str | None]
    reward: float
    raw: ContentDigest
    probabilities: tuple[float, ...] | None
    rng_state_sha256: str | None
    features_before: ContentDigest
    features_after: ContentDigest


@dataclass(frozen=True)
class _Trace:
    worker: WorkerIndex
    resets: tuple[_Reset, ...]
    samples: tuple[_Sample, ...]
    trailing_reset: _Reset | None


@dataclass(frozen=True)
class Difference:
    field: str
    count: ActionCount
    first_index: RolloutStep | None


@dataclass(frozen=True)
class WorkerComparison:
    worker: WorkerIndex
    transitions: ActionCount
    differences: tuple[Difference, ...]
    reset_metadata_differences: tuple[str, ...]
    bootstrap_metadata_differences: tuple[str, ...]
    bootstrap_present: tuple[bool, bool]
    matched: bool
    raw_identical: bool
    sampling_identical: bool | None
    policy_inputs_identical: bool
    policy_rollout_matched: bool


@dataclass(frozen=True)
class RolloutComparison:
    reference: str
    candidate: str
    workers: WorkerCount
    steps_per_worker: StepLimit
    reference_checkpoint_sha256: str
    candidate_checkpoint_sha256: str
    initial_model_equal: bool
    initial_model_incompatibilities: tuple[str, ...]
    initial_model_changed_tensors: tuple[str, ...]
    initial_model_changed_policy_rows: tuple[ActionIndex, ...]
    cases: tuple[WorkerComparison, ...]
    matched: bool
    raw_identical: bool
    sampling_identical: bool | None
    reference_feature_version: FeatureSpecVersion
    candidate_feature_version: FeatureSpecVersion
    policy_inputs_identical: bool
    policy_rollout_matched: bool
    raw_comparison_definition: str


def _digest(value: object) -> ContentDigest:
    # Canonical JSON key order only; no payload fields/paths/timestamps are removed.
    return ContentDigest(
        hashlib.sha256(
            json.dumps(
                value, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
    )


def _mask(observation: ObservationData) -> ContentDigest:
    return ContentDigest(
        hashlib.sha256(_training_action_mask(observation).tobytes()).hexdigest()
    )


def _json_object(raw: msgspec.Raw) -> JsonObject:
    value: object = json.loads(bytes(raw))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return cast(JsonObject, value)


def _raw_digest(raw: msgspec.Raw) -> ContentDigest:
    return _digest(json.loads(bytes(raw)))


def _policy_features(
    observation: ObservationData, version: FeatureSpecVersion
) -> ContentDigest:
    features = encode_observation(observation, spec_version=version)
    return ContentDigest(hashlib.sha256(features.tobytes()).hexdigest())


def _reset(
    header: _Header, episode: EpisodeIndex, version: FeatureSpecVersion
) -> _Reset:
    if header.type != "episode" or header.schema_version != 2:
        raise ValueError("invalid trajectory episode header")
    return _Reset(
        episode,
        _raw_digest(header.initial.observation),
        _mask(cast(ObservationData, _json_object(header.initial.observation))),
        _raw_digest(header.initial.raw_messages),
        header.initial.setup_keycodes,
        tuple(
            (key, _digest(value))
            for key, value in sorted(_json_object(header.metadata).items())
        ),
        _policy_features(
            cast(ObservationData, _json_object(header.initial.observation)), version
        ),
    )


def _episodes(worker: Path) -> tuple[Path, ...]:
    grouped: dict[int, list[tuple[int, Path]]] = {}
    for path in worker.iterdir():
        if path.is_dir() and (match := _EPISODE.fullmatch(path.name)):
            grouped.setdefault(int(match[1]), []).append((int(match[2]), path))
    if not grouped or sorted(grouped) != list(range(max(grouped) + 1)):
        raise ValueError(f"missing/noncontiguous episodes: {worker}")
    result: list[Path] = []
    for number in sorted(grouped):
        attempts = grouped[number]
        if len(attempts) != 1 or attempts[0][0] != 0:
            raise ValueError(
                f"multiple/retried attempts prevent exact comparison: {worker}/{number}"
            )
        result.append(attempts[0][1] / "trajectory.jsonl")
    return tuple(result)


def _trace(
    worker: Path,
    index: WorkerIndex,
    steps: StepLimit,
    checkpoint_hash: str,
    continuing: bool,
    version: FeatureSpecVersion,
) -> _Trace:
    samples: list[_Sample] = []
    resets: list[_Reset] = []
    trailing: _Reset | None = None
    paths = _episodes(worker)
    finished = False
    for number, path in enumerate(paths):
        if finished:
            break
        if len(samples) == steps and not continuing:
            break
        with path.open() as stream:
            header = msgspec.json.decode(next(stream), type=_Header)
            reset = _reset(header, EpisodeIndex(number), version)
            current = cast(ObservationData, _json_object(header.initial.observation))
            if len(samples) == steps:
                if not any(samples[-1].terminal[:2]):
                    raise ValueError(
                        "unexpected reset after nonterminal update boundary"
                    )
                trailing = reset
                if line := next(stream, None):
                    following = msgspec.json.decode(line, type=_Transition)
                    if following.collection_provenance.update <= 1:
                        raise ValueError(
                            "bootstrap reset has excess first-update actions"
                        )
                break
            resets.append(reset)
            episode_steps = 0
            terminal = False
            for line in stream:
                record = msgspec.json.decode(line, type=_Transition)
                if record.collection_provenance.update > 1:
                    if len(samples) != steps:
                        raise ValueError(
                            "second update starts before required coverage"
                        )
                    finished = True
                    break
                if record.collection_provenance.update != 1:
                    raise ValueError("first-update provenance must be update 1")
                if record.collection_provenance.checkpoint_sha256 != checkpoint_hash:
                    raise ValueError(
                        "transition provenance disagrees with its collector checkpoint"
                    )
                if (
                    record.type != "transition"
                    or record.step != episode_steps
                    or terminal
                ):
                    raise ValueError(
                        "invalid transition ordering or transition after terminal"
                    )
                if len(samples) >= steps:
                    raise ValueError("first update exceeds required transition count")
                after = apply_observation_delta(
                    current,
                    cast(ObservationDeltaData, _json_object(record.observation_delta)),
                )
                samples.append(
                    _Sample(
                        EpisodeIndex(number),
                        RolloutStep(episode_steps),
                        _digest(current),
                        _digest(after),
                        _mask(current),
                        _mask(after),
                        _digest((record.action_index, record.action)),
                        record.emitted_keycodes,
                        (record.terminated, record.truncated, record.outcome),
                        record.reward,
                        _raw_digest(record.raw_messages),
                        record.sampling_evidence.probabilities
                        if record.sampling_evidence
                        else None,
                        record.sampling_evidence.rng_state_sha256
                        if record.sampling_evidence
                        else None,
                        _policy_features(current, version),
                        _policy_features(after, version),
                    )
                )
                current = after
                episode_steps += 1
                terminal = record.terminated or record.truncated
            if len(samples) < steps and not terminal:
                raise ValueError(
                    "incomplete nonterminal episode before required coverage"
                )
    if len(samples) != steps:
        raise ValueError(
            f"worker {index}: expected {steps} transitions, got {len(samples)}"
        )
    if continuing and any(samples[-1].terminal[:2]) and trailing is None:
        raise ValueError("continuing terminal boundary lacks trailing bootstrap reset")
    return _Trace(index, tuple(resets), tuple(samples), trailing)


def _metadata_changes(left: _Reset, right: _Reset) -> tuple[str, ...]:
    before, after = dict(left.metadata), dict(right.metadata)
    return tuple(
        key
        for key in sorted(before.keys() | after.keys())
        if before.get(key) != after.get(key)
    )


def _compare(left: _Trace, right: _Trace) -> WorkerComparison:
    differences: list[Difference] = []
    for name in (
        "episode",
        "step",
        "before",
        "after",
        "mask_before",
        "mask_after",
        "action",
        "keys",
        "terminal",
        "reward",
        "raw",
        "probabilities",
        "rng_state_sha256",
        "features_before",
        "features_after",
    ):
        changed = [
            i
            for i, (old, new) in enumerate(
                zip(left.samples, right.samples, strict=True)
            )
            if getattr(old, name) != getattr(new, name)
        ]
        differences.append(
            Difference(
                name,
                ActionCount(len(changed)),
                RolloutStep(changed[0]) if changed else None,
            )
        )
    metadata: set[str] = set()
    for name in ("episode", "semantic", "mask", "keys", "raw", "features"):
        count = abs(len(left.resets) - len(right.resets))
        count += sum(
            getattr(old, name) != getattr(new, name)
            for old, new in zip(left.resets, right.resets, strict=False)
        )
        differences.append(Difference(f"reset_{name}", ActionCount(count), None))
    for old, new in zip(left.resets, right.resets, strict=False):
        metadata.update(_metadata_changes(old, new))
    bootstrap_metadata: tuple[str, ...] = ()
    if left.trailing_reset is not None and right.trailing_reset is not None:
        bootstrap_metadata = _metadata_changes(
            left.trailing_reset, right.trailing_reset
        )
        for name in ("episode", "semantic", "mask", "keys", "raw", "features"):
            count = int(
                getattr(left.trailing_reset, name)
                != getattr(right.trailing_reset, name)
            )
            differences.append(
                Difference(f"bootstrap_{name}", ActionCount(count), None)
            )
    else:
        differences.append(
            Difference(
                "bootstrap_presence",
                ActionCount(
                    int((left.trailing_reset is None) != (right.trailing_reset is None))
                ),
                None,
            )
        )
    sampling_identical = (
        False
        if any(
            item.count
            for item in differences
            if item.field in {"probabilities", "rng_state_sha256"}
        )
        else True
        if all(
            sample.probabilities is not None
            for sample in (*left.samples, *right.samples)
        )
        else None
    )
    policy_inputs_identical = not any(
        item.count
        for item in differences
        if item.field
        in {
            "features_before",
            "features_after",
            "mask_before",
            "mask_after",
            "reset_features",
            "reset_mask",
            "bootstrap_features",
            "bootstrap_mask",
            "bootstrap_presence",
        }
    )
    policy_rollout_matched = (
        sampling_identical is True
        and not any(
            item.count
            for item in differences
            if item.field
            not in {"before", "after", "reset_semantic", "bootstrap_semantic"}
            and not item.field.endswith("raw")
        )
        and not (
            {"seed", "character", "reward_spec"} & (metadata | set(bootstrap_metadata))
        )
    )
    return WorkerComparison(
        left.worker,
        ActionCount(len(left.samples)),
        tuple(differences),
        tuple(sorted(metadata)),
        bootstrap_metadata,
        (left.trailing_reset is not None, right.trailing_reset is not None),
        not any(
            item.count
            for item in differences
            if not item.field.endswith("raw")
            and item.field not in {"probabilities", "rng_state_sha256"}
        )
        and not (
            {"seed", "character", "reward_spec"} & (metadata | set(bootstrap_metadata))
        ),
        not any(item.count for item in differences if item.field.endswith("raw")),
        sampling_identical,
        policy_inputs_identical,
        policy_rollout_matched,
    )


def _continuing(path: Path, workers: WorkerCount, steps: StepLimit) -> bool:
    payload: object = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(
        metadata := payload.get("training_metadata"), dict
    ):
        raise ValueError("collector checkpoint lacks training metadata")
    if (
        metadata.get("worker_count") != workers
        or metadata.get("rollout_steps") != steps
    ):
        raise ValueError(
            "requested worker/step coverage disagrees with collector metadata"
        )
    return (
        ReturnBoundaryMode(metadata.get("return_boundary", "episodic"))
        is ReturnBoundaryMode.CONTINUING_RESET
    )


def compare_training_rollouts(
    reference: Path,
    candidate: Path,
    *,
    workers: WorkerCount = _DEFAULT_WORKERS,
    steps: StepLimit = _DEFAULT_STEPS,
) -> RolloutComparison:
    if workers < 1 or steps < 1:
        raise ValueError("worker and step counts must be positive")
    checkpoints = tuple(
        root / "collector-checkpoints/update-0000.pt" for root in (reference, candidate)
    )
    old, new = (load_checkpoint_contents(path) for path in checkpoints)
    if old.config.action_history_length or new.config.action_history_length:
        raise ValueError(
            "policy-input comparison does not yet support action-history checkpoints"
        )
    model = compare_checkpoints(new, old)
    initial_equal = not (
        model.incompatibilities
        or model.changed_policy_rows
        or model.changed_non_policy_tensors
    )
    for root in (reference, candidate):
        actual = sorted(
            int(match[1])
            for path in root.iterdir()
            if path.is_dir() and (match := _WORKER.fullmatch(path.name))
        )
        if actual != list(range(workers)):
            raise ValueError(
                f"expected exactly workers 0 through {workers - 1}: {root}"
            )
    old_continuing, new_continuing = (
        _continuing(path, workers, steps) for path in checkpoints
    )
    cases = tuple(
        _compare(
            _trace(
                reference / f"worker-{index}",
                WorkerIndex(index),
                steps,
                old.sha256,
                old_continuing,
                old.config.feature_spec_version,
            ),
            _trace(
                candidate / f"worker-{index}",
                WorkerIndex(index),
                steps,
                new.sha256,
                new_continuing,
                new.config.feature_spec_version,
            ),
        )
        for index in range(workers)
    )
    return RolloutComparison(
        str(reference),
        str(candidate),
        workers,
        steps,
        old.sha256,
        new.sha256,
        initial_equal,
        model.incompatibilities,
        model.changed_non_policy_tensors,
        model.changed_policy_rows,
        cases,
        initial_equal and all(case.matched for case in cases),
        all(case.raw_identical for case in cases),
        False
        if any(case.sampling_identical is False for case in cases)
        else True
        if all(case.sampling_identical is True for case in cases)
        else None,
        old.config.feature_spec_version,
        new.config.feature_spec_version,
        all(case.policy_inputs_identical for case in cases),
        initial_equal and all(case.policy_rollout_matched for case in cases),
        "Decoded raw_messages compared with canonical JSON key ordering only; "
        "all stored values, fields, message order, controls and strings retained. "
        "Original Python batch partitions are not stored by the recorder. "
        "File serialization whitespace is not compared. "
        "Metadata differences are separate.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-identical", action="store_true")
    parser.add_argument("--require-raw-identical", action="store_true")
    parser.add_argument("--require-policy-identical", action="store_true")
    args = parser.parse_args()
    try:
        report = compare_training_rollouts(
            args.reference,
            args.candidate,
            workers=WorkerCount(args.workers),
            steps=StepLimit(args.steps),
        )
    except (ValueError, OSError, StopIteration, msgspec.DecodeError) as error:
        parser.error(str(error))
    result = json.dumps(asdict(report), indent=2) + "\n"
    if args.output is not None:
        with args.output.open("x") as stream:
            stream.write(result)
    print(result, end="")
    if (
        (args.require_identical and not report.matched)
        or (args.require_raw_identical and not report.raw_identical)
        or (args.require_policy_identical and not report.policy_rollout_matched)
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
