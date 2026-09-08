"""Disposable, content-addressed environment-only imitation replay tensors."""

from __future__ import annotations

import hashlib
import json
import tempfile
import zlib
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import NewType
from zipfile import BadZipFile

import numpy as np

from dcss_rl.env import ACTION_COUNT
from dcss_rl.features import FEATURE_COUNT, FEATURE_SPEC_VERSION, FeatureVector
from dcss_rl.policy import ScriptedMibePolicy
from dcss_rl.training import (
    ActionMaskVector,
    ActionVector,
    load_imitation_replay_episode,
)

ReplayCacheKey = NewType("ReplayCacheKey", str)
# Explicit contract plus source digests: unversioned semantic changes invalidate too.
_CONTRACT = "environment-features/legal-mask/scripted-teacher-v1"
_DEPENDENCIES = (
    "actions.py",
    "env.py",
    "features.py",
    "observation.py",
    "policy.py",
    "schema.py",
    "training.py",
    "trajectory.py",
    "terrain.py",
    "replay_cache.py",
)


@dataclass(frozen=True, slots=True)
class PreparedImitationReplay:
    features: FeatureVector
    masks: ActionMaskVector
    teacher_actions: ActionVector
    cache_hit: bool


def imitation_replay_cache_key(
    trajectories: tuple[Path, ...], *, teacher: ScriptedMibePolicy
) -> ReplayCacheKey:
    digest = hashlib.sha256()
    contract = (
        _CONTRACT,
        int(FEATURE_SPEC_VERSION),
        int(FEATURE_COUNT),
        int(ACTION_COUNT),
        teacher.policy_id,
        len(trajectories),
        np.__version__,
    )
    digest.update(json.dumps(contract).encode())
    for name in _DEPENDENCIES:
        digest.update(
            hashlib.sha256(Path(__file__).with_name(name).read_bytes()).digest()
        )
    # Length-delimited content digests preserve order and multiplicity without paths.
    for path in trajectories:
        with path.open("rb") as source:
            digest.update(hashlib.file_digest(source, "sha256").digest())
    return ReplayCacheKey(digest.hexdigest())


def prepare_imitation_replay(
    trajectories: tuple[Path, ...],
    *,
    teacher: ScriptedMibePolicy,
    cache_directory: Path | None = None,
) -> PreparedImitationReplay:
    """Build/cache only feature, legality and teacher tensors; never model history.

    Input trajectories must be finalized before preparation. Only the exact stateless
    built-in teacher is cacheable: arbitrary subclasses may carry unkeyed state.
    """
    if not trajectories:
        return PreparedImitationReplay(
            np.empty((0, FEATURE_COUNT), dtype=np.float32),
            np.empty((0, ACTION_COUNT), dtype=np.bool_),
            np.empty(0, dtype=np.int64),
            False,
        )
    cache_path: Path | None = None
    key: ReplayCacheKey | None = None
    if cache_directory is not None and type(teacher) is ScriptedMibePolicy:
        key = imitation_replay_cache_key(trajectories, teacher=teacher)
        cache_path = cache_directory / f"{key}.npz"
        if (cached := _read_cache(cache_path, key)) is not None:
            return cached
    episodes = tuple(
        load_imitation_replay_episode(path, teacher=teacher) for path in trajectories
    )
    replay = PreparedImitationReplay(
        np.concatenate([np.stack(episode.features) for episode in episodes]),
        np.concatenate([np.stack(episode.masks) for episode in episodes]),
        np.concatenate(
            [np.asarray(episode.actions, dtype=np.int64) for episode in episodes]
        ),
        False,
    )
    if cache_path is not None and key is not None:
        # Do not install under a stale key if a producer changed a trajectory.
        if imitation_replay_cache_key(trajectories, teacher=teacher) == key:
            _write_cache(cache_path, key, replay)
    return replay


def _read_cache(path: Path, key: ReplayCacheKey) -> PreparedImitationReplay | None:
    try:
        loaded = np.load(path, allow_pickle=False)
        if not isinstance(loaded, np.lib.npyio.NpzFile):
            return None
        with loaded as cached:
            if cached["key"].shape != () or cached["key"].item() != key:
                return None
            features = cached["features"]
            masks = cached["masks"]
            actions = cached["teacher_actions"]
        if features.ndim != 2 or features.dtype != np.float32:
            return None
        count = features.shape[0]
        if (
            count == 0
            or features.shape[1] != FEATURE_COUNT
            or masks.dtype != np.bool_
            or masks.shape != (count, ACTION_COUNT)
            or actions.dtype != np.int64
            or actions.shape != (count,)
            or np.any(actions < 0)
            or np.any(actions >= ACTION_COUNT)
            or not np.isfinite(features).all()
            or not masks[np.arange(count), actions].all()
        ):
            return None
        return PreparedImitationReplay(features, masks, actions, True)
    except (OSError, ValueError, KeyError, EOFError, BadZipFile, zlib.error):
        return None


def _write_cache(
    path: Path, key: ReplayCacheKey, replay: PreparedImitationReplay
) -> None:
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            np.savez(
                temporary,
                key=np.asarray(key),
                features=replay.features,
                masks=replay.masks,
                teacher_actions=replay.teacher_actions,
            )
        temporary_path.replace(path)
    except OSError:
        # Cache availability must not determine whether training can start.
        pass
    finally:
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)
