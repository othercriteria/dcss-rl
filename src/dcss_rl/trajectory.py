"""Replayable JSONL trajectories with policy and ECHO loss segmentation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, NewType, cast

from dcss_rl.env import DcssEnv, index_to_action
from dcss_rl.schema import (
    CellView,
    CharacterData,
    EnvironmentInfo,
    ObservationData,
    ObservationDeltaData,
    PlayerView,
    Position,
    RawMessageData,
    ResetOptions,
)
from dcss_rl.units import ActionIndex, CheckpointId, Probability, UpdateCount
from dcss_rl.webtiles import ObservationBatch
from dcss_rl.webtiles.cache import StaticCachePreparationTiming, StaticDataIdentity

SCHEMA_VERSION = 2
SamplingStateHash = NewType("SamplingStateHash", str)


@dataclass(frozen=True, slots=True)
class SamplingEvidence:
    """Exact choice probabilities and pre-choice generator-state fingerprint."""

    probabilities: tuple[Probability, ...]
    rng_state_sha256: SamplingStateHash


@dataclass(frozen=True, slots=True)
class CollectionProvenance:
    """Actual frozen collector identity, independent of episode boundaries."""

    update: UpdateCount
    checkpoint_sha256: CheckpointId


@dataclass(frozen=True, slots=True)
class LossSegment:
    """Text span and the objective, if any, trained on that span."""

    role: Literal["environment", "action"]
    text: str
    loss: Literal["policy", "environment"] | None


@dataclass(frozen=True, slots=True)
class EpisodeMetadata:
    episode_id: str
    created_at: str
    dcss_version: str
    dcss_commit: str | None
    code_revision: str | None
    seed: int | None
    character: CharacterData
    agent_id: str
    checkpoint_id: str | None
    reward_spec: str
    rc_sha256: str
    static_data_identity: StaticDataIdentity | None = None
    static_cache_preparation_timing: StaticCachePreparationTiming | None = None


def _revision(directory: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _raw_batches(batches: tuple[ObservationBatch, ...]) -> list[RawMessageData]:
    return [
        {"control": message.control, "payload": message.payload}
        for batch in batches
        for message in batch.messages
    ]


def observation_delta(
    previous: ObservationData, current: ObservationData
) -> ObservationDeltaData:
    """Return the minimal deterministic patch from one semantic snapshot to another."""
    changed_player = {
        key: deepcopy(value)
        for key, value in current["player"].items()
        if previous["player"].get(key) != value
    }
    removed_player_fields = [
        key for key in previous["player"] if key not in current["player"]
    ]
    previous_cells = {(cell["x"], cell["y"]): cell for cell in previous["cells"]}
    current_cells = {(cell["x"], cell["y"]): cell for cell in current["cells"]}
    changed_cells = [
        deepcopy(cell)
        for point, cell in current_cells.items()
        if previous_cells.get(point) != cell
    ]
    removed_cells: list[Position] = [
        {"x": x, "y": y} for x, y in previous_cells.keys() - current_cells.keys()
    ]
    menu_changed = previous["menu"] != current["menu"]
    input_mode_changed = previous["input_mode"] != current["input_mode"]
    return {
        "player": cast(PlayerView, changed_player),
        "removed_player_fields": sorted(removed_player_fields),
        "cells": sorted(changed_cells, key=lambda cell: (cell["y"], cell["x"])),
        "removed_cells": sorted(
            removed_cells, key=lambda position: (position["y"], position["x"])
        ),
        "messages": deepcopy(current["messages"]),
        "menu_changed": menu_changed,
        "menu": deepcopy(current["menu"]) if menu_changed else None,
        "input_mode_changed": input_mode_changed,
        "input_mode": current["input_mode"] if input_mode_changed else None,
    }


def apply_observation_delta(
    previous: ObservationData, delta: ObservationDeltaData
) -> ObservationData:
    """Reconstruct the exact next semantic snapshot from a schema-v2 patch."""
    player_values = cast(dict[str, object], deepcopy(previous["player"]))
    for key in delta["removed_player_fields"]:
        player_values.pop(key, None)
    player_values.update(cast(dict[str, object], delta["player"]))
    player = cast(PlayerView, player_values)
    cells = {(cell["x"], cell["y"]): deepcopy(cell) for cell in previous["cells"]}
    for position in delta["removed_cells"]:
        cells.pop((position["x"], position["y"]), None)
    for cell in delta["cells"]:
        cells[(cell["x"], cell["y"])] = deepcopy(cell)
    ordered_cells: list[CellView] = sorted(
        cells.values(), key=lambda cell: (cell["y"], cell["x"])
    )
    return {
        "player": player,
        "cells": ordered_cells,
        "messages": deepcopy(delta["messages"]),
        "menu": deepcopy(delta["menu"])
        if delta["menu_changed"]
        else deepcopy(previous["menu"]),
        "input_mode": delta["input_mode"]
        if delta["input_mode_changed"]
        else previous["input_mode"],
    }


class TrajectoryWriter:
    """Append-only writer; every complete line remains independently readable."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("x", encoding="utf-8")
        self._started = False
        self._step = 0

    def start(
        self,
        env: DcssEnv,
        observation: ObservationData,
        *,
        agent_id: str,
        checkpoint_id: str | None = None,
    ) -> EpisodeMetadata:
        if self._started or env.game is None:
            raise RuntimeError("trajectory already started or environment not reset")
        rc_path = env.game.run_root / "crawl.rc"
        version = next(
            (
                message.payload.get("text", "unknown")
                for batch in env.last_exchange
                for message in batch.observations
                if message.kind == "version"
            ),
            "unknown",
        )
        metadata = EpisodeMetadata(
            episode_id=str(uuid.uuid4()),
            created_at=datetime.now(UTC).isoformat(),
            dcss_version=str(version),
            dcss_commit=_revision(env.binary.resolve().parent),
            code_revision=_revision(Path.cwd()),
            seed=env.game.config.seed,
            character={
                "species": env.game.config.species,
                "background": env.game.config.background,
                "starting_weapon_key": env.starting_weapon_key,
            },
            agent_id=agent_id,
            checkpoint_id=checkpoint_id,
            reward_spec="depth_xl_terminal_v1",
            rc_sha256=hashlib.sha256(rc_path.read_bytes()).hexdigest(),
            static_data_identity=env.game.static_cache.identity
            if env.game.static_cache is not None
            else None,
            static_cache_preparation_timing=env.game.static_cache_preparation_timing,
        )
        self._write(
            {
                "type": "episode",
                "schema_version": SCHEMA_VERSION,
                "metadata": asdict(metadata),
                "initial": {
                    "raw_messages": _raw_batches(env.last_exchange),
                    "setup_keycodes": list(env.last_keycodes),
                    "observation": observation,
                },
            }
        )
        self._started = True
        return metadata

    def transition(
        self,
        env: DcssEnv,
        previous: ObservationData,
        action_index: ActionIndex,
        observation: ObservationData,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: EnvironmentInfo,
        *,
        collection_provenance: CollectionProvenance | None = None,
        sampling_evidence: SamplingEvidence | None = None,
    ) -> None:
        if not self._started:
            raise RuntimeError("trajectory has not started")
        action = index_to_action(action_index).to_dict()
        delta = observation_delta(previous, observation)
        segments = (
            LossSegment("action", _json(action), "policy"),
            LossSegment("environment", _json(delta), "environment"),
        )
        self._write(
            {
                "type": "transition",
                **(
                    {"sampling_evidence": asdict(sampling_evidence)}
                    if sampling_evidence is not None
                    else {}
                ),
                **(
                    {"collection_provenance": asdict(collection_provenance)}
                    if collection_provenance is not None
                    else {}
                ),
                "step": self._step,
                "action_index": action_index,
                "action": action,
                "emitted_keycodes": list(env.last_keycodes),
                "raw_messages": _raw_batches(env.last_exchange),
                "observation_delta": delta,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "outcome": info.get("outcome"),
                "training_segments": [asdict(segment) for segment in segments],
            }
        )
        self._step += 1

    def _write(self, record: dict[str, Any]) -> None:
        self._file.write(_json(record) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> TrajectoryWriter:
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()


class RecordingEnv:
    """Small composition wrapper that records every reset and transition."""

    def __init__(
        self,
        env: DcssEnv,
        writer: TrajectoryWriter,
        *,
        agent_id: str,
        checkpoint_id: str | None = None,
    ) -> None:
        self.env = env
        self.writer = writer
        self.agent_id = agent_id
        self.checkpoint_id = checkpoint_id
        self._observation: ObservationData | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: ResetOptions | None = None,
    ) -> tuple[ObservationData, EnvironmentInfo]:
        observation, info = self.env.reset_typed(seed=seed, options=options)
        self.writer.start(
            self.env,
            observation,
            agent_id=self.agent_id,
            checkpoint_id=self.checkpoint_id,
        )
        self._observation = observation
        return observation, info

    def step(
        self, action_index: ActionIndex
    ) -> tuple[ObservationData, float, bool, bool, EnvironmentInfo]:
        if self._observation is None:
            raise RuntimeError("reset must be called before step")
        previous = self._observation
        observation, reward, terminated, truncated, info = self.env.step_typed(
            action_index
        )
        self.writer.transition(
            self.env,
            previous,
            action_index,
            observation,
            reward,
            terminated,
            truncated,
            info,
        )
        self._observation = observation
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self.writer.close()
        self.env.close()
