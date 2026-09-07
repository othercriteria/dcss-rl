"""Replayable JSONL trajectories with policy and ECHO loss segmentation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

from dcss_rl.env import DcssEnv, index_to_action
from dcss_rl.schema import ObservationData
from dcss_rl.units import ActionIndex
from dcss_rl.webtiles import ObservationBatch

SCHEMA_VERSION = 1


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
    character: dict[str, str]
    agent_id: str
    checkpoint_id: str | None
    reward_spec: str
    rc_sha256: str


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


def _raw_batches(batches: tuple[ObservationBatch, ...]) -> list[dict[str, Any]]:
    return [
        {"control": message.control, "payload": message.payload}
        for batch in batches
        for message in batch.messages
    ]


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
        info: dict[str, Any],
    ) -> None:
        if not self._started:
            raise RuntimeError("trajectory has not started")
        action = index_to_action(action_index).to_dict()
        segments = (
            LossSegment("environment", _json(previous), None),
            LossSegment("action", _json(action), "policy"),
            LossSegment("environment", _json(observation), "environment"),
        )
        self._write(
            {
                "type": "transition",
                "step": self._step,
                "action_index": action_index,
                "action": action,
                "emitted_keycodes": list(env.last_keycodes),
                "raw_messages": _raw_batches(env.last_exchange),
                "observation": observation,
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

    def reset(self, **kwargs: Any) -> tuple[ObservationData, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
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
    ) -> tuple[ObservationData, float, bool, bool, dict[str, Any]]:
        if self._observation is None:
            raise RuntimeError("reset must be called before step")
        previous = self._observation
        result = self.env.step(action_index)
        observation, reward, terminated, truncated, info = result
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
        return result

    def close(self) -> None:
        self.writer.close()
        self.env.close()
