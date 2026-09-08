"""Repeatable end-to-end compatibility checks for supported DCSS binaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from dcss_rl.actions import Action, ActionKind
from dcss_rl.env import DcssEnv, action_to_index
from dcss_rl.replay import replay_frames
from dcss_rl.trajectory import RecordingEnv, TrajectoryWriter
from dcss_rl.units import DcssVersion, GameSeed, StepLimit, VisibleCellCount
from dcss_rl.webtiles import GameConfig
from dcss_rl.webtiles.cache import StaticDataCache


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    """Evidence returned by one cross-version environment/trajectory smoke."""

    version: DcssVersion
    species: str
    depth: int
    visible_cells: VisibleCellCount


def run_compatibility_smoke(
    binary: Path, *, seed: GameSeed, static_cache: StaticDataCache | None = None
) -> CompatibilityReport:
    """Reset, step, record, and exactly replay one unmodified DCSS game."""
    with TemporaryDirectory(prefix="dcss-rl-compatibility-") as directory:
        root = Path(directory)
        trajectory_path = root / "trajectory.jsonl"
        recording = RecordingEnv(
            DcssEnv(
                binary,
                game_config=GameConfig(seed=seed),
                max_steps=StepLimit(1),
                run_root=root / "game",
                static_cache=static_cache,
            ),
            TrajectoryWriter(trajectory_path),
            agent_id="compatibility-smoke",
        )
        try:
            initial, _ = recording.reset()
            version = _version_from_initial_exchange(recording.env)
            current, _, _, _, _ = recording.step(
                action_to_index(Action(ActionKind.WAIT))
            )
        finally:
            recording.close()

        replayed = tuple(replay_frames(trajectory_path))
        if not replayed or replayed[-1].observation != current:
            raise AssertionError(
                "trajectory replay did not reconstruct the final state"
            )
        species = initial["player"].get("species")
        depth = initial["player"].get("depth")
        if not isinstance(species, str) or not isinstance(depth, int):
            raise AssertionError("initial semantic observation lacks character state")
        return CompatibilityReport(
            version=version,
            species=species,
            depth=depth,
            visible_cells=VisibleCellCount(len(initial["cells"])),
        )


def _version_from_initial_exchange(env: DcssEnv) -> DcssVersion:
    for batch in env.last_exchange:
        for message in batch.observations:
            if message.kind != "version":
                continue
            version = message.payload.get("text")
            if isinstance(version, str):
                return DcssVersion(version)
    raise AssertionError("initial WebTiles exchange lacks a version message")
