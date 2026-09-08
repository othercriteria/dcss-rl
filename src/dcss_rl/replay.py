"""Terminal replay support for champion trajectories."""

from __future__ import annotations

import json
import os
import select
import sys
import termios
import time
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import cast

from dcss_rl.schema import ObservationData, ObservationDeltaData
from dcss_rl.trajectory import apply_observation_delta
from dcss_rl.units import (
    CaseCount,
    CaseId,
    CheckpointId,
    FrameLimit,
    GridColumnCount,
    PolicyId,
    Seconds,
    SuiteId,
    TerminalOutcome,
    ViewRadius,
)

type TerminalAttributes = (
    list[int | list[bytes | int]] | list[int | list[bytes]] | list[int | list[int]]
)
_DEFAULT_GRID_COLUMNS = GridColumnCount(3)


@dataclass(frozen=True, slots=True)
class ReplayFrame:
    step: int
    action: str
    observation: ObservationData


@dataclass(frozen=True, slots=True)
class ChampionEpisode:
    case_id: CaseId
    outcome: TerminalOutcome
    trajectory: Path


@dataclass(frozen=True, slots=True)
class ChampionManifest:
    path: Path
    suite_id: SuiteId
    policy_id: PolicyId
    episodes: tuple[ChampionEpisode, ...]


@dataclass(frozen=True, slots=True)
class ReplayIdentity:
    suite_id: SuiteId
    policy_id: PolicyId
    checkpoint_id: CheckpointId | None
    case_id: CaseId
    outcome: TerminalOutcome
    manifest_path: Path


class PlaybackCommand(Enum):
    WAIT = "wait"
    ADVANCE = "advance"
    QUIT = "quit"


class _PlaybackControls:
    """Small terminal state machine shared by single and grid replay."""

    def __init__(self) -> None:
        self.paused = False

    def accept(self, key: str) -> PlaybackCommand:
        if key.lower() == "q":
            return PlaybackCommand.QUIT
        if key == " ":
            self.paused = not self.paused
            return PlaybackCommand.WAIT
        if self.paused and key.lower() == "n":
            return PlaybackCommand.ADVANCE
        return PlaybackCommand.WAIT


class _InteractiveTerminal:
    """Enable single-key replay controls while restoring the caller's terminal."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled and sys.stdin.isatty() and sys.stdout.isatty()
        self._descriptor: int | None = None
        self._attributes: TerminalAttributes | None = None
        self.controls = _PlaybackControls()

    def __enter__(self) -> _InteractiveTerminal:
        if self.enabled:
            self._descriptor = sys.stdin.fileno()
            self._attributes = termios.tcgetattr(self._descriptor)
            import tty

            tty.setcbreak(self._descriptor)
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        if self._descriptor is not None and self._attributes is not None:
            termios.tcsetattr(self._descriptor, termios.TCSADRAIN, self._attributes)

    def wait(self, delay: Seconds) -> PlaybackCommand:
        if not self.enabled or self._descriptor is None:
            if delay > 0:
                time.sleep(delay)
            return PlaybackCommand.ADVANCE
        deadline = time.monotonic() + delay
        while True:
            timeout = (
                None if self.controls.paused else max(deadline - time.monotonic(), 0)
            )
            readable, _, _ = select.select([self._descriptor], [], [], timeout)
            if not readable:
                return PlaybackCommand.ADVANCE
            raw_key = os.read(self._descriptor, 1)
            if not raw_key:
                return PlaybackCommand.ADVANCE
            command = self.controls.accept(raw_key.decode(errors="ignore"))
            if command is not PlaybackCommand.WAIT:
                return command


def champion_trajectory(manifest_path: Path, case_id: str | None = None) -> Path:
    """Resolve a champion episode deterministically in manifest suite order."""
    manifest = load_champion_manifest(manifest_path)
    return select_champion_episode(manifest, case_id).trajectory


def select_champion_episode(
    manifest: ChampionManifest, case_id: str | None = None
) -> ChampionEpisode:
    """Select one episode, defaulting deliberately to fixed manifest order."""
    episodes = manifest.episodes
    selected: ChampionEpisode | None = None
    if case_id is None:
        selected = episodes[0]
    else:
        for episode in episodes:
            if episode.case_id == case_id:
                selected = episode
                break
    if selected is None:
        raise ValueError(f"champion manifest has no case {case_id!r}")
    return selected


def champion_episodes(manifest_path: Path) -> tuple[ChampionEpisode, ...]:
    """Resolve every champion episode in fixed manifest order."""
    return load_champion_manifest(manifest_path).episodes


def load_champion_manifest(manifest_path: Path) -> ChampionManifest:
    """Decode one champion manifest into validated semantic replay objects."""
    manifest = _json_object(Path(manifest_path).read_text())
    suite_id = manifest.get("suite_id")
    policy_id = manifest.get("policy_id")
    summary = manifest.get("summary")
    if not isinstance(suite_id, str) or not isinstance(policy_id, str):
        raise ValueError("champion manifest lacks suite or policy ID")
    if not isinstance(summary, dict):
        raise ValueError("champion manifest has no summary object")
    raw_summary = cast(dict[str, object], summary)
    if (
        raw_summary.get("suite_id") != suite_id
        or raw_summary.get("policy_id") != policy_id
    ):
        raise ValueError("champion manifest and summary identities disagree")
    episodes = raw_summary.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("champion manifest has no episodes")
    result: list[ChampionEpisode] = []
    for raw_episode in episodes:
        if not isinstance(raw_episode, dict):
            raise ValueError("champion episode must be an object")
        episode = cast(dict[str, object], raw_episode)
        case_id = episode.get("case_id")
        outcome = episode.get("outcome")
        trajectory = episode.get("trajectory")
        if not isinstance(case_id, str):
            raise ValueError("champion episode lacks case")
        if not isinstance(outcome, str):
            raise ValueError("champion episode lacks outcome")
        if not isinstance(trajectory, str):
            raise ValueError("champion episode lacks case, outcome, or trajectory")
        result.append(
            ChampionEpisode(CaseId(case_id), TerminalOutcome(outcome), Path(trajectory))
        )
    return ChampionManifest(
        Path(manifest_path), SuiteId(suite_id), PolicyId(policy_id), tuple(result)
    )


def replay_identity(
    manifest: ChampionManifest, episode: ChampionEpisode
) -> ReplayIdentity:
    """Resolve manifest and checkpoint identity without reading trajectory payloads."""
    with episode.trajectory.open(encoding="utf-8") as trajectory_file:
        first_line = trajectory_file.readline()
    header = _json_object(first_line)
    metadata = header.get("metadata")
    checkpoint_id: CheckpointId | None = None
    if isinstance(metadata, dict):
        raw_checkpoint = cast(dict[str, object], metadata).get("checkpoint_id")
        if raw_checkpoint is not None and not isinstance(raw_checkpoint, str):
            raise ValueError("trajectory checkpoint ID must be a string or null")
        checkpoint_id = (
            CheckpointId(raw_checkpoint) if isinstance(raw_checkpoint, str) else None
        )
    return ReplayIdentity(
        manifest.suite_id,
        manifest.policy_id,
        checkpoint_id,
        episode.case_id,
        episode.outcome,
        manifest.path,
    )


def replay_frames(path: Path) -> Iterator[ReplayFrame]:
    """Reconstruct observations from either trajectory schema v1 or v2."""
    lines = Path(path).read_text().splitlines()
    if not lines:
        raise ValueError("trajectory is empty")
    header = _json_object(lines[0])
    initial = header.get("initial")
    if not isinstance(initial, dict):
        raise ValueError("trajectory has no initial record")
    raw_observation = cast(dict[str, object], initial).get("observation")
    if not isinstance(raw_observation, dict):
        raise ValueError("trajectory has no initial observation")
    observation = cast(ObservationData, raw_observation)
    yield ReplayFrame(-1, "initial", observation)
    for line in lines[1:]:
        record = _json_object(line)
        raw_action = record.get("action")
        action = (
            cast(dict[str, object], raw_action).get("kind", "unknown")
            if isinstance(raw_action, dict)
            else "unknown"
        )
        if not isinstance(action, str):
            action = "unknown"
        full = record.get("observation")
        delta = record.get("observation_delta")
        if isinstance(full, dict):
            observation = cast(ObservationData, full)
        elif isinstance(delta, dict):
            observation = apply_observation_delta(
                observation, cast(ObservationDeltaData, delta)
            )
        else:
            raise ValueError("transition has neither observation nor observation_delta")
        step = record.get("step")
        if not isinstance(step, int):
            raise ValueError("transition step is not an integer")
        yield ReplayFrame(step, action, observation)


def watch_replay(
    path: Path,
    *,
    frame_delay: Seconds,
    view_radius: ViewRadius,
    frame_limit: FrameLimit | None = None,
    animate: bool = True,
    identity: ReplayIdentity | None = None,
) -> None:
    """Render a trajectory as a compact player-centered terminal animation."""
    with _InteractiveTerminal(animate) as terminal:
        for number, frame in enumerate(replay_frames(path)):
            if frame_limit is not None and number >= frame_limit:
                break
            if animate:
                print("\x1b[2J\x1b[H", end="")
            heading = _identity_heading(identity, interactive=terminal.enabled)
            print(
                "\n".join((*heading, render_frame(frame, view_radius=view_radius))),
                flush=True,
            )
            if animate and terminal.wait(frame_delay) is PlaybackCommand.QUIT:
                break


def watch_grid(
    manifest_path: Path,
    *,
    frame_delay: Seconds,
    view_radius: ViewRadius,
    columns: GridColumnCount = _DEFAULT_GRID_COLUMNS,
    frame_limit: FrameLimit | None = None,
    animate: bool = True,
) -> None:
    """Animate all fixed-suite champion episodes in a synchronized terminal grid."""
    if columns < 1:
        raise ValueError("grid columns must be positive")
    manifest = load_champion_manifest(manifest_path)
    episodes = manifest.episodes
    identity = replay_identity(manifest, episodes[0])
    frame_sets = tuple(tuple(replay_frames(episode.trajectory)) for episode in episodes)
    frame_count = max(len(frames) for frames in frame_sets)
    if frame_limit is not None:
        frame_count = min(frame_count, frame_limit)
    with _InteractiveTerminal(animate) as terminal:
        for index in range(frame_count):
            panels = tuple(
                _grid_panel(
                    episode,
                    frames[min(index, len(frames) - 1)],
                    finished=index >= len(frames) - 1,
                    view_radius=view_radius,
                )
                for episode, frames in zip(episodes, frame_sets, strict=True)
            )
            if animate:
                print("\x1b[2J\x1b[H", end="")
            heading = _identity_heading(
                identity,
                interactive=terminal.enabled,
                case_count=CaseCount(len(episodes)),
            )
            print(
                "\n".join((*heading, _join_panels(panels, columns=columns))),
                flush=True,
            )
            if animate and terminal.wait(frame_delay) is PlaybackCommand.QUIT:
                break


def _identity_heading(
    identity: ReplayIdentity | None,
    *,
    interactive: bool,
    case_count: CaseCount | None = None,
) -> tuple[str, ...]:
    if identity is None:
        return ()
    checkpoint = identity.checkpoint_id or "none (scripted/unrecorded)"
    controls = "controls: [q] quit  [space] pause/resume  [n] step while paused"
    case_status = (
        f"cases={case_count} (fixed manifest order; status shown per panel)"
        if case_count is not None
        else (
            f"case={identity.case_id}  "
            f"recorded_status={_terminal_label(identity.outcome)}"
        )
    )
    return (
        f"suite={identity.suite_id}  policy={identity.policy_id}",
        f"checkpoint={checkpoint}",
        case_status,
        f"manifest={identity.manifest_path}",
        controls if interactive else "controls=disabled (stdin/stdout are not TTYs)",
        "",
    )


def render_frame(frame: ReplayFrame, *, view_radius: ViewRadius) -> str:
    observation = frame.observation
    player = observation["player"]
    position = player.get("pos", {"x": 0, "y": 0})
    center_x, center_y = position["x"], position["y"]
    glyphs = {
        (cell["x"], cell["y"]): cell.get("g", " ") for cell in observation["cells"]
    }
    rows = [
        "".join(
            glyphs.get((x, y), " ")
            for x in range(center_x - view_radius, center_x + view_radius + 1)
        )
        for y in range(center_y - view_radius, center_y + view_radius + 1)
    ]
    status = (
        f"step={frame.step} action={frame.action} "
        f"D:{player.get('depth', '?')} XL:{player.get('xl', '?')} "
        f"HP:{player.get('hp', '?')}/{player.get('hp_max', '?')}"
    )
    messages = "\n".join(observation["messages"][-3:])
    return "\n".join((status, *rows, messages)).rstrip()


def _grid_panel(
    episode: ChampionEpisode,
    frame: ReplayFrame,
    *,
    finished: bool,
    view_radius: ViewRadius,
) -> tuple[str, ...]:
    player = frame.observation["player"]
    position = player.get("pos", {"x": 0, "y": 0})
    glyphs = {
        (cell["x"], cell["y"]): cell.get("g", " ")
        for cell in frame.observation["cells"]
    }
    map_width = 2 * view_radius + 1
    terminal = _terminal_label(episode.outcome) if finished else "RUNNING"
    raw_status = (
        f"step={frame.step} D{player.get('depth', '?')} X{player.get('xl', '?')} "
        f"HP{player.get('hp', '?')}/{player.get('hp_max', '?')}"
    )
    panel_width = max(map_width, len(episode.case_id), len(terminal), len(raw_status))
    header = episode.case_id.ljust(panel_width)
    outcome = terminal.ljust(panel_width)
    status = raw_status.ljust(panel_width)
    rows = tuple(
        "".join(
            glyphs.get((x, y), " ")
            for x in range(position["x"] - view_radius, position["x"] + view_radius + 1)
        ).ljust(panel_width)
        for y in range(position["y"] - view_radius, position["y"] + view_radius + 1)
    )
    return (header, outcome, status, *rows)


def _terminal_label(outcome: str) -> str:
    if outcome == "won":
        return "★ ASCENDED"
    if outcome == "dead":
        return "☠ DEAD"
    if outcome == "truncated":
        return "◇ HORIZON"
    return f"■ {outcome.upper()}"


def _join_panels(
    panels: tuple[tuple[str, ...], ...], *, columns: GridColumnCount
) -> str:
    rows: list[str] = []
    for start in range(0, len(panels), columns):
        group = panels[start : start + columns]
        rows.extend("   ".join(lines) for lines in zip(*group, strict=True))
        rows.append("")
    return "\n".join(rows).rstrip()


def _json_object(text: str) -> dict[str, object]:
    decoded: object = json.loads(text)
    if not isinstance(decoded, dict) or not all(
        isinstance(key, str) for key in decoded
    ):
        raise ValueError("expected a JSON object")
    return cast(dict[str, object], decoded)
