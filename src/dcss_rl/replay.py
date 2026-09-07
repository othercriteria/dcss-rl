"""Terminal replay support for champion trajectories."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from dcss_rl.schema import ObservationData, ObservationDeltaData
from dcss_rl.trajectory import apply_observation_delta
from dcss_rl.units import FrameLimit, Seconds, ViewRadius


@dataclass(frozen=True, slots=True)
class ReplayFrame:
    step: int
    action: str
    observation: ObservationData


@dataclass(frozen=True, slots=True)
class ChampionEpisode:
    case_id: str
    outcome: str
    trajectory: Path


def champion_trajectory(manifest_path: Path, case_id: str | None = None) -> Path:
    """Resolve a champion episode deterministically in manifest suite order."""
    manifest = _json_object(Path(manifest_path).read_text())
    summary = manifest.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("champion manifest has no summary object")
    summary_fields = cast(dict[str, object], summary)
    episodes = summary_fields.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("champion manifest has no episodes")
    selected: object | None = None
    if case_id is None:
        selected = episodes[0]
    else:
        for episode in episodes:
            if (
                isinstance(episode, dict)
                and cast(dict[str, object], episode).get("case_id") == case_id
            ):
                selected = episode
                break
    if not isinstance(selected, dict):
        raise ValueError(f"champion manifest has no case {case_id!r}")
    trajectory = cast(dict[str, object], selected).get("trajectory")
    if not isinstance(trajectory, str):
        raise ValueError("champion episode has no trajectory path")
    return Path(trajectory)


def champion_episodes(manifest_path: Path) -> tuple[ChampionEpisode, ...]:
    """Resolve every champion episode in fixed manifest order."""
    manifest = _json_object(Path(manifest_path).read_text())
    summary = manifest.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("champion manifest has no summary object")
    episodes = cast(dict[str, object], summary).get("episodes")
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
        result.append(ChampionEpisode(case_id, outcome, Path(trajectory)))
    return tuple(result)


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
) -> None:
    """Render a trajectory as a compact player-centered terminal animation."""
    for number, frame in enumerate(replay_frames(path)):
        if frame_limit is not None and number >= frame_limit:
            break
        if animate:
            print("\x1b[2J\x1b[H", end="")
        print(render_frame(frame, view_radius=view_radius), flush=True)
        if animate and frame_delay > 0:
            time.sleep(frame_delay)


def watch_grid(
    manifest_path: Path,
    *,
    frame_delay: Seconds,
    view_radius: ViewRadius,
    columns: int = 3,
    frame_limit: FrameLimit | None = None,
    animate: bool = True,
) -> None:
    """Animate all fixed-suite champion episodes in a synchronized terminal grid."""
    if columns < 1:
        raise ValueError("grid columns must be positive")
    episodes = champion_episodes(manifest_path)
    frame_sets = tuple(tuple(replay_frames(episode.trajectory)) for episode in episodes)
    frame_count = max(len(frames) for frames in frame_sets)
    if frame_limit is not None:
        frame_count = min(frame_count, frame_limit)
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
        print(_join_panels(panels, columns=columns), flush=True)
        if animate and frame_delay > 0:
            time.sleep(frame_delay)


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
    width = 2 * view_radius + 1
    terminal = _terminal_label(episode.outcome) if finished else "RUNNING"
    case_label = (
        episode.case_id
        if len(episode.case_id) <= width
        else f"…{episode.case_id[-(width - 1) :]}"
    )
    header = case_label.ljust(width)
    outcome = terminal[:width].ljust(width)
    status = (
        f"D{player.get('depth', '?')} X{player.get('xl', '?')} "
        f"H{player.get('hp', '?')}/{player.get('hp_max', '?')}"
    )[:width].ljust(width)
    rows = tuple(
        "".join(
            glyphs.get((x, y), " ")
            for x in range(position["x"] - view_radius, position["x"] + view_radius + 1)
        )
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


def _join_panels(panels: tuple[tuple[str, ...], ...], *, columns: int) -> str:
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
