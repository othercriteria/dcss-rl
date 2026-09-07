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


def _json_object(text: str) -> dict[str, object]:
    decoded: object = json.loads(text)
    if not isinstance(decoded, dict) or not all(
        isinstance(key, str) for key in decoded
    ):
        raise ValueError("expected a JSON object")
    return cast(dict[str, object], decoded)
