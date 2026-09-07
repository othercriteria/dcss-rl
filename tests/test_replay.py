import json
from pathlib import Path

import pytest

from dcss_rl.replay import (
    champion_episodes,
    champion_trajectory,
    render_frame,
    replay_frames,
    watch_grid,
)
from dcss_rl.schema import ObservationData
from dcss_rl.trajectory import observation_delta
from dcss_rl.units import FrameLimit, Seconds, ViewRadius


def state(*, hp: int, x: int = 0) -> ObservationData:
    return {
        "player": {
            "hp": hp,
            "hp_max": 20,
            "xl": 2,
            "depth": 1,
            "pos": {"x": x, "y": 0},
        },
        "cells": [{"x": x, "y": 0, "g": "@"}],
        "messages": ["hello"],
        "menu": None,
        "input_mode": 1,
    }


def test_replays_compact_observation_delta(tmp_path: Path) -> None:
    initial = state(hp=20)
    current = state(hp=18, x=1)
    path = tmp_path / "trajectory.jsonl"
    records = [
        {"type": "episode", "initial": {"observation": initial}},
        {
            "type": "transition",
            "step": 0,
            "action": {"kind": "move_e"},
            "observation_delta": observation_delta(initial, current),
        },
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    frames = list(replay_frames(path))

    assert frames[-1].observation == current
    assert "HP:18/20" in render_frame(frames[-1], view_radius=ViewRadius(1))


def test_champion_defaults_to_first_manifest_case(tmp_path: Path) -> None:
    manifest = tmp_path / "champion.json"
    manifest.write_text(
        json.dumps(
            {
                "summary": {
                    "episodes": [
                        {
                            "case_id": "first",
                            "outcome": "dead",
                            "trajectory": "first.jsonl",
                        },
                        {
                            "case_id": "second",
                            "outcome": "won",
                            "trajectory": "second.jsonl",
                        },
                    ]
                }
            }
        )
    )

    assert champion_trajectory(manifest) == Path("first.jsonl")
    assert champion_trajectory(manifest, "second") == Path("second.jsonl")
    assert [episode.outcome for episode in champion_episodes(manifest)] == [
        "dead",
        "won",
    ]


def test_watch_grid_marks_terminal_outcomes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trajectories = []
    for case_id in ("dead-seed", "winning-seed"):
        path = tmp_path / f"{case_id}.jsonl"
        path.write_text(
            json.dumps({"type": "episode", "initial": {"observation": state(hp=20)}})
            + "\n"
        )
        trajectories.append(path)
    manifest = tmp_path / "champion.json"
    manifest.write_text(
        json.dumps(
            {
                "summary": {
                    "episodes": [
                        {
                            "case_id": "dead-seed",
                            "outcome": "dead",
                            "trajectory": str(trajectories[0]),
                        },
                        {
                            "case_id": "winning-seed",
                            "outcome": "won",
                            "trajectory": str(trajectories[1]),
                        },
                    ]
                }
            }
        )
    )

    watch_grid(
        manifest,
        frame_delay=Seconds(0),
        view_radius=ViewRadius(5),
        columns=2,
        frame_limit=FrameLimit(1),
        animate=False,
    )

    output = capsys.readouterr().out
    assert "☠ DEAD" in output
    assert "★ ASCENDED" in output
