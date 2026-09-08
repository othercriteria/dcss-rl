import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from dcss_rl.env import ACTION_COUNT
from dcss_rl.learned import ModelConfig, SemanticActorCritic
from dcss_rl.policy import ScriptedMibePolicy
from dcss_rl.ppo import _preloaded_imitation_replay
from dcss_rl.schema import JsonObject, ObservationData
from dcss_rl.training import (
    ImitationReplayEpisode,
    load_imitation_episode,
    load_imitation_replay_episode,
)
from dcss_rl.trajectory import observation_delta


def _observation(turn: int, *, messages: list[str]) -> ObservationData:
    return {
        "player": {
            "pos": {"x": 0, "y": 0},
            "turn": turn,
            "hp": 10,
            "hp_max": 10,
            "xl": 1,
            "depth": 1,
        },
        "cells": [
            {"x": 0, "y": 0, "g": "@"},
            {"x": 1, "y": 0, "g": ">"},
        ],
        "messages": messages,
        "menu": None,
        "input_mode": 1,
    }


def _trajectory(path: Path) -> None:
    observations = (
        _observation(0, messages=[]),
        _observation(1, messages=["You move."]),
        _observation(2, messages=["You move again."]),
    )
    records: tuple[JsonObject, ...] = (
        cast(
            JsonObject, {"type": "header", "initial": {"observation": observations[0]}}
        ),
        cast(
            JsonObject,
            {
                "action_index": 0,
                "reward": 0.0,
                "observation_delta": observation_delta(
                    observations[0], observations[1]
                ),
            },
        ),
        cast(
            JsonObject,
            {
                "action_index": 0,
                "reward": 1.0,
                "observation_delta": observation_delta(
                    observations[1], observations[2]
                ),
            },
        ),
    )
    path.write_text("".join(f"{json.dumps(record)}\n" for record in records))


def test_online_replay_loader_matches_full_offline_products(tmp_path: Path) -> None:
    trajectory = tmp_path / "trajectory.jsonl"
    _trajectory(trajectory)
    teacher = ScriptedMibePolicy()

    complete = load_imitation_episode(trajectory, discount=0.99, teacher=teacher)
    replay = load_imitation_replay_episode(trajectory, teacher=teacher)

    np.testing.assert_array_equal(replay.features, complete.features)
    np.testing.assert_array_equal(replay.masks, complete.masks)
    assert replay.actions == complete.actions


def test_preloaded_replay_cache_is_content_addressed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trajectory = tmp_path / "trajectory.jsonl"
    _trajectory(trajectory)
    cache_directory = tmp_path / "cache"
    teacher = ScriptedMibePolicy()
    model = SemanticActorCritic(ModelConfig(action_count=int(ACTION_COUNT)))
    load_calls = 0

    def count_load(
        path: Path, *, teacher: ScriptedMibePolicy
    ) -> ImitationReplayEpisode:
        nonlocal load_calls
        load_calls += 1
        return load_imitation_replay_episode(path, teacher=teacher)

    monkeypatch.setattr("dcss_rl.ppo.load_imitation_replay_episode", count_load)

    first = _preloaded_imitation_replay(
        (trajectory,),
        model=model,
        teacher=teacher,
        cache_directory=cache_directory,
    )
    second = _preloaded_imitation_replay(
        (trajectory,),
        model=model,
        teacher=teacher,
        cache_directory=cache_directory,
    )
    original = trajectory.read_text()
    trajectory.write_text(original.replace("\n", " \n", 1))
    third = _preloaded_imitation_replay(
        (trajectory,),
        model=model,
        teacher=teacher,
        cache_directory=cache_directory,
    )

    assert not first.cache_hit
    assert second.cache_hit
    assert not third.cache_hit
    assert load_calls == 2
    np.testing.assert_array_equal(second.replay[0].features, first.replay[0].features)
    np.testing.assert_array_equal(second.replay[0].masks, first.replay[0].masks)
    np.testing.assert_array_equal(
        second.replay[0].teacher_actions, first.replay[0].teacher_actions
    )
