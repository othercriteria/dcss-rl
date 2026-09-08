import json
from itertools import pairwise
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from dcss_rl.policy import ScriptedMibePolicy
from dcss_rl.replay_cache import imitation_replay_cache_key, prepare_imitation_replay
from dcss_rl.schema import JsonObject, ObservationData
from dcss_rl.training import (
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


@pytest.mark.parametrize("full", [False, True])
def test_mixed_reconstruction_matches_reference(tmp_path: Path, full: bool) -> None:
    observations = [_observation(i, messages=[]) for i in range(5)]
    observations[0]["cells"].reverse()
    observations[1]["player"].pop("hp")
    observations[1]["cells"] = [{"x": 0, "y": 0, "g": "@"}]
    observations[1]["menu"] = {
        "type": "ability",
        "prompt": "Use which ability?",
        "choices": [{"keycode": 97, "text": "Berserk"}],
    }
    observations[1]["input_mode"] = 2
    observations[2]["menu"] = {
        "type": "ability",
        "prompt": "Use which ability?",
        "choices": [
            {
                "keycode": 97,
                "text": "Berserk (unusable)",
                "applicability": "inapplicable",
            }
        ],
    }
    observations[3]["player"]["pos"] = {"x": 1, "y": 0}
    observations[3]["cells"].reverse()
    records = [json.dumps({"initial": {"observation": observations[0]}})]
    for previous, current in pairwise(observations):
        payload = (
            {"observation": current}
            if full
            else {"observation_delta": observation_delta(previous, current)}
        )
        records.append(json.dumps({"action_index": 0, "reward": 0.0, **payload}))
    path = tmp_path / "mixed.jsonl"
    path.write_text("\n".join(records) + "\n")
    reference = load_imitation_episode(
        path, discount=0.99, teacher=ScriptedMibePolicy()
    )
    replay = load_imitation_replay_episode(path, teacher=ScriptedMibePolicy())
    np.testing.assert_array_equal(replay.features, reference.features)
    np.testing.assert_array_equal(replay.masks, reference.masks)
    assert replay.actions == reference.actions


def test_cache_reuse_and_content_contract_invalidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "trajectory.jsonl"
    _trajectory(path)
    teacher = ScriptedMibePolicy()
    cold = prepare_imitation_replay((path,), teacher=teacher, cache_directory=tmp_path)
    warm = prepare_imitation_replay((path,), teacher=teacher, cache_directory=tmp_path)
    assert not cold.cache_hit and warm.cache_hit
    np.testing.assert_array_equal(cold.features, warm.features)
    np.testing.assert_array_equal(cold.masks, warm.masks)
    np.testing.assert_array_equal(cold.teacher_actions, warm.teacher_actions)
    key = imitation_replay_cache_key((path,), teacher=teacher)
    monkeypatch.setattr("dcss_rl.replay_cache._CONTRACT", "changed")
    assert imitation_replay_cache_key((path,), teacher=teacher) != key
    assert not prepare_imitation_replay(
        (path,), teacher=teacher, cache_directory=tmp_path
    ).cache_hit
    path.write_text(path.read_text().replace("\n", " \n", 1))
    assert not prepare_imitation_replay(
        (path,), teacher=teacher, cache_directory=tmp_path
    ).cache_hit


@pytest.mark.parametrize(
    "corruption", ["bytes", "npy", "scalar", "illegal", "wrong-key", "nan"]
)
def test_corrupt_cache_rebuilds(tmp_path: Path, corruption: str) -> None:
    path = tmp_path / "trajectory.jsonl"
    _trajectory(path)
    teacher = ScriptedMibePolicy()
    cold = prepare_imitation_replay((path,), teacher=teacher, cache_directory=tmp_path)
    key = imitation_replay_cache_key((path,), teacher=teacher)
    cache_path = tmp_path / f"{key}.npz"
    if corruption == "bytes":
        cache_path.write_bytes(b"PK\x03\x04broken")
    elif corruption == "npy":
        with cache_path.open("wb") as file:
            np.save(file, cold.features)
    else:
        features = cold.features.copy()
        masks = cold.masks.copy()
        if corruption == "scalar":
            features = np.asarray(0, dtype=np.float32)
        elif corruption == "illegal":
            masks[:] = False
        elif corruption == "nan":
            features[0, 0] = np.nan
        np.savez(
            cache_path,
            key=np.asarray("other" if corruption == "wrong-key" else key),
            features=features,
            masks=masks,
            teacher_actions=cold.teacher_actions,
        )
    rebuilt = prepare_imitation_replay(
        (path,), teacher=teacher, cache_directory=tmp_path
    )
    assert not rebuilt.cache_hit
    np.testing.assert_array_equal(rebuilt.features, cold.features)


def test_cache_unavailable_does_not_block_preparation(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.jsonl"
    _trajectory(path)
    blocked = tmp_path / "ordinary-file"
    blocked.write_text("not a directory")
    replay = prepare_imitation_replay(
        (path,), teacher=ScriptedMibePolicy(), cache_directory=blocked
    )
    assert not replay.cache_hit
    assert len(replay.teacher_actions) == 2


def test_cache_key_preserves_order_and_multiplicity(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _trajectory(first)
    second.write_text(first.read_text().replace("\n", " \n", 1))
    teacher = ScriptedMibePolicy()
    keys = {
        imitation_replay_cache_key(paths, teacher=teacher)
        for paths in ((first,), (first, first), (first, second), (second, first))
    }
    assert len(keys) == 4
