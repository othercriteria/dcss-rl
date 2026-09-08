import json
from copy import deepcopy
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from dcss_rl.actions import Action, ActionKind
from dcss_rl.env import DcssEnv, action_to_index
from dcss_rl.replay import replay_frames
from dcss_rl.schema import EnvironmentInfo, ObservationData
from dcss_rl.trajectory import (
    CollectionProvenance,
    RecordingEnv,
    SamplingEvidence,
    SamplingStateHash,
    TrajectoryWriter,
    apply_observation_delta,
)
from dcss_rl.units import CheckpointId, GameSeed, Probability, StepLimit, UpdateCount
from dcss_rl.webtiles import GameConfig, Message, ObservationBatch

_DCSS_BINARY = Path("vendor/crawl/crawl-ref/source/crawl")


def test_optional_collection_provenance_preserves_raw_replay_across_updates(
    tmp_path: Path,
) -> None:
    mock = MagicMock(spec=DcssEnv)
    mock.game = MagicMock()
    mock.game.run_root = tmp_path
    mock.game.config = GameConfig(seed=GameSeed(3001))
    mock.game.static_cache = None
    mock.binary = _DCSS_BINARY
    mock.starting_weapon_key = "c"
    mock.last_keycodes = (ord("c"),)
    raw = Message({"msg": "player", "turn": 0})
    mock.last_exchange = (ObservationBatch((raw,), ()),)
    rc = tmp_path / "crawl.rc"
    rc.touch()
    env = cast(DcssEnv, mock)
    initial: ObservationData = {
        "player": {"turn": 0},
        "cells": [],
        "messages": [],
        "menu": None,
        "input_mode": 1,
    }
    path = tmp_path / "trajectory.jsonl"
    info: EnvironmentInfo = {
        "action_mask": np.ones(1, dtype=np.bool_),
        "structured_action": None,
        "emitted_keycodes": (),
        "outcome": None,
        "steps": 0,
        "max_depth": 1,
        "max_xl": 1,
    }
    with TrajectoryWriter(path) as writer:
        writer.start(env, initial, agent_id="pilot", checkpoint_id=None)
        previous = initial
        for update in (1, 2):
            current = deepcopy(initial)
            current["player"] = {"turn": update}
            mock.last_keycodes = (ord("."),)
            mock.last_exchange = (
                ObservationBatch((Message({"msg": "player", "turn": update}),), ()),
            )
            writer.transition(
                env,
                previous,
                action_to_index(Action(ActionKind.WAIT)),
                current,
                0.0,
                False,
                False,
                info,
                collection_provenance=CollectionProvenance(
                    UpdateCount(update), CheckpointId(str(update) * 64)
                ),
                sampling_evidence=SamplingEvidence(
                    (Probability(0.25), Probability(0.75)), SamplingStateHash("a" * 64)
                ),
            )
            previous = current
        writer.transition(
            env,
            previous,
            action_to_index(Action(ActionKind.WAIT)),
            previous,
            0.0,
            False,
            False,
            info,
        )
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0]["metadata"]["checkpoint_id"] is None
    assert records[0]["initial"]["raw_messages"][0]["payload"]["turn"] == 0
    for update in (1, 2):
        record = records[update]
        assert record["collection_provenance"] == {
            "update": update,
            "checkpoint_sha256": str(update) * 64,
        }
        assert record["raw_messages"][0]["payload"]["turn"] == update
        assert record["sampling_evidence"] == {
            "probabilities": [0.25, 0.75],
            "rng_state_sha256": "a" * 64,
        }
    assert "collection_provenance" not in records[-1]
    assert "sampling_evidence" not in records[-1]
    assert [frame.observation["player"]["turn"] for frame in replay_frames(path)] == [
        0,
        1,
        2,
        2,
    ]
    with pytest.raises(FileExistsError):
        TrajectoryWriter(path)


@pytest.mark.integration
@pytest.mark.skipif(
    not _DCSS_BINARY.is_file(), reason="local DCSS binary has not been built"
)
def test_records_replayable_raw_and_echo_segmented_trajectory(tmp_path: Path) -> None:
    path = tmp_path / "episode.jsonl"
    base = DcssEnv(
        _DCSS_BINARY,
        game_config=GameConfig(seed=GameSeed(11)),
        max_steps=StepLimit(1),
    )
    env = RecordingEnv(base, TrajectoryWriter(path), agent_id="test-agent")
    try:
        observation, _ = env.reset()
        next_observation, *_ = env.step(action_to_index(Action(ActionKind.WAIT)))
    finally:
        env.close()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["type"] for record in records] == ["episode", "transition"]
    header, transition = records
    assert header["schema_version"] == 2
    assert header["metadata"]["seed"] == 11
    assert header["metadata"]["dcss_commit"]
    assert header["initial"]["setup_keycodes"] == [ord("c")]
    assert any(
        message["payload"].get("msg") == "player"
        for message in header["initial"]["raw_messages"]
    )
    reconstructed = apply_observation_delta(
        observation, transition["observation_delta"]
    )
    assert reconstructed == next_observation
    assert transition["emitted_keycodes"] == [ord(".")]
    assert [segment["loss"] for segment in transition["training_segments"]] == [
        "policy",
        "environment",
    ]
    assert (
        json.loads(transition["training_segments"][1]["text"])
        == transition["observation_delta"]
    )
