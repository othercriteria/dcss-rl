import json
from pathlib import Path

import pytest

from dcss_rl.actions import Action, ActionKind
from dcss_rl.env import DcssEnv, action_to_index
from dcss_rl.trajectory import RecordingEnv, TrajectoryWriter
from dcss_rl.units import GameSeed, StepLimit
from dcss_rl.webtiles import GameConfig

_DCSS_BINARY = Path("vendor/crawl/crawl-ref/source/crawl")


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
        env.step(action_to_index(Action(ActionKind.WAIT)))
    finally:
        env.close()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["type"] for record in records] == ["episode", "transition"]
    header, transition = records
    assert header["schema_version"] == 1
    assert header["metadata"]["seed"] == 11
    assert header["metadata"]["dcss_commit"]
    assert header["initial"]["setup_keycodes"] == [ord("c")]
    assert any(
        message["payload"].get("msg") == "player"
        for message in header["initial"]["raw_messages"]
    )
    assert transition["observation"]["player"]["species"] == "Minotaur"
    assert transition["emitted_keycodes"] == [ord(".")]
    assert [segment["loss"] for segment in transition["training_segments"]] == [
        None,
        "policy",
        "environment",
    ]
    assert json.loads(transition["training_segments"][0]["text"]) == observation
