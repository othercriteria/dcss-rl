from pathlib import Path

import numpy as np
import pytest

from dcss_rl.actions import Action, ActionKind
from dcss_rl.env import DcssEnv, action_to_index, index_to_action
from dcss_rl.webtiles import GameConfig


@pytest.mark.parametrize("kind", list(ActionKind))
def test_fixed_action_catalog_round_trips(kind: ActionKind) -> None:
    action = Action.menu_select(42) if kind is ActionKind.MENU_SELECT else Action(kind)
    assert index_to_action(action_to_index(action)) == action


@pytest.mark.integration
def test_gym_environment_resets_and_steps_real_trunk() -> None:
    binary = Path("vendor/crawl/crawl-ref/source/crawl")
    if not binary.is_file():
        pytest.skip("local DCSS binary has not been built")  # ty: ignore[too-many-positional-arguments]
    env = DcssEnv(binary, game_config=GameConfig(seed=7), max_steps=1)
    try:
        observation, info = env.reset()
        assert env.observation_space.contains(observation)
        assert observation["player"]["species"] == "Minotaur"
        assert info["action_mask"].dtype == np.bool_
        assert info["action_mask"][action_to_index(Action(ActionKind.WAIT))]

        next_observation, reward, terminated, truncated, step_info = env.step(
            action_to_index(Action(ActionKind.WAIT))
        )
        assert env.observation_space.contains(next_observation)
        assert isinstance(reward, float)
        assert not terminated
        assert truncated
        assert step_info["structured_action"] == {
            "kind": "wait",
            "keycode": None,
        }
    finally:
        env.close()
