from pathlib import Path

import numpy as np
import pytest

from dcss_rl.actions import Action, ActionKind
from dcss_rl.env import DcssEnv, action_to_index, index_to_action
from dcss_rl.units import GameSeed, Keycode, StepLimit
from dcss_rl.webtiles import GameConfig, Message, ObservationBatch

_DCSS_BINARY = Path("vendor/crawl/crawl-ref/source/crawl")


@pytest.mark.parametrize("kind", list(ActionKind))
def test_fixed_action_catalog_round_trips(kind: ActionKind) -> None:
    action = (
        Action.menu_select(Keycode(42))
        if kind is ActionKind.MENU_SELECT
        else Action(kind)
    )
    assert index_to_action(action_to_index(action)) == action


def test_player_visible_death_ends_episode_before_post_game_ui() -> None:
    batch = ObservationBatch(
        observations=(
            Message({"msg": "player", "hp": 0}),
            Message(
                {
                    "msg": "msgs",
                    "messages": [{"text": "<lightgrey>You die...", "turn": 7}],
                }
            ),
        ),
        controls=(),
    )

    assert DcssEnv._terminal_outcome(batch) == (True, "dead")


@pytest.mark.integration
@pytest.mark.skipif(
    not _DCSS_BINARY.is_file(), reason="local DCSS binary has not been built"
)
def test_gym_environment_resets_and_steps_real_trunk() -> None:
    env = DcssEnv(
        _DCSS_BINARY,
        game_config=GameConfig(seed=GameSeed(7)),
        max_steps=StepLimit(1),
    )
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
