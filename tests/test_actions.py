import pytest

from dcss_rl.actions import (
    Action,
    ActionKind,
    IllegalAction,
    encode_action,
    legal_actions,
)
from dcss_rl.env import ACTION_COUNT, action_to_index, index_to_action
from dcss_rl.observation import (
    MenuChoice,
    MenuChoiceApplicability,
    SemanticObservation,
)
from dcss_rl.units import Keycode


def observation(
    *,
    menu_type: str | None = None,
    choices: tuple[MenuChoice, ...] = (),
    messages: tuple[str, ...] = (),
    mode: int = 1,
) -> SemanticObservation:
    return SemanticObservation({}, (), messages, menu_type, None, choices, mode)


def test_command_mode_exposes_stable_structured_actions() -> None:
    available = legal_actions(observation())

    assert Action(ActionKind.MOVE_N) in available
    assert Action(ActionKind.EXPLORE) in available
    assert Action(ActionKind.ABILITIES) in available
    assert Action(ActionKind.CANCEL) not in available
    assert Action(ActionKind.STAIRS_DOWN) not in available
    assert encode_action(Action(ActionKind.MOVE_NW), observation()) == "y"


def test_ability_extension_preserves_every_legacy_menu_index() -> None:
    assert action_to_index(Action.menu_select(Keycode(0))) == 14
    assert action_to_index(Action.menu_select(Keycode(255))) == 269
    assert action_to_index(Action(ActionKind.ABILITIES)) == 270
    assert index_to_action(action_to_index(Action(ActionKind.ABILITIES))) == Action(
        ActionKind.ABILITIES
    )
    assert ACTION_COUNT == 271


def test_stairs_require_visible_under_player_affordance() -> None:
    downstairs = observation(
        messages=("There is a stone staircase leading down here.",)
    )
    upstairs = observation(messages=("There is a stone staircase leading up here.",))

    assert Action(ActionKind.STAIRS_DOWN) in legal_actions(downstairs)
    assert Action(ActionKind.STAIRS_UP) not in legal_actions(downstairs)
    assert Action(ActionKind.STAIRS_UP) in legal_actions(upstairs)
    assert Action(ActionKind.STAIRS_DOWN) not in legal_actions(upstairs)

    with pytest.raises(IllegalAction):
        encode_action(Action(ActionKind.STAIRS_DOWN), observation())

    hatch = observation(messages=("There is an escape hatch in the floor here.",))
    assert Action(ActionKind.STAIRS_DOWN) in legal_actions(hatch)


def test_menu_actions_are_derived_from_visible_choices() -> None:
    menu = observation(
        menu_type="newgame-choice",
        choices=(MenuChoice(Keycode(ord("c")), "c - hand axe"),),
    )

    assert legal_actions(menu) == (
        Action.menu_select(Keycode(ord("c"))),
        Action(ActionKind.CANCEL),
    )
    assert encode_action(Action.menu_select(Keycode(ord("c"))), menu) == ord("c")


def test_visible_inapplicability_does_not_become_a_tactical_mask() -> None:
    menu = observation(
        menu_type="ability",
        choices=(
            MenuChoice(
                Keycode(ord("a")),
                "a - Berserk",
                MenuChoiceApplicability.INAPPLICABLE,
            ),
        ),
    )

    assert Action.menu_select(Keycode(ord("a"))) in legal_actions(menu)


def test_action_mask_fails_closed_for_unknown_input_mode() -> None:
    available = legal_actions(observation(mode=999))

    assert available == (Action(ActionKind.CANCEL),)
    with pytest.raises(IllegalAction):
        encode_action(Action(ActionKind.EXPLORE), observation(mode=999))


def test_rejects_menu_key_that_dcss_did_not_offer() -> None:
    menu = observation(
        menu_type="newgame-choice",
        choices=(MenuChoice(Keycode(ord("c")), "c - hand axe"),),
    )

    with pytest.raises(IllegalAction):
        encode_action(Action.menu_select(Keycode(ord("a"))), menu)
