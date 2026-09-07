import pytest

from dcss_rl.actions import (
    Action,
    ActionKind,
    IllegalAction,
    encode_action,
    legal_actions,
)
from dcss_rl.observation import MenuChoice, SemanticObservation


def observation(
    *, menu_type: str | None = None, choices: tuple[MenuChoice, ...] = (), mode: int = 1
) -> SemanticObservation:
    return SemanticObservation({}, (), (), menu_type, None, choices, mode)


def test_command_mode_exposes_stable_structured_actions() -> None:
    available = legal_actions(observation())

    assert Action(ActionKind.MOVE_N) in available
    assert Action(ActionKind.EXPLORE) in available
    assert Action(ActionKind.STAIRS_DOWN) in available
    assert encode_action(Action(ActionKind.MOVE_NW), observation()) == "y"


def test_menu_actions_are_derived_from_visible_choices() -> None:
    menu = observation(
        menu_type="newgame-choice",
        choices=(MenuChoice(ord("c"), "c - hand axe"),),
    )

    assert legal_actions(menu) == (
        Action.menu_select(ord("c")),
        Action(ActionKind.CANCEL),
    )
    assert encode_action(Action.menu_select(ord("c")), menu) == ord("c")


def test_action_mask_fails_closed_for_unknown_input_mode() -> None:
    available = legal_actions(observation(mode=999))

    assert available == (Action(ActionKind.CANCEL),)
    with pytest.raises(IllegalAction):
        encode_action(Action(ActionKind.EXPLORE), observation(mode=999))


def test_rejects_menu_key_that_dcss_did_not_offer() -> None:
    menu = observation(
        menu_type="newgame-choice",
        choices=(MenuChoice(ord("c"), "c - hand axe"),),
    )

    with pytest.raises(IllegalAction):
        encode_action(Action.menu_select(ord("a")), menu)
