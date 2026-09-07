"""Structured, serializable actions over DCSS's player-facing command interface."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from dcss_rl.observation import SemanticObservation
from dcss_rl.schema import ActionData
from dcss_rl.units import Keycode


class ActionKind(StrEnum):
    MOVE_N = "move_n"
    MOVE_NE = "move_ne"
    MOVE_E = "move_e"
    MOVE_SE = "move_se"
    MOVE_S = "move_s"
    MOVE_SW = "move_sw"
    MOVE_W = "move_w"
    MOVE_NW = "move_nw"
    WAIT = "wait"
    EXPLORE = "explore"
    REST = "rest"
    STAIRS_DOWN = "stairs_down"
    STAIRS_UP = "stairs_up"
    MENU_SELECT = "menu_select"
    CANCEL = "cancel"


_COMMAND_KEYS = {
    ActionKind.MOVE_N: "k",
    ActionKind.MOVE_NE: "u",
    ActionKind.MOVE_E: "l",
    ActionKind.MOVE_SE: "n",
    ActionKind.MOVE_S: "j",
    ActionKind.MOVE_SW: "b",
    ActionKind.MOVE_W: "h",
    ActionKind.MOVE_NW: "y",
    ActionKind.WAIT: ".",
    ActionKind.EXPLORE: "o",
    ActionKind.REST: "5",
    ActionKind.STAIRS_DOWN: ">",
    ActionKind.STAIRS_UP: "<",
    ActionKind.CANCEL: 27,
}


@dataclass(frozen=True, slots=True)
class Action:
    kind: ActionKind
    keycode: Keycode | None = None

    @classmethod
    def menu_select(cls, keycode: Keycode) -> Action:
        return cls(ActionKind.MENU_SELECT, keycode)

    def to_dict(self) -> ActionData:
        return {"kind": self.kind.value, "keycode": self.keycode}


class IllegalAction(ValueError):
    """Raised when an action is invalid for the current player-visible UI state."""


def legal_actions(observation: SemanticObservation) -> tuple[Action, ...]:
    """Return deterministic syntactic affordances for the current input boundary."""
    if observation.menu_type is not None:
        menu_actions = tuple(
            Action.menu_select(choice.keycode) for choice in observation.choices
        )
        return (*menu_actions, Action(ActionKind.CANCEL))

    # input_mode 1 is the ordinary command mode. Treat unknown mode as unsafe;
    # protocol changes should fail closed instead of emitting arbitrary keystrokes.
    if observation.input_mode != 1:
        return (Action(ActionKind.CANCEL),)

    return tuple(
        Action(kind)
        for kind in _COMMAND_KEYS
        if kind not in {ActionKind.STAIRS_DOWN, ActionKind.STAIRS_UP}
        or _stairs_affordance_visible(observation, kind)
    )


def _stairs_affordance_visible(
    observation: SemanticObservation, kind: ActionKind
) -> bool:
    direction = "down" if kind is ActionKind.STAIRS_DOWN else "up"
    phrase = f"staircase leading {direction} here"
    return any(phrase in message.casefold() for message in observation.messages)


def encode_action(action: Action, observation: SemanticObservation) -> str | int:
    """Validate and lower a structured action to one DCSS keycode."""
    if action not in legal_actions(observation):
        raise IllegalAction(f"action {action.to_dict()} is not legal in the current UI")
    if action.kind == ActionKind.MENU_SELECT:
        if action.keycode is None:
            raise IllegalAction("menu_select requires a keycode")
        return action.keycode
    return _COMMAND_KEYS[action.kind]
