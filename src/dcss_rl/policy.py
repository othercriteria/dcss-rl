"""Transparent deterministic policies for environment and evaluation baselines."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from dcss_rl.actions import Action, ActionKind, stairs_affordance_visible
from dcss_rl.env import action_to_index
from dcss_rl.schema import CellView, ObservationData
from dcss_rl.units import ActionIndex, CheckpointId, Coordinate, Keycode

type ActionHistory = tuple[ActionIndex, ...]


class Policy(Protocol):
    """An agent that selects one catalog action at an input boundary."""

    @property
    def policy_id(self) -> str: ...

    @property
    def checkpoint_id(self) -> CheckpointId | None: ...

    def select(
        self,
        observation: ObservationData,
        action_mask: np.ndarray,
        action_history: ActionHistory = (),
    ) -> ActionIndex: ...


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """A baseline choice with a stable explanation for debugging and replay."""

    action: Action
    reason: str


_DIRECTION_BY_DELTA = {
    (0, -1): ActionKind.MOVE_N,
    (1, -1): ActionKind.MOVE_NE,
    (1, 0): ActionKind.MOVE_E,
    (1, 1): ActionKind.MOVE_SE,
    (0, 1): ActionKind.MOVE_S,
    (-1, 1): ActionKind.MOVE_SW,
    (-1, 0): ActionKind.MOVE_W,
    (-1, -1): ActionKind.MOVE_NW,
}
_NEIGHBORS = tuple(_DIRECTION_BY_DELTA)
_IMPASSABLE_GLYPHS = frozenset({" ", "#", "≈", "♣"})
_IGNORED_STATIONARY_MONSTERS = frozenset(
    {"bush", "plant", "fungus", "withered plant", "demonic plant"}
)


class ScriptedMibePolicy:
    """Deterministic MiBe baseline using only the semantic player view."""

    policy_id = "scripted-mibe-v8"
    checkpoint_id = None

    def decide(self, observation: ObservationData) -> PolicyDecision:
        menu = observation["menu"]
        if menu is not None:
            choices = menu["choices"]
            berserk = next(
                (
                    item
                    for item in choices
                    if "berserk" in item["text"].casefold()
                    and item.get("applicability") != "inapplicable"
                ),
                None,
            )
            if menu["type"] == "ability" and berserk is not None:
                return PolicyDecision(
                    Action.menu_select(Keycode(berserk["keycode"])), "invoke Berserk"
                )
            strength = next(
                (item for item in choices if "strength" in item["text"].casefold()),
                None,
            )
            if strength is not None:
                return PolicyDecision(
                    Action.menu_select(Keycode(strength["keycode"])),
                    "increase strength",
                )
            if menu["type"] == "more" and choices:
                return PolicyDecision(
                    Action.menu_select(Keycode(choices[0]["keycode"])),
                    "continue message",
                )
            if menu["type"] == "prompt":
                rejection = next(
                    (
                        item
                        for item in choices
                        if "no" in item["text"].casefold()
                        or item["text"].casefold().startswith("n")
                    ),
                    None,
                )
                if rejection is not None:
                    return PolicyDecision(
                        Action.menu_select(Keycode(rejection["keycode"])),
                        "reject confirmation prompt",
                    )
            return PolicyDecision(
                Action(ActionKind.CANCEL), f"dismiss {menu['type']} menu"
            )

        position = _player_position(observation)
        cells = _cells_by_position(observation["cells"])
        monsters = {
            point for point, cell in cells.items() if _is_tactical_monster(cell)
        }
        hp = observation["player"].get("hp")
        hp_max = observation["player"].get("hp_max")
        statuses = " ".join(
            str(value).casefold()
            for status in observation["player"].get("status", [])
            for value in status.values()
        )

        if position is not None:
            adjacent = sorted(monsters & set(_adjacent(position)))
            if adjacent:
                should_berserk = (
                    len(monsters) >= 2
                    or observation["player"].get("depth", 1) >= 3
                    or (
                        isinstance(hp, int)
                        and isinstance(hp_max, int)
                        and hp * 4 <= hp_max * 3
                    )
                )
                if (
                    should_berserk
                    and "berserk" not in statuses
                    and "exhaust" not in statuses
                ):
                    return PolicyDecision(
                        Action(ActionKind.ABILITIES), "open abilities before melee"
                    )
                return PolicyDecision(
                    _move_toward(position, adjacent[0]), "attack adjacent monster"
                )

        if "berserk" in statuses and not monsters:
            return PolicyDecision(Action(ActionKind.WAIT), "wait out idle Berserk")

        messages = " ".join(observation["messages"]).casefold()
        if "lethal amount of poison" in messages or "you are on fire" in messages:
            return PolicyDecision(
                Action(ActionKind.WAIT), "advance blocking damage-over-time state"
            )

        if (
            not monsters
            and isinstance(hp, int)
            and isinstance(hp_max, int)
            and hp < hp_max
        ):
            return PolicyDecision(
                Action(ActionKind.REST), "recover with no visible threat"
            )

        if position is not None and monsters:
            step = _shortest_step(position, monsters, cells, stop_adjacent=True)
            if step is not None:
                return PolicyDecision(
                    _move_toward(position, step), "approach visible monster"
                )

        if stairs_affordance_visible(observation["messages"], ActionKind.STAIRS_DOWN):
            return PolicyDecision(
                Action(ActionKind.STAIRS_DOWN), "descend stairs under player"
            )

        if "nearby" in messages:
            return PolicyDecision(Action(ActionKind.WAIT), "wait for nearby threat")

        if position is not None:
            downstairs = {
                point for point, cell in cells.items() if cell.get("g") == ">"
            }
            if position in downstairs:
                return PolicyDecision(
                    Action(ActionKind.STAIRS_DOWN), "descend known stairs"
                )
            step = _shortest_step(position, downstairs, cells)
            if step is not None:
                return PolicyDecision(
                    _move_toward(position, step), "route to known stairs"
                )

        return PolicyDecision(Action(ActionKind.EXPLORE), "continue autoexploration")

    def select(
        self,
        observation: ObservationData,
        action_mask: np.ndarray,
        action_history: ActionHistory = (),
    ) -> ActionIndex:
        del action_history
        decision = self.decide(observation)
        index = action_to_index(decision.action)
        if index >= len(action_mask) or not bool(action_mask[index]):
            cancel = action_to_index(Action(ActionKind.CANCEL))
            if cancel < len(action_mask) and bool(action_mask[cancel]):
                return cancel
            legal = np.flatnonzero(action_mask)
            if len(legal) == 0:
                raise RuntimeError("environment supplied an empty action mask")
            return ActionIndex(int(legal[0]))
        return index


def _player_position(observation: ObservationData) -> Coordinate | None:
    position = observation["player"].get("pos")
    if position is None:
        return None
    return position["x"], position["y"]


def _cells_by_position(cells: list[CellView]) -> dict[Coordinate, CellView]:
    return {(cell["x"], cell["y"]): cell for cell in cells}


def _adjacent(point: Coordinate) -> tuple[Coordinate, ...]:
    x, y = point
    return tuple((x + dx, y + dy) for dx, dy in _NEIGHBORS)


def _move_toward(origin: Coordinate, destination: Coordinate) -> Action:
    delta = destination[0] - origin[0], destination[1] - origin[1]
    return Action(_DIRECTION_BY_DELTA[delta])


def _walkable(cell: CellView) -> bool:
    glyph = cell.get("g")
    map_feature = cell.get("mf")
    return (
        glyph is not None
        and glyph not in _IMPASSABLE_GLYPHS
        and map_feature in {None, 1}
    )


def _is_tactical_monster(cell: CellView) -> bool:
    monster = cell.get("mon")
    if not isinstance(monster, dict):
        return False
    name = monster.get("name")
    return not isinstance(name, str) or name not in _IGNORED_STATIONARY_MONSTERS


def _shortest_step(
    start: Coordinate,
    targets: set[Coordinate],
    cells: dict[Coordinate, CellView],
    *,
    stop_adjacent: bool = False,
) -> Coordinate | None:
    if not targets:
        return None
    queue = deque([start])
    predecessors: dict[Coordinate, Coordinate | None] = {start: None}
    destination: Coordinate | None = None
    while queue:
        current = queue.popleft()
        reached = (
            bool(set(_adjacent(current)) & targets)
            if stop_adjacent
            else current in targets
        )
        if current != start and reached:
            destination = current
            break
        for candidate in _adjacent(current):
            if candidate in predecessors:
                continue
            cell = cells.get(candidate)
            if cell is None or (candidate not in targets and not _walkable(cell)):
                continue
            predecessors[candidate] = current
            queue.append(candidate)
    if destination is None:
        return None
    while predecessors[destination] != start:
        previous = predecessors[destination]
        if previous is None:
            return None
        destination = previous
    return destination


def _exploration_complete(observation: ObservationData) -> bool:
    return any(
        "done exploring" in message.casefold() for message in observation["messages"]
    )
