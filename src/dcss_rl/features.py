"""Deterministic fixed-width features derived from player-visible semantic state."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from dcss_rl.schema import CellView, ObservationData
from dcss_rl.units import Coordinate, FeatureCount

FEATURE_SPEC_VERSION = 4
_V2_FEATURE_SPEC_VERSION = 2
_V3_FEATURE_SPEC_VERSION = 3
LOCAL_RADIUS = 5
_SIDE = 2 * LOCAL_RADIUS + 1
_MAP_CHANNELS = 9
_V2_SCALAR_FEATURES = 56
_V3_SCALAR_FEATURES = 64
_SCALAR_FEATURES = 66
FEATURE_COUNT = FeatureCount(_MAP_CHANNELS * _SIDE * _SIDE + _SCALAR_FEATURES)

type FeatureVector = NDArray[np.float32]

_ITEM_GLYPHS = frozenset(")([!?%$=:|/\\}")
_WALL_GLYPHS = frozenset({" ", "#", "≈", "♣"})


def feature_count(spec_version: int) -> FeatureCount:
    """Return the fixed width for a supported versioned feature contract."""
    if spec_version == _V2_FEATURE_SPEC_VERSION:
        return FeatureCount(_MAP_CHANNELS * _SIDE * _SIDE + _V2_SCALAR_FEATURES)
    if spec_version == _V3_FEATURE_SPEC_VERSION:
        return FeatureCount(_MAP_CHANNELS * _SIDE * _SIDE + _V3_SCALAR_FEATURES)
    if spec_version == FEATURE_SPEC_VERSION:
        return FEATURE_COUNT
    raise ValueError(f"unsupported feature specification {spec_version}")


def encode_observation(
    observation: ObservationData, *, spec_version: int = FEATURE_SPEC_VERSION
) -> FeatureVector:
    """Encode one semantic observation without learned or fitted preprocessing."""
    result = np.zeros(feature_count(spec_version), dtype=np.float32)
    player = observation["player"]
    position = player.get("pos", {"x": 0, "y": 0})
    origin_x, origin_y = position["x"], position["y"]
    for cell in observation["cells"]:
        local_x = cell["x"] - origin_x + LOCAL_RADIUS
        local_y = cell["y"] - origin_y + LOCAL_RADIUS
        if not 0 <= local_x < _SIDE or not 0 <= local_y < _SIDE:
            continue
        glyph = cell.get("g")
        channels = (
            glyph is not None,
            glyph is not None and glyph not in _WALL_GLYPHS,
            glyph == "#",
            "mon" in cell,
            glyph in _ITEM_GLYPHS,
            glyph == ">",
            glyph == "<",
            glyph == "+",
            glyph == "@",
        )
        for channel, active in enumerate(channels):
            if active:
                result[(channel * _SIDE + local_y) * _SIDE + local_x] = 1.0

    offset = _MAP_CHANNELS * _SIDE * _SIDE
    hp = _number(player.get("hp"))
    hp_max = max(_number(player.get("hp_max")), 1.0)
    mp = _number(player.get("mp"))
    mp_max = max(_number(player.get("mp_max")), 1.0)
    messages = " ".join(observation["messages"]).casefold()
    cells = {(cell["x"], cell["y"]): cell for cell in observation["cells"]}
    origin = (origin_x, origin_y)
    monster_targets = {point for point, cell in cells.items() if "mon" in cell}
    stair_targets = {point for point, cell in cells.items() if cell.get("g") == ">"}
    adjacent_monsters = tuple(
        float((origin_x + dx, origin_y + dy) in monster_targets)
        for dx, dy in _DIRECTIONS
    )
    adjacent_walkable = tuple(
        float(_walkable(cells.get((origin_x + dx, origin_y + dy))))
        for dx, dy in _DIRECTIONS
    )
    monster_step = _direction_one_hot(
        _shortest_step(origin, monster_targets, cells, stop_adjacent=True)
    )
    stair_step = _direction_one_hot(_shortest_step(origin, stair_targets, cells))
    scalar = (
        hp / hp_max,
        min(hp_max / 100.0, 1.0),
        mp / mp_max,
        min(mp_max / 30.0, 1.0),
        min(_number(player.get("xl")) / 27.0, 1.0),
        min(_number(player.get("depth")) / 27.0, 1.0),
        min(_number(player.get("ac")) / 40.0, 1.0),
        min(_number(player.get("ev")) / 40.0, 1.0),
        min(_number(player.get("sh")) / 40.0, 1.0),
        min(_number(player.get("str")) / 40.0, 1.0),
        min(_number(player.get("int")) / 40.0, 1.0),
        min(_number(player.get("dex")) / 40.0, 1.0),
        min(_number(player.get("gold")) / 1000.0, 1.0),
        min(math.log1p(_number(player.get("turn"))) / 10.0, 1.0),
        min(len(player.get("inv", {})) / 52.0, 1.0),
        min(len(player.get("status", [])) / 10.0, 1.0),
        float(observation["menu"] is not None),
        float(observation["input_mode"] == 1),
        float("done exploring" in messages),
        float(" is nearby" in messages),
        float("staircase leading down here" in messages),
        float(any("mon" in cell for cell in observation["cells"])),
        float(any(cell.get("g") == ">" for cell in observation["cells"])),
        1.0,
        *adjacent_monsters,
        *adjacent_walkable,
        *monster_step,
        *stair_step,
    )
    if spec_version >= 3:
        menu_type = (
            observation["menu"]["type"] if observation["menu"] is not None else None
        )
        scalar = (
            *scalar,
            float(menu_type == "shop"),
            float(menu_type == "more"),
            float(menu_type == "prompt"),
            float("done waiting" in messages),
            float("done exploring" in messages),
            float("lethal amount of poison" in messages),
            float("you are on fire" in messages),
            float("nearby" in messages),
        )
    if spec_version >= 4:
        statuses = " ".join(
            str(value).casefold()
            for status in player.get("status", [])
            for value in status.values()
        )
        scalar = (
            *scalar,
            float("berserk" in statuses),
            float("exhaust" in statuses),
        )
    result[offset:] = scalar
    return result


def _number(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


_DIRECTIONS = (
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
)


def _walkable(cell: CellView | None) -> bool:
    if cell is None:
        return False
    glyph = cell.get("g")
    return isinstance(glyph, str) and glyph not in _WALL_GLYPHS


def _shortest_step(
    start: Coordinate,
    targets: set[Coordinate],
    cells: Mapping[Coordinate, CellView],
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
            any(neighbor in targets for neighbor in _neighbors(current))
            if stop_adjacent
            else current in targets
        )
        if current != start and reached:
            destination = current
            break
        for candidate in _neighbors(current):
            if candidate in predecessors or (
                candidate not in targets and not _walkable(cells.get(candidate))
            ):
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
    return destination[0] - start[0], destination[1] - start[1]


def _neighbors(point: Coordinate) -> tuple[Coordinate, ...]:
    return tuple((point[0] + dx, point[1] + dy) for dx, dy in _DIRECTIONS)


def _direction_one_hot(direction: Coordinate | None) -> tuple[float, ...]:
    return tuple(float(direction == candidate) for candidate in _DIRECTIONS)
