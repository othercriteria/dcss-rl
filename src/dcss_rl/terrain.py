"""Conservative player-visible terrain evidence for navigation features.

This is not action legality or tactical safety. ``mf`` is a minimap category,
not a universal terrain/passability enum. The category numbering below agrees in
upstream map-feature.h at trunk 96832895, 0.34.1 1eebc1a2 and 0.33.1 9cb173b2.
Their feature-data.h maps solid statues/walls to WALL, but also maps solid
endless lava to LAVA, open sea to DEEP_WATER and endless salt to FLOOR.
Consequently flight cannot turn those categories into positive passage evidence.
Unknown results require an explicit caller fallback; raw ``f`` values are not
interpreted without revision context.
"""

from enum import IntEnum, StrEnum

from dcss_rl.schema import CellView, PlayerView


class NavigationTerrainHint(StrEnum):
    """Terrain evidence only; UNKNOWN is not a positive passage assertion."""

    BLOCKED = "blocked"
    UNKNOWN = "unknown"


class _MapCategory(IntEnum):
    WALL = 2
    MAPPED_WALL = 4
    LAVA = 17


def visibly_flying(player: PlayerView) -> bool:
    """Recognize the persistent upstream airborne status, not descriptive prose.

    status.cc::_describe_airborne emits light=Fly and short_text=flying on all
    three inspected revisions. Exact fields avoid matching hypothetical flight
    in another status description. Expiring flight still means currently flying.
    """
    return any(
        status.get("light") == "Fly" or status.get("text") == "flying"
        for status in player.get("status", [])
    )


def navigation_terrain_hint(
    cell: CellView | None, *, flying: bool
) -> NavigationTerrainHint:
    """Identify known obstruction without treating occupants as terrain.

    Known walls override their display glyph (e.g. a statue or cloud). Grounded
    lava is excluded for the MiBe curriculum; other species/forms could require
    additional locomotion evidence. Deep water is deliberately not excluded:
    swimming/water-walking is not represented by the flight argument. Door,
    floor, item, monster and unknown categories defer to the caller's existing
    navigation fallback. They are never inferred from an arbitrary ``f`` value.

    Flying lava remains UNKNOWN: ordinary lava is traversable in flight, but
    solid endless lava shares its minimap category. A version-aware terrain
    adapter would be needed to distinguish both reliably beneath overlays.
    """
    if cell is None:
        return NavigationTerrainHint.UNKNOWN
    category = cell.get("mf")
    if category in {_MapCategory.WALL, _MapCategory.MAPPED_WALL}:
        return NavigationTerrainHint.BLOCKED
    if category == _MapCategory.LAVA and not flying:
        return NavigationTerrainHint.BLOCKED
    return NavigationTerrainHint.UNKNOWN
