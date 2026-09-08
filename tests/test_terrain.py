import pytest

from dcss_rl.schema import CellView, PlayerView
from dcss_rl.terrain import (
    NavigationTerrainHint,
    navigation_terrain_hint,
    visibly_flying,
)


@pytest.mark.parametrize(
    "cell",
    [
        {"x": 9, "y": 42, "f": 30, "mf": 17, "g": "§"},
        {"x": -4, "y": 47, "f": 219, "mf": 2, "g": "ß"},
    ],
)
def test_development_blockers_survive_glyph_overlays(cell: CellView) -> None:
    assert navigation_terrain_hint(cell, flying=False) is NavigationTerrainHint.BLOCKED


def test_flight_does_not_remove_solid_wall_evidence() -> None:
    for category in (2, 4):
        cell: CellView = {"x": 0, "y": 0, "mf": category, "g": "§"}
        assert (
            navigation_terrain_hint(cell, flying=True) is NavigationTerrainHint.BLOCKED
        )


def test_flying_lava_needs_revision_aware_terrain_evidence() -> None:
    # Ordinary lava and solid endless lava share mf; even a cloud glyph cannot
    # tell them apart. Flight removes the grounded objection, not all blockers.
    cell: CellView = {"x": 0, "y": 0, "mf": 17, "g": "§"}
    assert navigation_terrain_hint(cell, flying=True) is NavigationTerrainHint.UNKNOWN


def test_floor_door_occupant_and_water_categories_defer_to_fallback() -> None:
    for category, glyph in ((1, "."), (5, "+"), (6, "!"), (10, "g"), (22, "≈")):
        cell: CellView = {"x": 0, "y": 0, "mf": category, "g": glyph}
        for flying in (False, True):
            assert (
                navigation_terrain_hint(cell, flying=flying)
                is NavigationTerrainHint.UNKNOWN
            )


def test_raw_terrain_numbers_and_missing_categories_are_not_guessed() -> None:
    cells: tuple[CellView | None, ...] = (
        None,
        {"x": 0, "y": 0, "f": 219, "g": "ß"},
    )
    for cell in cells:
        assert (
            navigation_terrain_hint(cell, flying=False) is NavigationTerrainHint.UNKNOWN
        )


def test_flight_comes_from_status_identity_not_description() -> None:
    bat: PlayerView = {
        "status": [{"text": "bat-form"}, {"light": "Fly", "text": "flying"}]
    }
    assert visibly_flying(bat)
    assert visibly_flying({"status": [{"text": "flying"}]})
    assert not visibly_flying({"status": [{"text": "fungus-form"}]})
    assert not visibly_flying({"status": [{"desc": "Flight would cross lava."}]})
    assert not visibly_flying({})
