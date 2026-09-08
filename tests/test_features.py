import numpy as np
import pytest

from dcss_rl.features import (
    FEATURE_COUNT,
    FEATURE_SPEC_VERSION,
    encode_observation,
    feature_count,
)
from dcss_rl.schema import CellView, ObservationData
from dcss_rl.units import FeatureSpecVersion


def state(*, offset: int = 0, downstairs: bool = True) -> ObservationData:
    cells: list[CellView] = [
        {"x": offset, "y": offset, "g": "@"},
        {"x": offset + 1, "y": offset, "g": ".", "mf": 1},
    ]
    if downstairs:
        cells.append({"x": offset + 2, "y": offset, "g": ">", "mf": 44})
    return {
        "player": {
            "pos": {"x": offset, "y": offset},
            "hp": 12,
            "hp_max": 20,
            "xl": 2,
            "depth": 1,
        },
        "cells": cells,
        "messages": [],
        "menu": None,
        "input_mode": 1,
    }


def test_feature_encoding_is_fixed_width_and_translation_invariant() -> None:
    first = encode_observation(state())
    translated = encode_observation(state(offset=17))

    assert FEATURE_SPEC_VERSION == 6
    assert first.shape == (FEATURE_COUNT,)
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, translated)


def test_feature_v3_extends_v2_without_changing_legacy_values() -> None:
    observation = state()
    observation["messages"] = ["Done waiting."]
    legacy = encode_observation(observation, spec_version=FeatureSpecVersion(2))
    current = encode_observation(observation, spec_version=FeatureSpecVersion(3))

    assert legacy.shape == (feature_count(FeatureSpecVersion(2)),)
    np.testing.assert_array_equal(current[: len(legacy)], legacy)
    assert current[-5] == 1.0


def test_feature_v4_distinguishes_berserk_and_exhaustion() -> None:
    observation = state()
    observation["player"]["status"] = [
        {"light": "Berserk", "text": "berserking"},
        {"light": "Exhausted", "text": "recovering"},
    ]

    legacy = encode_observation(observation, spec_version=FeatureSpecVersion(3))
    current = encode_observation(observation, spec_version=FeatureSpecVersion(4))

    np.testing.assert_array_equal(current[: len(legacy)], legacy)
    np.testing.assert_array_equal(current[-2:], np.asarray([1.0, 1.0]))


def test_feature_v5_distinguishes_berserk_applicability_and_outcome() -> None:
    observation = state()
    observation["messages"] = ["You are too berserk!"]
    observation["menu"] = {
        "type": "ability",
        "prompt": "Ability - do what?",
        "choices": [
            {
                "keycode": ord("a"),
                "text": "a - Berserk",
                "applicability": "inapplicable",
            }
        ],
    }

    legacy = encode_observation(observation, spec_version=FeatureSpecVersion(4))
    current = encode_observation(observation)

    np.testing.assert_array_equal(current[: len(legacy)], legacy)
    np.testing.assert_array_equal(
        current[-13:],
        np.asarray(
            [
                0.0,
                0.0,
                1.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
    )


def test_feature_v5_does_not_alias_berserk_cooldown_with_active() -> None:
    observation = state()
    observation["player"]["status"] = [
        {
            "light": "-Berserk",
            "text": "on berserk cooldown",
            "desc": "You are recovering from your berserk rage.",
        }
    ]

    legacy = encode_observation(observation, spec_version=FeatureSpecVersion(4))
    current = encode_observation(observation)

    assert legacy[-2] == 1.0
    np.testing.assert_array_equal(current[-13:-11], np.asarray([0.0, 1.0]))


def test_global_navigation_summary_distinguishes_known_downstairs() -> None:
    with_stairs = encode_observation(state())
    without_stairs = encode_observation(state(downstairs=False))

    assert not np.array_equal(with_stairs, without_stairs)


@pytest.mark.parametrize("version", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("duplicate", [False, True])
def test_global_presence_preserves_raw_cell_semantics(
    version: int, duplicate: bool
) -> None:
    observation = state(downstairs=False)
    # The monster/stair can be outside the encoded local map, and duplicate raw
    # coordinates must retain presence even if the navigation map overwrites them.
    observation["cells"].append({"x": 100, "y": 100, "g": ">", "mon": {}})
    if duplicate:
        observation["cells"].append({"x": 100, "y": 100, "g": "."})
    encoded = encode_observation(observation, spec_version=FeatureSpecVersion(version))
    scalars = encoded[9 * 11 * 11 :]
    np.testing.assert_array_equal(scalars[21:23], np.asarray([1.0, 1.0]))


@pytest.mark.parametrize("category,glyph", [(2, "ß"), (4, "§"), (17, "§")])
def test_v6_rejects_structured_blockers_in_all_navigation_hints(
    category: int, glyph: str
) -> None:
    observation = state()
    observation["cells"][1].update({"g": glyph, "mf": category})
    observation["cells"].append({"x": 3, "y": 0, "g": "g", "mon": {}})
    legacy = encode_observation(observation, spec_version=FeatureSpecVersion(5))
    current = encode_observation(observation, spec_version=FeatureSpecVersion(6))
    scalar_offset = 9 * 11 * 11
    # Local passage at east, adjacent passage east, monster-route east,
    # stair-route east. The legacy encoder treats both glyphs as traversable.
    changed = [11 * 11 + 5 * 11 + 6, *(scalar_offset + i for i in (34, 42, 50))]
    np.testing.assert_array_equal(legacy[changed], np.ones(4))
    np.testing.assert_array_equal(current[changed], np.zeros(4))
    remaining = np.ones(len(legacy), dtype=bool)
    remaining[changed] = False
    np.testing.assert_array_equal(legacy[remaining], current[remaining])


def test_v6_target_entry_exception_cannot_cross_known_solid_terrain() -> None:
    observation = state(downstairs=False)
    observation["cells"][1].update({"g": ">", "mf": 2})
    legacy = encode_observation(observation, spec_version=FeatureSpecVersion(5))
    current = encode_observation(observation)
    assert legacy[9 * 11 * 11 + 50] == 1
    assert current[9 * 11 * 11 + 50] == 0


@pytest.mark.parametrize("glyph,category", [(".", 1), ("+", 5)])
def test_v6_preserves_floor_and_door_encoding(glyph: str, category: int) -> None:
    observation = state()
    observation["cells"][1].update({"g": glyph, "mf": category})
    np.testing.assert_array_equal(
        encode_observation(observation, spec_version=FeatureSpecVersion(5)),
        encode_observation(observation, spec_version=FeatureSpecVersion(6)),
    )
    assert feature_count(FeatureSpecVersion(5)) == feature_count(FeatureSpecVersion(6))


def test_v6_flight_removes_only_grounded_lava_objection() -> None:
    observation = state()
    observation["cells"][1].update({"g": "§", "mf": 17})
    observation["player"]["status"] = [{"text": "flying"}]
    np.testing.assert_array_equal(
        encode_observation(observation, spec_version=FeatureSpecVersion(5)),
        encode_observation(observation),
    )
    # Flight does not add a new positive terrain rule: the old glyph fallback
    # still rejects water glyphs, pending a version-aware terrain adapter.
    observation["cells"][1]["g"] = "≈"
    np.testing.assert_array_equal(
        encode_observation(observation, spec_version=FeatureSpecVersion(5)),
        encode_observation(observation),
    )
