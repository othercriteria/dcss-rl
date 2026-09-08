import numpy as np

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

    assert FEATURE_SPEC_VERSION == 5
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
