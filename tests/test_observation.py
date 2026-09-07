from dcss_rl.observation import ObservationReducer, plain_text
from dcss_rl.schema import JsonObject
from dcss_rl.webtiles import Message, ObservationBatch


def batch(*payloads: JsonObject) -> ObservationBatch:
    return ObservationBatch(tuple(Message(dict(value)) for value in payloads), ())


def test_plain_text_removes_dcss_markup() -> None:
    assert plain_text("<lightred>Trog says: <white>Fight!<lightgrey>") == (
        "Trog says: Fight!"
    )


def test_reducer_merges_sparse_player_and_map_updates() -> None:
    reducer = ObservationReducer()
    first = reducer.apply(
        batch(
            {"msg": "player", "hp": 19, "hp_max": 19, "depth": 1},
            {
                "msg": "map",
                "clear": True,
                "cells": [
                    {"x": -1, "y": 0, "g": "#", "f": 7},
                    {"g": "@", "f": 60},
                ],
            },
        )
    )
    second = reducer.apply(
        batch(
            {"msg": "player", "hp": 17},
            {"msg": "map", "cells": [{"x": 0, "y": 0, "g": "."}]},
        )
    )

    assert first.player == {"hp": 19, "hp_max": 19, "depth": 1}
    assert first.cells[1] == {"x": 0, "y": 0, "g": "@", "f": 60}
    assert second.player == {"hp": 17, "hp_max": 19, "depth": 1}
    assert second.cells[1] == {"x": 0, "y": 0, "g": ".", "f": 60}


def test_reducer_extracts_messages_and_menu_choices() -> None:
    observation = ObservationReducer().apply(
        batch(
            {
                "msg": "ui-push",
                "type": "newgame-choice",
                "prompt": "<cyan>Choose.",
                "main-items": {
                    "buttons": [{"hotkey": 99, "labels": ["c - hand axe", "(+2 apt)"]}]
                },
            },
            {"msg": "msgs", "messages": [{"text": "<yellow>Welcome!"}]},
        )
    )

    assert observation.menu_type == "newgame-choice"
    assert observation.prompt == "Choose."
    assert observation.choices[0].keycode == ord("c")
    assert observation.choices[0].text == "c - hand axe (+2 apt)"
    assert observation.messages == ("Welcome!",)


def test_reducer_omits_empty_inventory_slots() -> None:
    observation = ObservationReducer().apply(
        batch(
            {
                "msg": "player",
                "inv": {
                    "0": {"quantity": 1, "name": "+0 hand axe"},
                    "1": {"quantity": 0, "base_type": 100},
                },
            }
        )
    )

    assert observation.player["inv"] == {"0": {"quantity": 1, "name": "+0 hand axe"}}
