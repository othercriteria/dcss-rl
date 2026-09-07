"""Version-tolerant semantic state derived from player-visible WebTiles deltas."""

from __future__ import annotations

import copy
import html
import re
from dataclasses import dataclass
from typing import Any

from dcss_rl.webtiles import ObservationBatch

_TAG = re.compile(r"<[^>]*>")


def plain_text(value: str) -> str:
    """Remove DCSS formatting tags while preserving human-visible text."""
    return html.unescape(_TAG.sub("", value)).strip()


def _merge(target: dict[str, Any], delta: dict[str, Any]) -> None:
    for key, value in delta.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


@dataclass(frozen=True, slots=True)
class MenuChoice:
    keycode: int
    text: str


@dataclass(frozen=True, slots=True)
class SemanticObservation:
    """A compact policy-facing snapshot at one DCSS input boundary."""

    player: dict[str, Any]
    cells: tuple[dict[str, Any], ...]
    messages: tuple[str, ...]
    menu_type: str | None
    prompt: str | None
    choices: tuple[MenuChoice, ...]
    input_mode: int | None

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, JSON-compatible representation for trajectories."""
        return {
            "player": self.player,
            "cells": list(self.cells),
            "messages": list(self.messages),
            "menu": {
                "type": self.menu_type,
                "prompt": self.prompt,
                "choices": [
                    {"keycode": choice.keycode, "text": choice.text}
                    for choice in self.choices
                ],
            }
            if self.menu_type is not None
            else None,
            "input_mode": self.input_mode,
        }


class ObservationReducer:
    """Apply sparse WebTiles updates and emit stable semantic snapshots."""

    PLAYER_FIELDS = (
        "name",
        "title",
        "species",
        "god",
        "hp",
        "hp_max",
        "mp",
        "mp_max",
        "ac",
        "ev",
        "sh",
        "str",
        "int",
        "dex",
        "xl",
        "progress",
        "gold",
        "place",
        "depth",
        "pos",
        "status",
        "inv",
        "weapon_index",
        "offhand_index",
        "quiver_item",
        "turn",
        "time",
        "piety_rank",
        "penance",
    )
    CELL_FIELDS = ("g", "col", "f", "mf", "mon")

    def __init__(self) -> None:
        self._player: dict[str, Any] = {}
        self._cells: dict[tuple[int, int], dict[str, Any]] = {}
        self._map_x: int | None = None
        self._map_y: int | None = None
        self._menu: dict[str, Any] | None = None
        self._input_mode: int | None = None

    def apply(self, batch: ObservationBatch) -> SemanticObservation:
        messages: list[str] = []
        for message in batch.observations:
            payload = message.payload
            if message.kind == "player":
                _merge(self._player, {k: v for k, v in payload.items() if k != "msg"})
            elif message.kind == "map":
                self._apply_map(payload)
            elif message.kind == "msgs":
                messages.extend(self._read_messages(payload))
            elif message.kind == "input_mode":
                mode = payload.get("mode")
                self._input_mode = mode if isinstance(mode, int) else None
            elif message.kind == "ui-push":
                self._menu = copy.deepcopy(payload)
            elif message.kind in {"ui-pop", "close_menu", "close_all_menus"}:
                self._menu = None

        return SemanticObservation(
            player=self._semantic_player(),
            cells=self._semantic_cells(),
            messages=tuple(messages),
            menu_type=self._menu_type(),
            prompt=self._prompt(),
            choices=self._choices(),
            input_mode=self._input_mode,
        )

    def _apply_map(self, payload: dict[str, Any]) -> None:
        if payload.get("clear") is True:
            self._cells.clear()
        cells = payload.get("cells")
        if not isinstance(cells, list):
            return
        for delta in cells:
            if not isinstance(delta, dict):
                continue
            x = delta.get("x")
            y = delta.get("y")
            if not isinstance(x, int):
                if self._map_x is None:
                    continue
                x = self._map_x + 1
            if not isinstance(y, int):
                if self._map_y is None:
                    continue
                y = self._map_y
            self._map_x, self._map_y = x, y
            cell = self._cells.setdefault((x, y), {"x": x, "y": y})
            _merge(cell, {k: v for k, v in delta.items() if k not in {"x", "y"}})

    @staticmethod
    def _read_messages(payload: dict[str, Any]) -> list[str]:
        result: list[str] = []
        values = payload.get("messages", [])
        if not isinstance(values, list):
            return result
        for value in values:
            if isinstance(value, dict) and isinstance(value.get("text"), str):
                result.append(plain_text(value["text"]))
        return result

    def _semantic_player(self) -> dict[str, Any]:
        result = {
            key: copy.deepcopy(self._player[key])
            for key in self.PLAYER_FIELDS
            if key in self._player
        }
        inventory = result.get("inv")
        if isinstance(inventory, dict):
            result["inv"] = {
                slot: item
                for slot, item in inventory.items()
                if isinstance(item, dict) and item.get("quantity", 0) > 0
            }
        return result

    def _semantic_cells(self) -> tuple[dict[str, Any], ...]:
        cells: list[dict[str, Any]] = []
        ordered = sorted(self._cells.items(), key=lambda item: item[0][::-1])
        for (x, y), source in ordered:
            cell = {"x": x, "y": y}
            cell.update(
                {
                    key: copy.deepcopy(source[key])
                    for key in self.CELL_FIELDS
                    if key in source and source[key] is not None
                }
            )
            cells.append(cell)
        return tuple(cells)

    def _menu_type(self) -> str | None:
        if self._menu is None:
            return None
        value = self._menu.get("type")
        return value if isinstance(value, str) else "unknown"

    def _prompt(self) -> str | None:
        if self._menu is None or not isinstance(self._menu.get("prompt"), str):
            return None
        return plain_text(self._menu["prompt"])

    def _choices(self) -> tuple[MenuChoice, ...]:
        if self._menu is None:
            return ()
        choices: list[MenuChoice] = []
        for section in ("main-items", "sub-items"):
            values = self._menu.get(section, {}).get("buttons", [])
            if not isinstance(values, list):
                continue
            for value in values:
                if not isinstance(value, dict) or not isinstance(
                    value.get("hotkey"), int
                ):
                    continue
                labels = value.get("labels")
                if isinstance(labels, list):
                    text = " ".join(plain_text(v) for v in labels if isinstance(v, str))
                else:
                    label = value.get("label", "")
                    text = plain_text(label) if isinstance(label, str) else ""
                choices.append(MenuChoice(value["hotkey"], text))
        return tuple(choices)
