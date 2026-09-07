"""Named serialization-boundary types used across environments and trajectories."""

from __future__ import annotations

from typing import Literal, NotRequired, Required, TypedDict

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


class Position(TypedDict):
    x: int
    y: int


class InventoryItem(TypedDict, total=False):
    quantity: int
    name: str
    letter: int
    base_type: int
    sub_type: int
    plus: int
    plus2: int
    flags: int
    inscription: str
    useless: int


class PlayerView(TypedDict, total=False):
    name: str
    title: str
    species: str
    god: str
    hp: int
    hp_max: int
    mp: int
    mp_max: int
    ac: int
    ev: int
    sh: int
    str: int
    int: int
    dex: int
    xl: int
    progress: int
    gold: int
    place: str
    depth: int
    pos: Position
    status: list[JsonObject]
    inv: dict[str, InventoryItem]
    weapon_index: int
    offhand_index: int
    quiver_item: int
    turn: int
    time: int
    piety_rank: int
    penance: int


class CellView(TypedDict, total=False):
    x: Required[int]
    y: Required[int]
    g: str
    col: int
    f: int
    mf: int
    mon: JsonObject


class MenuChoiceData(TypedDict):
    keycode: int
    text: str


class MenuView(TypedDict):
    type: str
    prompt: str | None
    choices: list[MenuChoiceData]


class ObservationData(TypedDict):
    player: PlayerView
    cells: list[CellView]
    messages: list[str]
    menu: MenuView | None
    input_mode: int | None


class ObservationDeltaData(TypedDict):
    player: PlayerView
    removed_player_fields: list[str]
    cells: list[CellView]
    removed_cells: list[Position]
    messages: list[str]
    menu_changed: bool
    menu: MenuView | None
    input_mode_changed: bool
    input_mode: int | None


class ActionData(TypedDict):
    kind: str
    keycode: int | None


class RawMessageData(TypedDict):
    control: bool
    payload: JsonObject


class LossSegmentData(TypedDict):
    role: Literal["environment", "action"]
    text: str
    loss: Literal["policy", "environment"] | None


class CharacterData(TypedDict):
    species: str
    background: str
    starting_weapon_key: str


class GymMetadata(TypedDict):
    render_modes: list[str]
    render_fps: NotRequired[int]
