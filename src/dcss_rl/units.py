"""Semantic scalar types used at domain, protocol, and scheduling boundaries."""

from typing import NewType

Seconds = NewType("Seconds", float)
"""A duration measured in seconds."""

GameSeed = NewType("GameSeed", int)
ActionIndex = NewType("ActionIndex", int)
Keycode = NewType("Keycode", int)
StepLimit = NewType("StepLimit", int)
WorkerCount = NewType("WorkerCount", int)

type Coordinate = tuple[int, int]
