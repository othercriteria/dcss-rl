"""Semantic scalar types used at domain, protocol, and scheduling boundaries."""

from typing import NewType

Seconds = NewType("Seconds", float)
"""A duration measured in seconds."""

GameSeed = NewType("GameSeed", int)
ActionIndex = NewType("ActionIndex", int)
Keycode = NewType("Keycode", int)
StepLimit = NewType("StepLimit", int)
WorkerCount = NewType("WorkerCount", int)
FrameLimit = NewType("FrameLimit", int)
ViewRadius = NewType("ViewRadius", int)
DcssVersion = NewType("DcssVersion", str)
VisibleCellCount = NewType("VisibleCellCount", int)
EpochCount = NewType("EpochCount", int)
BatchSize = NewType("BatchSize", int)
FeatureCount = NewType("FeatureCount", int)
ActionCount = NewType("ActionCount", int)
LearningRate = NewType("LearningRate", float)
LossWeight = NewType("LossWeight", float)

type Coordinate = tuple[int, int]
