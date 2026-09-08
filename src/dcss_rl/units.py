"""Semantic scalar types used at domain, protocol, and scheduling boundaries."""

from typing import NewType

Seconds = NewType("Seconds", float)
"""A duration measured in seconds."""

GameSeed = NewType("GameSeed", int)
ActionIndex = NewType("ActionIndex", int)
Keycode = NewType("Keycode", int)
StepLimit = NewType("StepLimit", int)
WorkerCount = NewType("WorkerCount", int)
WorkerIndex = NewType("WorkerIndex", int)
EpisodeIndex = NewType("EpisodeIndex", int)
CaseCount = NewType("CaseCount", int)
CaseIndex = NewType("CaseIndex", int)
FrameLimit = NewType("FrameLimit", int)
ViewRadius = NewType("ViewRadius", int)
DcssVersion = NewType("DcssVersion", str)
VisibleCellCount = NewType("VisibleCellCount", int)
EpochCount = NewType("EpochCount", int)
BatchSize = NewType("BatchSize", int)
InferenceBatchSize = NewType("InferenceBatchSize", int)
InferenceBatchCount = NewType("InferenceBatchCount", int)
MeanInferenceBatchSize = NewType("MeanInferenceBatchSize", float)
FeatureCount = NewType("FeatureCount", int)
ActionCount = NewType("ActionCount", int)
LearningRate = NewType("LearningRate", float)
LossWeight = NewType("LossWeight", float)
Probability = NewType("Probability", float)
RewardWeight = NewType("RewardWeight", float)
CheckpointId = NewType("CheckpointId", str)
DecisionsPerSecond = NewType("DecisionsPerSecond", float)
DecisionProgressArea = NewType("DecisionProgressArea", int)
DepthWeightedDiscovery = NewType("DepthWeightedDiscovery", int)
LevelCount = NewType("LevelCount", int)
PlaceId = NewType("PlaceId", str)
type LevelId = tuple[PlaceId, int]
UpdateCount = NewType("UpdateCount", int)
RolloutLength = NewType("RolloutLength", int)
ActionHistoryLength = NewType("ActionHistoryLength", int)
UnixSocketPathBytes = NewType("UnixSocketPathBytes", int)

type Coordinate = tuple[int, int]
