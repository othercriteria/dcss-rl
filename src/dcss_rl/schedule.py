"""Deterministic scheduling primitives for parallel training."""

from dataclasses import dataclass

from dcss_rl.units import (
    CaseCount,
    CaseIndex,
    EpisodeIndex,
    WorkerCount,
    WorkerIndex,
)


@dataclass(frozen=True, slots=True)
class TrainingSeedSchedule:
    """Allocate disjoint worker blocks before wrapping a fixed seed suite."""

    case_count: CaseCount
    worker_count: WorkerCount

    def __post_init__(self) -> None:
        if self.case_count < 1 or self.worker_count < 1:
            raise ValueError("training seed schedule counts must be positive")

    def case_index(
        self, worker_index: WorkerIndex, episode_index: EpisodeIndex
    ) -> CaseIndex:
        return CaseIndex(
            (worker_index + episode_index * self.worker_count) % self.case_count
        )
