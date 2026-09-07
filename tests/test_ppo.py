import pytest

from dcss_rl.schedule import TrainingSeedSchedule
from dcss_rl.units import CaseCount, EpisodeIndex, WorkerCount, WorkerIndex


def test_training_seed_schedule_allocates_disjoint_worker_blocks() -> None:
    schedule = TrainingSeedSchedule(CaseCount(64), WorkerCount(20))

    rounds = [
        {
            schedule.case_index(WorkerIndex(worker), EpisodeIndex(episode))
            for worker in range(20)
        }
        for episode in range(4)
    ]

    assert rounds[0] == set(range(0, 20))
    assert rounds[1] == set(range(20, 40))
    assert rounds[2] == set(range(40, 60))
    assert rounds[3] == {*range(60, 64), *range(0, 16)}


def test_training_seed_schedule_rejects_empty_suite() -> None:
    with pytest.raises(ValueError, match="counts must be positive"):
        TrainingSeedSchedule(CaseCount(0), WorkerCount(20))
