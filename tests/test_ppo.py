import numpy as np
import pytest

from dcss_rl.returns import generalized_advantage_estimate
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


def test_gae_bootstraps_time_limit_without_crossing_episode_boundary() -> None:
    rewards = np.asarray([[0.0], [0.0], [100.0]], dtype=np.float32)
    values = np.asarray([[2.0], [3.0], [50.0]], dtype=np.float32)
    episode_ends = np.asarray([[False], [True], [False]], dtype=np.bool_)
    time_limit_bootstraps = np.asarray([[0.0], [7.0], [0.0]], dtype=np.float32)

    _, returns = generalized_advantage_estimate(
        rewards,
        values,
        episode_ends,
        time_limit_bootstraps,
        np.asarray([60.0], dtype=np.float32),
        discount=1.0,
        gae_lambda=1.0,
    )

    # The time-limited episode bootstraps V=7, while the next episode's reward 100
    # cannot leak backward across the reset boundary.
    np.testing.assert_array_equal(returns[:, 0], np.asarray([7.0, 7.0, 160.0]))
