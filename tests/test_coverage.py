import numpy as np
import pytest

from dcss_rl.coverage import replay_coverage


def test_coverage_preserves_absent_classes_and_nontarget_exposure() -> None:
    coverage = replay_coverage(
        np.array([0, 2, 0], dtype=np.int64),
        np.array([[1, 1, 0], [1, 1, 1], [1, 0, 1]], dtype=np.bool_),
    )
    assert coverage.samples == 3
    assert coverage.targets == (2, 0, 1)
    assert coverage.legal_exposures == (3, 2, 2)


def test_coverage_rejects_illegal_teacher_target() -> None:
    with pytest.raises(ValueError, match="illegal targets"):
        replay_coverage(
            np.array([1], dtype=np.int64), np.array([[1, 0]], dtype=np.bool_)
        )
