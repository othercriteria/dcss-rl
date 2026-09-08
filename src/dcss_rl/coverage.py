"""Action targets and syntactically legal exposure in imitation replay."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from dcss_rl.units import ActionCount


@dataclass(frozen=True, slots=True)
class ReplayCoverage:
    """Catalog-ordered counts, including absent target classes."""

    samples: ActionCount
    targets: tuple[ActionCount, ...]
    legal_exposures: tuple[ActionCount, ...]


def replay_coverage(
    targets: NDArray[np.int64], masks: NDArray[np.bool_]
) -> ReplayCoverage:
    if masks.ndim != 2 or targets.shape != (len(masks),):
        raise ValueError("replay targets and masks must have matching sample rows")
    action_count = masks.shape[1]
    if np.any(targets < 0) or np.any(targets >= action_count):
        raise ValueError("replay target outside action catalog")
    if not np.all(masks[np.arange(len(targets)), targets]):
        raise ValueError("imitation replay contains syntactically illegal targets")
    return ReplayCoverage(
        ActionCount(len(targets)),
        tuple(
            ActionCount(int(value))
            for value in np.bincount(targets, minlength=action_count)
        ),
        tuple(ActionCount(int(value)) for value in masks.sum(axis=0)),
    )
