from pathlib import Path

import pytest

from dcss_rl.checkpointing import update_checkpoint_path
from dcss_rl.units import UpdateCount


def test_update_checkpoint_paths_sort_in_training_order() -> None:
    directory = Path("checkpoints/run")

    assert update_checkpoint_path(directory, UpdateCount(12)) == Path(
        "checkpoints/run/update-0012.pt"
    )
    with pytest.raises(ValueError, match="positive"):
        update_checkpoint_path(directory, UpdateCount(0))
