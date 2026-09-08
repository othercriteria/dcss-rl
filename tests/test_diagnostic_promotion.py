from pathlib import Path

import msgspec
import pytest

from dcss_rl.diagnostic_promotion import promote_diagnostic
from dcss_rl.evaluation import EvaluationSummary


@pytest.mark.parametrize("suite_id", ["mibe-heldout-v5", "mibe-diagnostic-v2"])
def test_diagnostic_promotion_rejects_wrong_or_incomplete_suite(
    tmp_path: Path, suite_id: str
) -> None:
    path = tmp_path / "summary.json"
    path.write_bytes(
        msgspec.json.encode(EvaluationSummary(suite_id, "test", "now", ()))
    )
    with pytest.raises(ValueError, match="complete current diagnostic suite"):
        promote_diagnostic(path)
