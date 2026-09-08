import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from dcss_rl.actions import Action, ActionKind
from dcss_rl.checkpoint_audit import audit_checkpoint, main
from dcss_rl.env import ACTION_COUNT, action_to_index
from dcss_rl.learned import ModelConfig, SemanticActorCritic


def _checkpoint(path: Path, *, change: str = "none") -> None:
    torch.manual_seed(4)
    model = SemanticActorCritic(ModelConfig(ACTION_COUNT, hidden_size=4))
    ability = action_to_index(Action(ActionKind.ABILITIES))
    with torch.no_grad():
        if change == "ability":
            model.policy_head.bias[ability] += 0.1
        elif change == "hidden":
            model.input_layer.weight[0, 0] += 0.1
        elif change == "cancel":
            model.policy_head.weight[action_to_index(Action(ActionKind.CANCEL)), 0] += (
                0.1
            )
    targets = [0] * ACTION_COUNT
    targets[ability] = 2
    legal = [2] * ACTION_COUNT
    torch.save(
        {
            "schema_version": 1,
            "feature_spec_version": model.config.feature_spec_version,
            "model_config": asdict(model.config),
            "model_state": model.state_dict(),
            "training_metadata": {
                "anchor_coverage": {
                    "samples": 2,
                    "targets": targets,
                    "legal_exposures": legal,
                }
            },
        },
        path,
    )


@pytest.mark.parametrize(
    "change,passes", [("ability", True), ("hidden", False), ("cancel", False)]
)
def test_ownership_allows_only_declared_policy_rows(
    tmp_path: Path,
    change: str,
    passes: bool,
) -> None:
    baseline, current = tmp_path / "baseline.pt", tmp_path / "current.pt"
    _checkpoint(baseline)
    _checkpoint(current, change=change)
    report = audit_checkpoint(
        current, baseline, allowed_rows=(action_to_index(Action(ActionKind.ABILITIES)),)
    )
    assert report.comparison is not None
    assert report.comparison.ownership_passed is passes
    counts = {item.action: item for item in report.coverage[0].actions}
    assert counts["abilities"].targets == 2
    assert counts["cancel"].targets == 0
    assert counts["cancel"].legal_exposures == 2
    informative = audit_checkpoint(current, baseline)
    assert informative.comparison is not None
    assert informative.comparison.ownership_passed is None


def test_shapes_and_configuration_are_not_silently_migrated(tmp_path: Path) -> None:
    baseline, current = tmp_path / "baseline.pt", tmp_path / "current.pt"
    _checkpoint(baseline)
    payload = torch.load(baseline, weights_only=True)
    payload["model_config"]["hidden_size"] = 5
    payload["model_state"]["value_head.weight"] = torch.zeros((1, 5))
    torch.save(payload, current)
    report = audit_checkpoint(current, baseline, allowed_rows=())
    assert report.comparison is not None
    assert report.comparison.ownership_passed is False
    assert len(report.comparison.incompatibilities) == 2


def test_cli_nonzero_failure_still_writes_reviewable_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline, current = tmp_path / "baseline.pt", tmp_path / "current.pt"
    output = tmp_path / "audit.json"
    _checkpoint(baseline)
    _checkpoint(current, change="hidden")
    monkeypatch.setattr(
        "sys.argv",
        [
            "checkpoint-audit",
            "--checkpoint",
            str(current),
            "--reference",
            str(baseline),
            "--allow-action-kind",
            "abilities",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    report = json.loads(output.read_text())
    assert report["comparison"]["changed_non_policy_tensors"] == ["encoder.0.weight"]
    assert json.loads(capsys.readouterr().out) == report
