import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dcss_rl.ability_probe import (
    _A,
    _CANCEL,
    _ROWS,
    _V4_WIDTH,
    _X,
    ProbeData,
    measure,
    run_probe,
    selective_step,
    unowned_parameters_exact,
    validate_training_paths,
)
from dcss_rl.coverage import replay_coverage
from dcss_rl.env import ACTION_COUNT
from dcss_rl.features import FEATURE_COUNT
from dcss_rl.learned import ModelConfig, SemanticActorCritic


def _data() -> ProbeData:
    features = torch.ones((3, FEATURE_COUNT))
    masks = torch.zeros((3, ACTION_COUNT), dtype=torch.bool)
    masks[:, list(_ROWS)] = True
    targets = torch.tensor([_A, _CANCEL, _CANCEL])
    return ProbeData(
        features,
        masks,
        targets,
        torch.tensor([True, False, False]),
        torch.tensor([False, True, False]),
        torch.tensor([True, True, False]),
        replay_coverage(np.asarray(targets, dtype=np.int64), masks.numpy()),
        (),
        (),
    )


def test_decision_margin_includes_third_legal_competitor() -> None:
    model = SemanticActorCritic(ModelConfig(ACTION_COUNT, hidden_size=4))
    with torch.no_grad():
        model.policy_head.weight.zero_()
        model.policy_head.bias.zero_()
        model.policy_head.bias[_CANCEL] = 8
        model.policy_head.bias[_X] = 12
    data = _data()
    metrics = measure(model, data, model(data.features)[0].detach())
    assert metrics.inapplicable_margin == 8
    assert metrics.inapplicable_decision_margin == -4
    assert metrics.inapplicable_accuracy == 0
    assert metrics.inapplicable_target_probability < 0.02
    assert metrics.inapplicable_cross_entropy > 4


def test_selective_step_preserves_weights_but_can_change_unowned_outputs() -> None:
    torch.manual_seed(0)
    model = SemanticActorCritic(ModelConfig(ACTION_COUNT, hidden_size=4))
    data = _data()
    baseline = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    with torch.no_grad():
        baseline_logits = model(data.features)[0]
    columns = tuple(range(_V4_WIDTH, FEATURE_COUNT))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
    selective_step(model, data, optimizer, baseline, columns)
    assert unowned_parameters_exact(model, baseline, columns)
    assert measure(model, data, baseline_logits).nonmenu_max_unowned_logit_change > 0
    with torch.no_grad():
        model.value_head.bias.add_(0.001)
    assert not unowned_parameters_exact(model, baseline, columns)


def test_probe_refuses_existing_output_before_loading_inputs(tmp_path: Path) -> None:
    with pytest.raises(FileExistsError):
        run_probe(tmp_path / "missing.pt", tmp_path / "missing", tmp_path)


def test_probe_rejects_nontraining_and_duplicate_seeds(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(json.dumps({"metadata": {"seed": -1}}) + "\n")
    with pytest.raises(ValueError, match="outside current training suite"):
        validate_training_paths((first,))
    for path in (first, second):
        path.write_text(json.dumps({"metadata": {"seed": 3001}}) + "\n")
    with pytest.raises(ValueError, match="unique episode seeds"):
        validate_training_paths((first, second))
