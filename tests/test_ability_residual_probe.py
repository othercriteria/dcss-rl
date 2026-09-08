import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from dcss_rl.ability_residual_probe import (
    _A,
    _CANCEL,
    _X,
    MenuContext,
    ResidualData,
    base_tensors_exact,
    load_residual_data,
    measure_residual,
    outputs_exact,
    residual_step,
    run_residual_probe,
    selected_coverage,
)
from dcss_rl.env import ACTION_COUNT
from dcss_rl.features import FEATURE_COUNT, feature_count
from dcss_rl.learned import ModelConfig, SemanticActorCritic, enable_ability_residual
from dcss_rl.observation import MenuChoiceApplicability
from dcss_rl.schema import ObservationData
from dcss_rl.units import FeatureSpecVersion


def _data() -> ResidualData:
    features = torch.zeros(6, FEATURE_COUNT)
    offset = feature_count(FeatureSpecVersion(4))
    features[:4, offset + 2] = 1
    features[:2, offset + 3] = 1
    features[0, offset + 4] = 1
    features[1, offset + 5] = 1
    masks = torch.zeros(6, ACTION_COUNT, dtype=torch.bool)
    masks[:, [_A, _X, _CANCEL]] = True
    masks[2, _A] = False
    return ResidualData(
        features,
        masks,
        torch.tensor([_A, _CANCEL, _CANCEL, _CANCEL, _CANCEL, _CANCEL]),
        torch.tensor(list(MenuContext)),
        (),
    )


def _model() -> SemanticActorCritic:
    model = enable_ability_residual(
        SemanticActorCritic(ModelConfig(ACTION_COUNT, hidden_size=4))
    )
    with torch.no_grad():
        model.policy_head.weight.zero_()
        model.policy_head.bias.zero_()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("ability_residual_head."))
    return model


def test_three_context_step_preserves_base_and_other_contexts() -> None:
    model, data = _model(), _data()
    baseline = {
        name: tensor.clone()
        for name, tensor in model.state_dict().items()
        if not name.startswith("ability_residual_head.")
    }
    with torch.no_grad():
        logits = model(data.features)[0]
    before = measure_residual(model, data, logits)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=0.1
    )
    residual_step(model, data, optimizer)
    after = measure_residual(model, data, logits)
    assert base_tensors_exact(model, baseline)
    assert after.unknown_excluded == 1
    for old, new in zip(before.contexts, after.contexts, strict=True):
        assert new.count == 1
        assert new.target_probability_mean is not None
        assert old.target_probability_mean is not None
        assert new.target_probability_mean > old.target_probability_mean
    assert after.contexts[2].berserk_a_probability_maximum == 0
    assert all(
        group.logits_exact and group.probabilities_exact and group.argmax_exact
        for group in after.preservation
    )
    with torch.no_grad():
        model.value_head.bias.add_(1)
    assert not base_tensors_exact(model, baseline)


def test_balanced_ce_does_not_reweight_duplicated_context() -> None:
    data = _data()
    first, second = _model(), _model()
    second.load_state_dict(first.state_dict())
    duplicated = replace(
        data,
        **{
            name: torch.cat((getattr(data, name), getattr(data, name)[:1]))
            for name in ("features", "masks", "targets", "contexts")
        },
    )
    for model, batch in ((first, data), (second, duplicated)):
        residual_step(
            model,
            batch,
            torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1),
        )
    for name, tensor in first.state_dict().items():
        torch.testing.assert_close(tensor, second.state_dict()[name], atol=1e-7, rtol=0)


def test_missing_training_context_and_illegal_targets_fail() -> None:
    data, model = _data(), _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="all three"):
        residual_step(
            model, replace(data, contexts=torch.zeros_like(data.contexts)), optimizer
        )
    masks = data.masks.clone()
    masks[0, _A] = False
    with pytest.raises(ValueError, match="syntactically legal"):
        residual_step(model, replace(data, masks=masks), optimizer)


def test_loader_includes_missing_excludes_final_and_retains_unknown(
    tmp_path: Path,
) -> None:
    states: list[ObservationData] = []
    for applicability in (
        "applicable",
        "inapplicable",
        "missing",
        "unknown",
        "missing",
    ):
        state: ObservationData = {
            "player": {},
            "cells": [],
            "messages": [],
            "input_mode": 0,
            "menu": {"type": "ability", "prompt": None, "choices": []},
        }
        assert state["menu"] is not None
        state["menu"]["choices"].append({"keycode": 88, "text": "Renounce Religion"})
        if applicability != "missing":
            state["menu"]["choices"].append(
                {
                    "keycode": 97,
                    "text": "Berserk",
                    "applicability": MenuChoiceApplicability(applicability).value,
                }
            )
        states.append(state)
    path = tmp_path / "trajectory.jsonl"
    records = [
        json.dumps({"metadata": {"seed": 3049}, "initial": {"observation": states[0]}})
    ]
    records.extend(
        json.dumps({"step": i, "action": {"kind": "cancel"}, "observation": state})
        for i, state in enumerate(states[1:])
    )
    path.write_text("\n".join(records) + "\n")
    data = load_residual_data((path,))
    assert data.contexts.tolist() == [0, 1, 2, 3]
    assert data.targets.tolist() == [_A, _CANCEL, _CANCEL, _CANCEL]
    assert data.episodes[0].seed == 3049


def test_output_is_immutable_before_loading(tmp_path: Path) -> None:
    with pytest.raises(FileExistsError):
        run_residual_probe(tmp_path / "missing.pt", tmp_path, tmp_path)


def test_selected_coverage_includes_missing_target_not_unknown() -> None:
    coverage = selected_coverage(_data())
    assert coverage.samples == 3
    assert coverage.targets[_A] == 1
    assert coverage.targets[_CANCEL] == 2
    assert coverage.targets[_X] == 0
    assert coverage.legal_exposures[_A] == 2
    assert coverage.legal_exposures[_X] == coverage.legal_exposures[_CANCEL] == 3


def test_zero_identity_checks_all_three_outputs() -> None:
    source = SemanticActorCritic(ModelConfig(ACTION_COUNT, hidden_size=4))
    candidate = enable_ability_residual(source)
    data = _data()
    assert outputs_exact(source, candidate, data)
    with torch.no_grad():
        candidate.echo_head.bias.add_(1)
    assert not outputs_exact(source, candidate, data)
