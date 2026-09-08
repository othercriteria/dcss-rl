from pathlib import Path

import pytest
import torch

from dcss_rl.env import ACTION_COUNT
from dcss_rl.features import feature_count
from dcss_rl.learned import (
    _ABILITY_MENU_FLAG,
    _ABILITY_RESIDUAL_INPUTS,
    _ABILITY_RESIDUAL_ROWS,
    AbilityResidualVersion,
    CheckpointTrainingMetadata,
    LearnedPolicy,
    ModelConfig,
    SemanticActorCritic,
    add_action_history,
    align_action_count,
    align_feature_spec,
    enable_ability_residual,
    save_checkpoint,
)
from dcss_rl.units import ActionHistoryLength, FeatureSpecVersion


def _base(*, version: int = 6) -> SemanticActorCritic:
    return SemanticActorCritic(
        ModelConfig(
            int(ACTION_COUNT),
            hidden_size=4,
            feature_spec_version=FeatureSpecVersion(version),
        )
    )


def test_zero_residual_clones_all_base_tensors_and_outputs_exactly() -> None:
    base = _base()
    enabled = enable_ability_residual(base)
    assert base.config.ability_residual_version == 0
    assert base.ability_residual_head is None
    assert enabled.config.ability_residual_version == 1
    assert enabled.ability_residual_head is not None
    assert sum(p.numel() for p in enabled.ability_residual_head.parameters()) == 18
    for name, value in base.state_dict().items():
        assert torch.equal(value, enabled.state_dict()[name])
        assert value.data_ptr() != enabled.state_dict()[name].data_ptr()
    features = torch.randn(4, feature_count(FeatureSpecVersion(6)))
    features[:, _ABILITY_MENU_FLAG] = torch.tensor([0, 1, 0, 1])
    for left, right in zip(base(features), enabled(features), strict=True):
        assert torch.equal(left, right)


def test_learned_residual_changes_only_ability_menu_selected_logits() -> None:
    base = _base()
    enabled = enable_ability_residual(base)
    assert enabled.ability_residual_head is not None
    with torch.no_grad():
        enabled.ability_residual_head.weight.fill_(0.2)
        enabled.ability_residual_head.bias.fill_(0.3)
    features = torch.zeros(4, feature_count(FeatureSpecVersion(6)))
    features[:, list(_ABILITY_RESIDUAL_INPUTS)] = 1
    features[0, _ABILITY_MENU_FLAG] = 1
    # Other rows represent ordinary play, a shop, and a more prompt. Having
    # identical status/applicability values cannot enable the ability-only gate.
    scalar_offset = 9 * 11 * 11
    features[2:, scalar_offset + 16] = 1
    features[2, scalar_offset + 56] = 1
    features[3, scalar_offset + 57] = 1
    before = base(features)
    after = enabled(features)
    assert torch.equal(before[0][1:], after[0][1:])
    unchanged = torch.ones(int(ACTION_COUNT), dtype=torch.bool)
    unchanged[list(_ABILITY_RESIDUAL_ROWS)] = False
    assert torch.equal(before[0][0, unchanged], after[0][0, unchanged])
    assert not torch.equal(before[0][0], after[0][0])
    assert torch.equal(before[1], after[1])
    assert torch.equal(before[2], after[2])
    again = enable_ability_residual(enabled)
    for name, tensor in enabled.state_dict().items():
        assert torch.equal(tensor, again.state_dict()[name])


def test_residual_uses_semantic_prefix_and_survives_migrations() -> None:
    enabled = enable_ability_residual(_base(version=5))
    assert enabled.ability_residual_head is not None
    with torch.no_grad():
        enabled.ability_residual_head.weight.fill_(0.25)
    history_model = add_action_history(enabled, ActionHistoryLength(2))
    migrated = align_feature_spec(history_model, FeatureSpecVersion(6))
    expanded = align_action_count(migrated, int(ACTION_COUNT) + 1)
    for candidate in (history_model, migrated, expanded):
        assert candidate.config.ability_residual_version == 1
        assert candidate.ability_residual_head is not None
        assert torch.equal(
            candidate.ability_residual_head.weight, enabled.ability_residual_head.weight
        )
    features = torch.ones(2, feature_count(FeatureSpecVersion(5)))
    histories = torch.randn(2, int(ACTION_COUNT) * 2)
    # Newly added history weights are zero. Different history cannot replace
    # semantic flags at the residual's fixed input locations.
    for left, right in zip(
        enabled(features), history_model(features, histories), strict=True
    ):
        # Wider GEMM shapes may round differently despite identical base weights.
        torch.testing.assert_close(left, right)
    for left, right in zip(
        history_model(features, histories),
        history_model(features, torch.zeros_like(histories)),
        strict=True,
    ):
        assert torch.equal(left, right)
    with pytest.raises(ValueError, match="downgrade"):
        align_feature_spec(migrated, FeatureSpecVersion(4))
    with pytest.raises(ValueError, match="shrink"):
        align_action_count(enabled, int(ACTION_COUNT) - 1)


@pytest.mark.parametrize("residual", [False, True])
def test_residual_checkpoint_roundtrip_and_legacy_absent_field(
    tmp_path: Path,
    residual: bool,
) -> None:
    model = enable_ability_residual(_base()) if residual else _base()
    if model.ability_residual_head is not None:
        with torch.no_grad():
            model.ability_residual_head.bias.fill_(0.7)
    path = tmp_path / "model.pt"
    save_checkpoint(
        path,
        model=model,
        policy_id="test",
        training_metadata=CheckpointTrainingMetadata("test", 0, 0, 0, 0, 0, 0, 0, 1, 0),
    )
    if not residual:
        payload = torch.load(path, weights_only=True)
        del payload["model_config"]["ability_residual_version"]
        torch.save(payload, path)
    restored = LearnedPolicy(path).model
    assert restored.config == model.config
    for name, value in model.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value)


def test_residual_rejects_unknown_versions_and_incompatible_inputs() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        SemanticActorCritic(
            ModelConfig(
                int(ACTION_COUNT), ability_residual_version=AbilityResidualVersion(2)
            )
        )
    for config in (
        ModelConfig(270),
        ModelConfig(int(ACTION_COUNT), feature_spec_version=FeatureSpecVersion(4)),
    ):
        with pytest.raises(ValueError, match="requires"):
            enable_ability_residual(SemanticActorCritic(config))


def test_zero_residual_history_identity_and_trainable_gradient() -> None:
    base = add_action_history(_base(), ActionHistoryLength(1))
    enabled = enable_ability_residual(base)
    features = torch.ones(2, feature_count(FeatureSpecVersion(6)))
    histories = torch.randn(2, int(ACTION_COUNT))
    for left, right in zip(
        base(features, histories), enabled(features, histories), strict=True
    ):
        assert torch.equal(left, right)
    for name, parameter in enabled.named_parameters():
        parameter.requires_grad_(name.startswith("ability_residual_head."))
    enabled(features, histories)[0].sum().backward()
    assert enabled.ability_residual_head is not None
    assert enabled.ability_residual_head.weight.grad is not None
    assert torch.count_nonzero(enabled.ability_residual_head.weight.grad) == 15
