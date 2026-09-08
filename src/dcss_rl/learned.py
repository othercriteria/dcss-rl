"""Torch policy model and portable learned-policy checkpoint boundary."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from torch import Tensor, nn

from dcss_rl.coverage import ReplayCoverage
from dcss_rl.env import ACTION_COUNT
from dcss_rl.features import FEATURE_SPEC_VERSION, encode_observation, feature_count
from dcss_rl.history import encode_action_history
from dcss_rl.policy import ActionHistory, Policy
from dcss_rl.schema import ObservationData
from dcss_rl.units import (
    ActionHistoryLength,
    ActionIndex,
    CheckpointId,
    FeatureSpecVersion,
    Probability,
)
from dcss_rl.webtiles.cache import StaticDataIdentity

CHECKPOINT_SCHEMA_VERSION = 1
_ZERO_ACTION_HISTORY_LENGTH = ActionHistoryLength(0)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    action_count: int
    hidden_size: int = 256
    feature_spec_version: FeatureSpecVersion = FEATURE_SPEC_VERSION
    action_history_length: ActionHistoryLength = _ZERO_ACTION_HISTORY_LENGTH


@dataclass(frozen=True, slots=True)
class CheckpointTrainingMetadata:
    training_method: str
    seed: int
    epochs: int
    sample_count: int
    trajectory_count: int
    learning_rate: float
    echo_weight: float
    value_weight: float
    teacher_balance_exponent: float
    validation_accuracy: float


@dataclass(frozen=True, slots=True)
class FeatureMigrationMetadata:
    training_method: str
    source_checkpoint_sha256: CheckpointId
    source_feature_spec: FeatureSpecVersion
    target_feature_spec: FeatureSpecVersion
    preprocessing_sha256: tuple[tuple[str, str], ...]
    optimizer_steps: int = 0


@dataclass(frozen=True, slots=True)
class PpoCheckpointMetadata:
    training_method: str
    seed: int
    updates: int
    rollout_steps: int
    worker_count: int
    inference_batch_size: int
    inference_batch_wait_seconds: float
    learning_rate: float
    echo_weight: float
    policy_weight: float
    value_weight: float
    imitation_weight: float
    aggregate_imitation_replay: bool
    teacher_balance_exponent: float
    explored_cell_reward: float
    depth_progress_reward: float
    experience_progress_reward: float
    hp_fraction_reward: float
    clip_ratio: float
    mean_episode_return: float
    action_history_length: int = 0
    new_action_warmup_updates: int = 0
    new_action_warmup_menu_keycodes: tuple[int, ...] = ()
    warmup_action_kinds: tuple[str, ...] = ()
    imitation_trajectories: tuple[str, ...] = ()
    return_boundary: str = "episodic"
    decision_cost: float = 0.0
    short_cycle_cost: float = 0.0
    short_cycle_window: int = 8
    ui_interaction_capacity: float = 2.0
    ui_interaction_refill_per_turn: float = 0.25
    ui_interaction_cost: float = 0.0
    ui_interaction_overflows: int = 0
    anchor_coverage: ReplayCoverage | None = None
    imitation_coverage: ReplayCoverage | None = None
    static_data_identity: StaticDataIdentity | None = None


@dataclass(frozen=True, slots=True)
class PolicyProposal:
    action: ActionIndex
    confidence: Probability


class SemanticActorCritic(nn.Module):
    """Shared semantic encoder with policy, value, and ECHO prediction heads."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(
                feature_count(config.feature_spec_version)
                + config.action_count * config.action_history_length,
                config.hidden_size,
            ),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
        )
        self.policy_head = nn.Linear(config.hidden_size, config.action_count)
        self.value_head = nn.Linear(config.hidden_size, 1)
        self.echo_head = nn.Linear(
            config.hidden_size, feature_count(config.feature_spec_version)
        )

    @property
    def input_layer(self) -> nn.Linear:
        """Return the typed semantic/history projection at the model boundary."""
        layer = self.encoder[0]
        if not isinstance(layer, nn.Linear):
            raise TypeError("actor-critic encoder must begin with a linear layer")
        return layer

    def forward(
        self, features: Tensor, action_history: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.config.action_history_length:
            if action_history is None:
                raise ValueError("model requires action-history features")
            features = torch.cat((features, action_history), dim=-1)
        encoded = self.encoder(features)
        return (
            self.policy_head(encoded),
            self.value_head(encoded).squeeze(-1),
            self.echo_head(encoded),
        )


class LearnedPolicy:
    """Deterministic masked policy restored from a versioned checkpoint."""

    def __init__(self, checkpoint: Path, *, device: str = "cpu") -> None:
        checkpoint = Path(checkpoint)
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("checkpoint must contain an object")
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported checkpoint schema")
        raw_feature_spec_version = payload.get("feature_spec_version")
        if not isinstance(raw_feature_spec_version, int):
            raise ValueError("checkpoint lacks a feature specification")
        feature_spec_version = FeatureSpecVersion(raw_feature_spec_version)
        feature_count(feature_spec_version)
        raw_config = payload.get("model_config")
        if not isinstance(raw_config, dict):
            raise ValueError("checkpoint lacks model configuration")
        config = ModelConfig(
            action_count=int(raw_config["action_count"]),
            hidden_size=int(raw_config["hidden_size"]),
            feature_spec_version=feature_spec_version,
            action_history_length=ActionHistoryLength(
                int(raw_config.get("action_history_length", 0))
            ),
        )
        self.checkpoint_action_count = config.action_count
        self.checkpoint_feature_spec_version = config.feature_spec_version
        self.model = SemanticActorCritic(config).to(device)
        self.model.load_state_dict(payload["model_state"])
        self.model = align_action_count(self.model, int(ACTION_COUNT))
        self.model.eval()
        self.device = torch.device(device)
        identifier = payload.get("policy_id")
        if not isinstance(identifier, str):
            raise ValueError("checkpoint lacks policy ID")
        self.policy_id = identifier
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        self.checkpoint_id = CheckpointId(digest)

    def select(
        self,
        observation: ObservationData,
        action_mask: np.ndarray,
        action_history: ActionHistory = (),
    ) -> ActionIndex:
        return self.propose(observation, action_mask, action_history).action

    def propose(
        self,
        observation: ObservationData,
        action_mask: np.ndarray,
        action_history: ActionHistory = (),
    ) -> PolicyProposal:
        features = torch.from_numpy(
            encode_observation(
                observation, spec_version=self.model.config.feature_spec_version
            )
        ).to(self.device)
        mask = torch.from_numpy(action_mask).to(self.device)
        history = torch.from_numpy(
            encode_action_history(
                action_history,
                action_count=self.model.config.action_count,
                length=self.model.config.action_history_length,
            )
        ).to(self.device)
        with torch.inference_mode():
            logits, _, _ = self.model(features.unsqueeze(0), history.unsqueeze(0))
            logits = logits.squeeze(0).masked_fill(~mask, -torch.inf)
            probabilities = torch.softmax(logits, dim=-1)
            action = ActionIndex(int(torch.argmax(probabilities).item()))
            confidence = Probability(float(probabilities[action].item()))
            return PolicyProposal(action, confidence)


class ConfidenceGatedPolicy:
    """Use learned decisions above a threshold and a visible-state expert otherwise."""

    def __init__(
        self,
        learned: LearnedPolicy,
        fallback: Policy,
        *,
        threshold: Probability,
    ) -> None:
        self.learned = learned
        self.fallback = fallback
        self.threshold = threshold
        self.policy_id = f"{learned.policy_id}-gated-{threshold:.3f}"
        self.checkpoint_id = learned.checkpoint_id
        self._lock = Lock()
        self._learned_decisions = 0
        self._fallback_decisions = 0

    def select(
        self,
        observation: ObservationData,
        action_mask: np.ndarray,
        action_history: ActionHistory = (),
    ) -> ActionIndex:
        proposal = self.learned.propose(observation, action_mask, action_history)
        if proposal.confidence >= self.threshold:
            with self._lock:
                self._learned_decisions += 1
            return proposal.action
        with self._lock:
            self._fallback_decisions += 1
        return self.fallback.select(observation, action_mask, action_history)

    @property
    def learned_fraction(self) -> float:
        with self._lock:
            total = self._learned_decisions + self._fallback_decisions
            return self._learned_decisions / total if total else 0.0


def add_action_history(
    model: SemanticActorCritic, length: ActionHistoryLength
) -> SemanticActorCritic:
    """Expand a stateless checkpoint while preserving its exact initial policy."""
    if model.config.action_history_length == length:
        return model
    if model.config.action_history_length:
        raise ValueError("cannot resize an existing action-history model")
    expanded = SemanticActorCritic(
        ModelConfig(
            action_count=model.config.action_count,
            hidden_size=model.config.hidden_size,
            feature_spec_version=model.config.feature_spec_version,
            action_history_length=length,
        )
    ).to(next(model.parameters()).device)
    old_state = model.state_dict()
    new_state = expanded.state_dict()
    for name, value in old_state.items():
        if name == "encoder.0.weight":
            new_state[name].zero_()
            new_state[name][:, : value.shape[1]] = value
        else:
            new_state[name] = value
    expanded.load_state_dict(new_state)
    return expanded


def align_feature_spec(
    model: SemanticActorCritic, feature_spec_version: FeatureSpecVersion
) -> SemanticActorCritic:
    """Migrate model coordinates; same-width versions may reinterpret input meaning.

    Older checkpoints keep their recorded preprocessing until explicitly migrated.
    Weight preservation alone does not imply equal behavior under new semantics.
    """
    if model.config.feature_spec_version == feature_spec_version:
        return model
    if model.config.feature_spec_version > feature_spec_version:
        raise ValueError("cannot downgrade a checkpoint feature specification")
    old_semantic_width = feature_count(model.config.feature_spec_version)
    new_semantic_width = feature_count(feature_spec_version)
    if old_semantic_width > new_semantic_width:
        raise ValueError("cannot shrink a checkpoint feature specification")
    expanded = SemanticActorCritic(
        ModelConfig(
            action_count=model.config.action_count,
            hidden_size=model.config.hidden_size,
            feature_spec_version=feature_spec_version,
            action_history_length=model.config.action_history_length,
        )
    ).to(next(model.parameters()).device)
    old = model.state_dict()
    new = expanded.state_dict()
    new["encoder.0.weight"].zero_()
    new["encoder.0.weight"][:, :old_semantic_width] = old["encoder.0.weight"][
        :, :old_semantic_width
    ]
    old_history_start = old_semantic_width
    new_history_start = new_semantic_width
    new["encoder.0.weight"][:, new_history_start:] = old["encoder.0.weight"][
        :, old_history_start:
    ]
    for name, value in old.items():
        if name == "encoder.0.weight":
            continue
        if name in {"echo_head.weight", "echo_head.bias"}:
            new[name].zero_()
            new[name][:old_semantic_width] = value
        else:
            new[name] = value
    expanded.load_state_dict(new)
    return expanded


def align_action_count(
    model: SemanticActorCritic, action_count: int
) -> SemanticActorCritic:
    """Append action outputs while preserving every established catalog index."""
    if model.config.action_count == action_count:
        return model
    if model.config.action_count > action_count:
        raise ValueError("cannot shrink a checkpoint action catalog")
    expanded = SemanticActorCritic(
        ModelConfig(
            action_count=action_count,
            hidden_size=model.config.hidden_size,
            feature_spec_version=model.config.feature_spec_version,
            action_history_length=model.config.action_history_length,
        )
    ).to(next(model.parameters()).device)
    old = model.state_dict()
    new = expanded.state_dict()
    semantic_width = feature_count(model.config.feature_spec_version)
    new["encoder.0.weight"].zero_()
    new["encoder.0.weight"][:, :semantic_width] = old["encoder.0.weight"][
        :, :semantic_width
    ]
    for slot in range(model.config.action_history_length):
        old_start = semantic_width + slot * model.config.action_count
        new_start = semantic_width + slot * action_count
        old_end = old_start + model.config.action_count
        new_end = new_start + model.config.action_count
        new["encoder.0.weight"][:, new_start:new_end] = old["encoder.0.weight"][
            :, old_start:old_end
        ]
    for name, value in old.items():
        if name == "encoder.0.weight":
            continue
        if name == "policy_head.weight":
            new[name][: model.config.action_count] = value
            new[name][model.config.action_count :].zero_()
        elif name == "policy_head.bias":
            new[name][: model.config.action_count] = value
            new[name][model.config.action_count :] = -10.0
        else:
            new[name] = value
    expanded.load_state_dict(new)
    return expanded


def save_checkpoint(
    path: Path,
    *,
    model: SemanticActorCritic,
    policy_id: str,
    training_metadata: CheckpointTrainingMetadata
    | PpoCheckpointMetadata
    | FeatureMigrationMetadata,
) -> None:
    """Persist weights plus every contract needed for deterministic restoration."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "feature_spec_version": model.config.feature_spec_version,
            "model_config": asdict(model.config),
            "model_state": model.state_dict(),
            "policy_id": policy_id,
            "training_metadata": asdict(training_metadata),
        },
        temporary_path,
    )
    temporary_path.replace(path)
