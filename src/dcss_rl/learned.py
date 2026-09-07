"""Torch policy model and portable learned-policy checkpoint boundary."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from torch import Tensor, nn

from dcss_rl.features import FEATURE_COUNT, FEATURE_SPEC_VERSION, encode_observation
from dcss_rl.policy import Policy
from dcss_rl.schema import ObservationData
from dcss_rl.units import ActionIndex, CheckpointId, Probability

CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ModelConfig:
    action_count: int
    hidden_size: int = 256


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
    validation_accuracy: float


@dataclass(frozen=True, slots=True)
class PpoCheckpointMetadata:
    training_method: str
    seed: int
    updates: int
    rollout_steps: int
    worker_count: int
    learning_rate: float
    echo_weight: float
    value_weight: float
    imitation_weight: float
    explored_cell_reward: float
    experience_progress_reward: float
    hp_fraction_reward: float
    clip_ratio: float
    mean_episode_return: float


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
            nn.Linear(FEATURE_COUNT, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
        )
        self.policy_head = nn.Linear(config.hidden_size, config.action_count)
        self.value_head = nn.Linear(config.hidden_size, 1)
        self.echo_head = nn.Linear(config.hidden_size, FEATURE_COUNT)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
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
        if payload.get("feature_spec_version") != FEATURE_SPEC_VERSION:
            raise ValueError("checkpoint feature specification does not match")
        raw_config = payload.get("model_config")
        if not isinstance(raw_config, dict):
            raise ValueError("checkpoint lacks model configuration")
        config = ModelConfig(
            action_count=int(raw_config["action_count"]),
            hidden_size=int(raw_config["hidden_size"]),
        )
        self.model = SemanticActorCritic(config).to(device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        self.device = torch.device(device)
        identifier = payload.get("policy_id")
        if not isinstance(identifier, str):
            raise ValueError("checkpoint lacks policy ID")
        self.policy_id = identifier
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        self.checkpoint_id = CheckpointId(digest)

    def select(
        self, observation: ObservationData, action_mask: np.ndarray
    ) -> ActionIndex:
        return self.propose(observation, action_mask).action

    def propose(
        self, observation: ObservationData, action_mask: np.ndarray
    ) -> PolicyProposal:
        features = torch.from_numpy(encode_observation(observation)).to(self.device)
        mask = torch.from_numpy(action_mask).to(self.device)
        with torch.inference_mode():
            logits, _, _ = self.model(features.unsqueeze(0))
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
        self, observation: ObservationData, action_mask: np.ndarray
    ) -> ActionIndex:
        proposal = self.learned.propose(observation, action_mask)
        if proposal.confidence >= self.threshold:
            with self._lock:
                self._learned_decisions += 1
            return proposal.action
        with self._lock:
            self._fallback_decisions += 1
        return self.fallback.select(observation, action_mask)

    @property
    def learned_fraction(self) -> float:
        with self._lock:
            total = self._learned_decisions + self._fallback_decisions
            return self._learned_decisions / total if total else 0.0


def save_checkpoint(
    path: Path,
    *,
    model: SemanticActorCritic,
    policy_id: str,
    training_metadata: CheckpointTrainingMetadata | PpoCheckpointMetadata,
) -> None:
    """Persist weights plus every contract needed for deterministic restoration."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "feature_spec_version": FEATURE_SPEC_VERSION,
            "model_config": asdict(model.config),
            "model_state": model.state_dict(),
            "policy_id": policy_id,
            "training_metadata": asdict(training_metadata),
        },
        path,
    )
