"""Offline semantic behavior-cloning trainer with value and ECHO auxiliaries."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from dcss_rl.env import ACTION_COUNT
from dcss_rl.features import FeatureVector, encode_observation
from dcss_rl.learned import (
    CheckpointTrainingMetadata,
    ModelConfig,
    SemanticActorCritic,
    save_checkpoint,
)
from dcss_rl.policy import Policy, ScriptedMibePolicy
from dcss_rl.schema import ObservationData, ObservationDeltaData
from dcss_rl.trajectory import apply_observation_delta
from dcss_rl.units import BatchSize, EpochCount, LearningRate, LossWeight

_DEFAULT_EPOCHS = EpochCount(20)
_DEFAULT_BATCH_SIZE = BatchSize(256)
_DEFAULT_LEARNING_RATE = LearningRate(3e-4)
_DEFAULT_ECHO_WEIGHT = LossWeight(0.1)
_DEFAULT_VALUE_WEIGHT = LossWeight(0.1)
type ActionVector = NDArray[np.int64]
type ActionMaskVector = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int = 1
    epochs: EpochCount = _DEFAULT_EPOCHS
    batch_size: BatchSize = _DEFAULT_BATCH_SIZE
    learning_rate: LearningRate = _DEFAULT_LEARNING_RATE
    echo_weight: LossWeight = _DEFAULT_ECHO_WEIGHT
    value_weight: LossWeight = _DEFAULT_VALUE_WEIGHT
    discount: float = 0.99
    hidden_size: int = 256
    device: str = "cuda"
    relabel_with_scripted: bool = False


@dataclass(frozen=True, slots=True)
class TrainingReport:
    sample_count: int
    trajectory_count: int
    validation_accuracy: float
    validation_policy_loss: float
    checkpoint: Path


@dataclass(frozen=True, slots=True)
class _EpisodeArrays:
    features: tuple[FeatureVector, ...]
    masks: tuple[ActionMaskVector, ...]
    actions: tuple[int, ...]
    next_deltas: tuple[FeatureVector, ...]
    returns: tuple[float, ...]


def train_imitation(
    trajectories: tuple[Path, ...],
    checkpoint: Path,
    *,
    config: TrainingConfig,
    policy_id: str,
) -> TrainingReport:
    """Fit a learned policy to visible trajectories and persist the best epoch."""
    if not trajectories:
        raise ValueError("training requires at least one trajectory")
    _seed_everything(config.seed)
    teacher: Policy | None = (
        ScriptedMibePolicy() if config.relabel_with_scripted else None
    )
    episodes = tuple(
        _load_episode(path, discount=config.discount, teacher=teacher)
        for path in trajectories
    )
    features = np.stack([item for episode in episodes for item in episode.features])
    masks = np.stack([item for episode in episodes for item in episode.masks])
    actions = np.asarray(
        [item for episode in episodes for item in episode.actions], dtype=np.int64
    )
    deltas = np.stack([item for episode in episodes for item in episode.next_deltas])
    returns = np.asarray(
        [item for episode in episodes for item in episode.returns], dtype=np.float32
    )
    if len(actions) < 2:
        raise ValueError("training requires at least two transitions")
    train_indices, validation_indices = _stratified_split(actions, config.seed)
    train = TensorDataset(
        torch.from_numpy(features[train_indices]),
        torch.from_numpy(masks[train_indices]),
        torch.from_numpy(actions[train_indices]),
        torch.from_numpy(deltas[train_indices]),
        torch.from_numpy(returns[train_indices]),
    )
    validation = TensorDataset(
        torch.from_numpy(features[validation_indices]),
        torch.from_numpy(masks[validation_indices]),
        torch.from_numpy(actions[validation_indices]),
        torch.from_numpy(deltas[validation_indices]),
        torch.from_numpy(returns[validation_indices]),
    )
    class_weights = _class_weights(actions[train_indices]).to(config.device)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        train,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    device = torch.device(config.device)
    model = SemanticActorCritic(
        ModelConfig(action_count=ACTION_COUNT, hidden_size=config.hidden_size)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    for _ in range(config.epochs):
        model.train()
        for batch in loader:
            batch_features, batch_masks, batch_actions, batch_deltas, batch_returns = (
                value.to(device) for value in batch
            )
            logits, values, predicted_deltas = model(batch_features)
            logits = logits.masked_fill(~batch_masks, -torch.inf)
            loss = (
                F.cross_entropy(logits, batch_actions, weight=class_weights)
                + config.value_weight * F.mse_loss(values, batch_returns)
                + config.echo_weight * F.smooth_l1_loss(predicted_deltas, batch_deltas)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    accuracy, policy_loss = _validate(model, validation, device)
    metadata = CheckpointTrainingMetadata(
        training_method="scripted-dagger" if teacher is not None else "imitation",
        seed=config.seed,
        epochs=config.epochs,
        sample_count=len(actions),
        trajectory_count=len(trajectories),
        learning_rate=config.learning_rate,
        echo_weight=config.echo_weight,
        value_weight=config.value_weight,
        validation_accuracy=accuracy,
    )
    save_checkpoint(
        checkpoint, model=model, policy_id=policy_id, training_metadata=metadata
    )
    return TrainingReport(
        len(actions), len(trajectories), accuracy, policy_loss, checkpoint
    )


def _load_episode(
    path: Path, *, discount: float, teacher: Policy | None = None
) -> _EpisodeArrays:
    lines = Path(path).read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"trajectory has no transitions: {path}")
    header = _object(lines[0])
    initial = header.get("initial")
    if not isinstance(initial, dict):
        raise ValueError(f"trajectory has no initial state: {path}")
    raw_observation = cast(dict[str, object], initial).get("observation")
    if not isinstance(raw_observation, dict):
        raise ValueError(f"trajectory has no initial observation: {path}")
    observation = cast(ObservationData, raw_observation)
    features: list[FeatureVector] = []
    masks: list[ActionMaskVector] = []
    actions: list[int] = []
    deltas: list[FeatureVector] = []
    rewards: list[float] = []
    for line in lines[1:]:
        record = _object(line)
        action = record.get("action_index")
        reward = record.get("reward")
        if not isinstance(action, int) or not isinstance(reward, (int, float)):
            raise ValueError(f"invalid transition target in {path}")
        full = record.get("observation")
        delta = record.get("observation_delta")
        if isinstance(full, dict):
            next_observation = cast(ObservationData, full)
        elif isinstance(delta, dict):
            next_observation = apply_observation_delta(
                observation, cast(ObservationDeltaData, delta)
            )
        else:
            raise ValueError(f"transition has no next observation: {path}")
        current_features = encode_observation(observation)
        next_features = encode_observation(next_observation)
        mask = _training_action_mask(observation)
        features.append(current_features)
        masks.append(mask)
        actions.append(
            int(teacher.select(observation, mask)) if teacher is not None else action
        )
        deltas.append(next_features - current_features)
        rewards.append(float(reward))
        observation = next_observation
    return _EpisodeArrays(
        tuple(features),
        tuple(masks),
        tuple(actions),
        tuple(deltas),
        tuple(_discounted_returns(rewards, discount)),
    )


def _discounted_returns(rewards: list[float], discount: float) -> list[float]:
    result = [0.0] * len(rewards)
    future = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        future = rewards[index] + discount * future
        result[index] = future
    return result


def _validate(
    model: SemanticActorCritic, dataset: TensorDataset, device: torch.device
) -> tuple[float, float]:
    model.eval()
    features, masks, actions, _, _ = (tensor.to(device) for tensor in dataset.tensors)
    with torch.inference_mode():
        logits, _, _ = model(features)
        logits = logits.masked_fill(~masks, -torch.inf)
        loss = F.cross_entropy(logits, actions)
        accuracy = (logits.argmax(dim=-1) == actions).float().mean()
    return float(accuracy.item()), float(loss.item())


def _stratified_split(
    actions: ActionVector, seed: int
) -> tuple[ActionVector, ActionVector]:
    generator = np.random.default_rng(seed)
    training: list[int] = []
    validation: list[int] = []
    for action in np.unique(actions):
        indices = np.flatnonzero(actions == action)
        generator.shuffle(indices)
        validation_count = max(1, round(len(indices) * 0.1)) if len(indices) > 1 else 0
        validation.extend(int(index) for index in indices[:validation_count])
        training.extend(int(index) for index in indices[validation_count:])
    generator.shuffle(training)
    generator.shuffle(validation)
    return np.asarray(training, dtype=np.int64), np.asarray(validation, dtype=np.int64)


def _class_weights(actions: ActionVector) -> Tensor:
    counts = np.bincount(actions, minlength=ACTION_COUNT)
    weights = np.zeros(ACTION_COUNT, dtype=np.float32)
    present = counts > 0
    weights[present] = np.sqrt(len(actions) / counts[present])
    weights[present] /= weights[present].mean()
    return torch.from_numpy(weights)


def _training_action_mask(
    observation: ObservationData,
) -> ActionMaskVector:
    from dcss_rl.actions import Action, ActionKind
    from dcss_rl.env import action_to_index
    from dcss_rl.units import Keycode

    result = np.zeros(ACTION_COUNT, dtype=np.bool_)
    menu = observation["menu"]
    if menu is not None:
        for choice in menu["choices"]:
            index = action_to_index(Action.menu_select(Keycode(choice["keycode"])))
            result[index] = True
        result[action_to_index(Action(ActionKind.CANCEL))] = True
        return result
    if observation["input_mode"] == 1:
        for kind in ActionKind:
            if kind is not ActionKind.MENU_SELECT:
                result[action_to_index(Action(kind))] = True
    else:
        result[action_to_index(Action(ActionKind.CANCEL))] = True
    return result


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _object(text: str) -> dict[str, object]:
    decoded: object = json.loads(text)
    if not isinstance(decoded, dict):
        raise ValueError("trajectory record must be an object")
    return cast(dict[str, object], decoded)
