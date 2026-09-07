"""Concurrent online masked PPO fine-tuning for the semantic actor-critic."""

from __future__ import annotations

import random
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from numpy.typing import NDArray
from torch.distributions import Categorical
from torch.nn import functional as F

from dcss_rl.env import DcssEnv
from dcss_rl.evaluation import EvaluationSuite
from dcss_rl.features import encode_observation
from dcss_rl.learned import (
    LearnedPolicy,
    PpoCheckpointMetadata,
    SemanticActorCritic,
    save_checkpoint,
)
from dcss_rl.policy import ScriptedMibePolicy
from dcss_rl.schema import ObservationData
from dcss_rl.units import (
    ActionIndex,
    BatchSize,
    DecisionsPerSecond,
    EpochCount,
    GameSeed,
    LearningRate,
    LossWeight,
    Probability,
    RolloutLength,
    StepLimit,
    UpdateCount,
    WorkerCount,
)
from dcss_rl.webtiles import GameConfig

type FloatArray = NDArray[np.float32]
type BoolArray = NDArray[np.bool_]
type IntArray = NDArray[np.int64]

_DEFAULT_UPDATES = UpdateCount(4)
_DEFAULT_ROLLOUT_LENGTH = RolloutLength(128)
_DEFAULT_WORKERS = WorkerCount(5)
_DEFAULT_BATCH_SIZE = BatchSize(256)
_DEFAULT_LEARNING_RATE = LearningRate(1e-4)
_DEFAULT_ECHO_WEIGHT = LossWeight(0.1)
_DEFAULT_VALUE_WEIGHT = LossWeight(0.5)
_DEFAULT_ENTROPY_WEIGHT = LossWeight(0.01)
_DEFAULT_IMITATION_WEIGHT = LossWeight(0.1)
_DEFAULT_CLIP_RATIO = Probability(0.2)
_DEFAULT_DISCOUNT = Probability(0.99)
_DEFAULT_GAE_LAMBDA = Probability(0.95)
_DEFAULT_EPOCHS_PER_UPDATE = EpochCount(4)


@dataclass(frozen=True, slots=True)
class PpoConfig:
    seed: int = 1
    updates: UpdateCount = _DEFAULT_UPDATES
    rollout_length: RolloutLength = _DEFAULT_ROLLOUT_LENGTH
    workers: WorkerCount = _DEFAULT_WORKERS
    minibatch_size: BatchSize = _DEFAULT_BATCH_SIZE
    learning_rate: LearningRate = _DEFAULT_LEARNING_RATE
    echo_weight: LossWeight = _DEFAULT_ECHO_WEIGHT
    value_weight: LossWeight = _DEFAULT_VALUE_WEIGHT
    entropy_weight: LossWeight = _DEFAULT_ENTROPY_WEIGHT
    imitation_weight: LossWeight = _DEFAULT_IMITATION_WEIGHT
    clip_ratio: Probability = _DEFAULT_CLIP_RATIO
    discount: Probability = _DEFAULT_DISCOUNT
    gae_lambda: Probability = _DEFAULT_GAE_LAMBDA
    epochs_per_update: EpochCount = _DEFAULT_EPOCHS_PER_UPDATE
    device: str = "cuda"

    def __post_init__(self) -> None:
        positive_integers = (
            self.updates,
            self.rollout_length,
            self.workers,
            self.minibatch_size,
            self.epochs_per_update,
        )
        if any(value < 1 for value in positive_integers):
            raise ValueError("PPO counts must be positive")
        if self.learning_rate <= 0:
            raise ValueError("PPO learning rate must be positive")
        if not 0 < self.discount <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("PPO discount and GAE lambda must be probabilities")
        if not 0 < self.clip_ratio < 1:
            raise ValueError("PPO clip ratio must be between zero and one")
        if any(
            weight < 0
            for weight in (
                self.echo_weight,
                self.value_weight,
                self.entropy_weight,
                self.imitation_weight,
            )
        ):
            raise ValueError("PPO loss weights cannot be negative")


@dataclass(frozen=True, slots=True)
class PpoReport:
    checkpoint: Path
    decisions: int
    completed_episodes: int
    mean_episode_return: float
    policy_loss: float
    value_loss: float
    echo_loss: float


@dataclass(frozen=True, slots=True)
class PpoUpdateReport:
    update: UpdateCount
    decisions: int
    decision_rate: DecisionsPerSecond
    completed_episodes: int
    mean_completed_return: float
    policy_loss: float
    value_loss: float
    echo_loss: float


@dataclass(slots=True)
class _Worker:
    binary: Path
    suite: EvaluationSuite
    root: Path
    worker_index: int
    episode_index: int = 0
    env: DcssEnv | None = None
    observation: ObservationData | None = None
    action_mask: BoolArray | None = None
    episode_return: float = 0.0

    def ready(self) -> tuple[ObservationData, BoolArray]:
        if self.observation is None:
            self._reset()
        if self.observation is None or self.action_mask is None:
            raise RuntimeError("online worker failed to reset")
        return self.observation, self.action_mask

    def step(
        self, action: ActionIndex
    ) -> tuple[ObservationData, float, bool, float | None]:
        if self.env is None:
            raise RuntimeError("online worker is not ready")
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.episode_return += reward
        completed_return = self.episode_return if done else None
        if done:
            self.env.close()
            self.env = None
            self.observation = None
            self.action_mask = None
            self.episode_return = 0.0
        else:
            self.observation = observation
            mask = info.get("action_mask")
            if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
                raise RuntimeError("environment returned an invalid action mask")
            self.action_mask = mask
        return observation, reward, done, completed_return

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def _reset(self) -> None:
        case = self.suite.cases[
            (self.worker_index + self.episode_index * len(self.suite.cases))
            % len(self.suite.cases)
        ]
        run_root = (
            self.root
            / f"worker-{self.worker_index}"
            / (f"episode-{self.episode_index}-{case.case_id}")
        )
        self.episode_index += 1
        self.env = DcssEnv(
            self.binary,
            game_config=GameConfig(seed=GameSeed(case.seed)),
            max_steps=StepLimit(self.suite.step_limit),
            run_root=run_root,
        )
        observation, info = self.env.reset()
        mask = info.get("action_mask")
        if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
            raise RuntimeError("environment returned an invalid action mask")
        self.observation = observation
        self.action_mask = mask


@dataclass(frozen=True, slots=True)
class _Rollout:
    features: FloatArray
    masks: BoolArray
    actions: IntArray
    log_probabilities: FloatArray
    values: FloatArray
    advantages: FloatArray
    returns: FloatArray
    next_deltas: FloatArray
    teacher_actions: IntArray
    completed_returns: tuple[float, ...]


def train_ppo(
    binary: Path,
    suite: EvaluationSuite,
    initial_checkpoint: Path,
    output_checkpoint: Path,
    run_root: Path,
    *,
    config: PpoConfig,
    policy_id: str,
    progress: Callable[[PpoUpdateReport], None] | None = None,
) -> PpoReport:
    """Fine-tune an imitation checkpoint with concurrent on-policy PPO updates."""
    _seed_everything(config.seed)
    restored = LearnedPolicy(initial_checkpoint, device=config.device)
    model = restored.model
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    teacher = ScriptedMibePolicy()
    workers = tuple(
        _Worker(binary, suite, run_root, index) for index in range(config.workers)
    )
    completed_returns: list[float] = []
    policy_loss = value_loss = echo_loss = 0.0
    try:
        with ThreadPoolExecutor(max_workers=config.workers) as executor:
            for update_index in range(config.updates):
                update_started = perf_counter()
                rollout = _collect_rollout(
                    model, workers, executor, teacher, config=config
                )
                completed_returns.extend(rollout.completed_returns)
                policy_loss, value_loss, echo_loss = _ppo_update(
                    model, optimizer, rollout, config=config
                )
                if progress is not None:
                    update_decisions = int(config.rollout_length * config.workers)
                    update_returns = rollout.completed_returns
                    progress(
                        PpoUpdateReport(
                            UpdateCount(update_index + 1),
                            update_decisions,
                            DecisionsPerSecond(
                                update_decisions / (perf_counter() - update_started)
                            ),
                            len(update_returns),
                            float(np.mean(update_returns)) if update_returns else 0.0,
                            policy_loss,
                            value_loss,
                            echo_loss,
                        )
                    )
    finally:
        for worker in workers:
            worker.close()
    mean_return = float(np.mean(completed_returns)) if completed_returns else 0.0
    metadata = PpoCheckpointMetadata(
        training_method="masked-ppo",
        seed=config.seed,
        updates=config.updates,
        rollout_steps=config.rollout_length,
        worker_count=config.workers,
        learning_rate=config.learning_rate,
        echo_weight=config.echo_weight,
        value_weight=config.value_weight,
        imitation_weight=config.imitation_weight,
        clip_ratio=config.clip_ratio,
        mean_episode_return=mean_return,
    )
    save_checkpoint(
        output_checkpoint,
        model=model,
        policy_id=policy_id,
        training_metadata=metadata,
    )
    return PpoReport(
        output_checkpoint,
        int(config.updates * config.rollout_length * config.workers),
        len(completed_returns),
        mean_return,
        policy_loss,
        value_loss,
        echo_loss,
    )


def _collect_rollout(
    model: SemanticActorCritic,
    workers: tuple[_Worker, ...],
    executor: ThreadPoolExecutor,
    teacher: ScriptedMibePolicy,
    *,
    config: PpoConfig,
) -> _Rollout:
    device = torch.device(config.device)
    feature_steps: list[FloatArray] = []
    mask_steps: list[BoolArray] = []
    action_steps: list[IntArray] = []
    log_probability_steps: list[FloatArray] = []
    value_steps: list[FloatArray] = []
    reward_steps: list[FloatArray] = []
    done_steps: list[BoolArray] = []
    delta_steps: list[FloatArray] = []
    teacher_steps: list[IntArray] = []
    completed_returns: list[float] = []
    for _ in range(config.rollout_length):
        states = tuple(worker.ready() for worker in workers)
        observations = tuple(state[0] for state in states)
        features = np.stack([encode_observation(value) for value in observations])
        masks = np.stack([state[1] for state in states])
        feature_tensor = torch.from_numpy(features).to(device)
        mask_tensor = torch.from_numpy(masks).to(device)
        with torch.inference_mode():
            logits, values, _ = model(feature_tensor)
            logits = logits.masked_fill(~mask_tensor, -torch.inf)
            distribution = Categorical(logits=logits)
            actions = distribution.sample()
            log_probabilities = distribution.log_prob(actions)
        action_indices = tuple(ActionIndex(int(value)) for value in actions.tolist())
        results = tuple(
            executor.map(
                lambda pair: pair[0].step(pair[1]),
                zip(workers, action_indices, strict=True),
            )
        )
        next_features = np.stack([encode_observation(result[0]) for result in results])
        rewards = np.asarray([result[1] for result in results], dtype=np.float32)
        dones = np.asarray([result[2] for result in results], dtype=np.bool_)
        completed_returns.extend(
            result[3] for result in results if result[3] is not None
        )
        teacher_actions = np.asarray(
            [int(teacher.select(obs, mask)) for obs, mask in states], dtype=np.int64
        )
        feature_steps.append(features)
        mask_steps.append(masks)
        action_steps.append(actions.cpu().numpy())
        log_probability_steps.append(log_probabilities.cpu().numpy())
        value_steps.append(values.cpu().numpy())
        reward_steps.append(rewards)
        done_steps.append(dones)
        delta_steps.append(next_features - features)
        teacher_steps.append(teacher_actions)

    final_states = tuple(worker.ready() for worker in workers)
    final_features = torch.from_numpy(
        np.stack([encode_observation(state[0]) for state in final_states])
    ).to(device)
    with torch.inference_mode():
        _, final_values, _ = model(final_features)
    advantages, returns = _gae(
        np.stack(reward_steps),
        np.stack(value_steps),
        np.stack(done_steps),
        final_values.cpu().numpy(),
        discount=config.discount,
        gae_lambda=config.gae_lambda,
    )
    return _Rollout(
        np.concatenate(feature_steps),
        np.concatenate(mask_steps),
        np.concatenate(action_steps),
        np.concatenate(log_probability_steps),
        np.concatenate(value_steps),
        advantages.reshape(-1),
        returns.reshape(-1),
        np.concatenate(delta_steps),
        np.concatenate(teacher_steps),
        tuple(completed_returns),
    )


def _gae(
    rewards: FloatArray,
    values: FloatArray,
    dones: BoolArray,
    final_values: FloatArray,
    *,
    discount: float,
    gae_lambda: float,
) -> tuple[FloatArray, FloatArray]:
    advantages = np.zeros_like(rewards)
    future_advantage = np.zeros(rewards.shape[1], dtype=np.float32)
    next_values = final_values
    for step in range(len(rewards) - 1, -1, -1):
        alive = 1.0 - dones[step].astype(np.float32)
        delta = rewards[step] + discount * next_values * alive - values[step]
        future_advantage = delta + discount * gae_lambda * alive * future_advantage
        advantages[step] = future_advantage
        next_values = values[step]
    return advantages, advantages + values


def _ppo_update(
    model: SemanticActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: _Rollout,
    *,
    config: PpoConfig,
) -> tuple[float, float, float]:
    device = torch.device(config.device)
    tensors = tuple(
        torch.from_numpy(value).to(device)
        for value in (
            rollout.features,
            rollout.masks,
            rollout.actions,
            rollout.log_probabilities,
            rollout.advantages,
            rollout.returns,
            rollout.next_deltas,
            rollout.teacher_actions,
        )
    )
    features, masks, actions, old_logs, advantages, returns, deltas, teachers = tensors
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    last_losses = (0.0, 0.0, 0.0)
    for _ in range(config.epochs_per_update):
        for indices in torch.randperm(len(actions), device=device).split(
            config.minibatch_size
        ):
            logits, values, predicted_deltas = model(features[indices])
            logits = logits.masked_fill(~masks[indices], -torch.inf)
            distribution = Categorical(logits=logits)
            logs = distribution.log_prob(actions[indices])
            ratio = torch.exp(logs - old_logs[indices])
            unclipped = ratio * advantages[indices]
            clipped = (
                torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio)
                * advantages[indices]
            )
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, returns[indices])
            echo_loss = F.smooth_l1_loss(predicted_deltas, deltas[indices])
            imitation_loss = F.cross_entropy(logits, teachers[indices])
            loss = (
                policy_loss
                + config.value_weight * value_loss
                - config.entropy_weight * distribution.entropy().mean()
                + config.echo_weight * echo_loss
                + config.imitation_weight * imitation_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            last_losses = (
                float(policy_loss.item()),
                float(value_loss.item()),
                float(echo_loss.item()),
            )
    return last_losses


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
