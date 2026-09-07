"""Concurrent online masked PPO fine-tuning for the semantic actor-critic."""

from __future__ import annotations

import random
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter

import numpy as np
import torch
from numpy.typing import NDArray
from torch.distributions import Categorical
from torch.nn import functional as F

from dcss_rl.env import DcssEnv, RewardShaping
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
    RewardWeight,
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
_ZERO_REWARD_WEIGHT = RewardWeight(0.0)
_GAME_START_ATTEMPTS = 3


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
    explored_cell_reward: RewardWeight = _ZERO_REWARD_WEIGHT
    depth_progress_reward: RewardWeight = _ZERO_REWARD_WEIGHT
    experience_progress_reward: RewardWeight = _ZERO_REWARD_WEIGHT
    hp_fraction_reward: RewardWeight = _ZERO_REWARD_WEIGHT
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
                self.explored_cell_reward,
                self.depth_progress_reward,
                self.experience_progress_reward,
                self.hp_fraction_reward,
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
    reward_shaping: RewardShaping
    rng: np.random.Generator
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
            (self.worker_index + self.episode_index) % len(self.suite.cases)
        ]
        episode_index = self.episode_index
        self.episode_index += 1
        last_timeout: TimeoutError | None = None
        for attempt in range(_GAME_START_ATTEMPTS):
            run_root = (
                self.root
                / f"worker-{self.worker_index}"
                / f"episode-{episode_index}-{case.case_id}-attempt-{attempt}"
            )
            self.env = DcssEnv(
                self.binary,
                game_config=GameConfig(seed=GameSeed(case.seed)),
                max_steps=StepLimit(self.suite.step_limit),
                run_root=run_root,
                reward_shaping=self.reward_shaping,
            )
            try:
                observation, info = self.env.reset()
                break
            except TimeoutError as error:
                last_timeout = error
                self.env.close()
                self.env = None
        else:
            raise RuntimeError(
                f"worker {self.worker_index} could not start {case.case_id} after "
                f"{_GAME_START_ATTEMPTS} attempts"
            ) from last_timeout
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


@dataclass(frozen=True, slots=True)
class _WorkerRollout:
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
        _Worker(
            binary,
            suite,
            run_root,
            index,
            RewardShaping(
                explored_cell=config.explored_cell_reward,
                depth_progress=config.depth_progress_reward,
                experience_progress=config.experience_progress_reward,
                hp_fraction=config.hp_fraction_reward,
            ),
            np.random.default_rng(config.seed + index),
        )
        for index in range(config.workers)
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
                mean_return = (
                    float(np.mean(completed_returns)) if completed_returns else 0.0
                )
                save_checkpoint(
                    output_checkpoint,
                    model=model,
                    policy_id=policy_id,
                    training_metadata=_checkpoint_metadata(
                        config,
                        updates=UpdateCount(update_index + 1),
                        mean_episode_return=mean_return,
                    ),
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
    return PpoReport(
        output_checkpoint,
        int(config.updates * config.rollout_length * config.workers),
        len(completed_returns),
        mean_return,
        policy_loss,
        value_loss,
        echo_loss,
    )


def _checkpoint_metadata(
    config: PpoConfig,
    *,
    updates: UpdateCount,
    mean_episode_return: float,
) -> PpoCheckpointMetadata:
    return PpoCheckpointMetadata(
        training_method="masked-ppo",
        seed=config.seed,
        updates=updates,
        rollout_steps=config.rollout_length,
        worker_count=config.workers,
        learning_rate=config.learning_rate,
        echo_weight=config.echo_weight,
        value_weight=config.value_weight,
        imitation_weight=config.imitation_weight,
        explored_cell_reward=config.explored_cell_reward,
        depth_progress_reward=config.depth_progress_reward,
        experience_progress_reward=config.experience_progress_reward,
        hp_fraction_reward=config.hp_fraction_reward,
        clip_ratio=config.clip_ratio,
        mean_episode_return=mean_episode_return,
    )


def _collect_rollout(
    model: SemanticActorCritic,
    workers: tuple[_Worker, ...],
    executor: ThreadPoolExecutor,
    teacher: ScriptedMibePolicy,
    *,
    config: PpoConfig,
) -> _Rollout:
    model_lock = Lock()
    worker_rollouts = tuple(
        executor.map(
            lambda worker: _collect_worker_rollout(
                model,
                worker,
                teacher,
                model_lock=model_lock,
                config=config,
            ),
            workers,
        )
    )
    return _Rollout(
        np.concatenate([rollout.features for rollout in worker_rollouts]),
        np.concatenate([rollout.masks for rollout in worker_rollouts]),
        np.concatenate([rollout.actions for rollout in worker_rollouts]),
        np.concatenate([rollout.log_probabilities for rollout in worker_rollouts]),
        np.concatenate([rollout.values for rollout in worker_rollouts]),
        np.concatenate([rollout.advantages for rollout in worker_rollouts]),
        np.concatenate([rollout.returns for rollout in worker_rollouts]),
        np.concatenate([rollout.next_deltas for rollout in worker_rollouts]),
        np.concatenate([rollout.teacher_actions for rollout in worker_rollouts]),
        tuple(
            value for rollout in worker_rollouts for value in rollout.completed_returns
        ),
    )


def _collect_worker_rollout(
    model: SemanticActorCritic,
    worker: _Worker,
    teacher: ScriptedMibePolicy,
    *,
    model_lock: Lock,
    config: PpoConfig,
) -> _WorkerRollout:
    device = torch.device(config.device)
    features: list[FloatArray] = []
    masks: list[BoolArray] = []
    actions: list[int] = []
    log_probabilities: list[float] = []
    values: list[float] = []
    rewards: list[float] = []
    dones: list[bool] = []
    next_deltas: list[FloatArray] = []
    teacher_actions: list[int] = []
    completed_returns: list[float] = []
    for _ in range(config.rollout_length):
        observation, mask = worker.ready()
        feature = encode_observation(observation)
        with model_lock, torch.inference_mode():
            feature_tensor = torch.from_numpy(feature).to(device).unsqueeze(0)
            mask_tensor = torch.from_numpy(mask).to(device).unsqueeze(0)
            logits, value_tensor, _ = model(feature_tensor)
            masked_logits = logits.squeeze(0).masked_fill(
                ~mask_tensor.squeeze(0), -torch.inf
            )
            probabilities = torch.softmax(masked_logits, dim=-1).cpu().numpy()
            value = float(value_tensor.item())
        action = int(worker.rng.choice(len(probabilities), p=probabilities))
        next_observation, reward, done, completed_return = worker.step(
            ActionIndex(action)
        )
        next_feature = encode_observation(next_observation)
        features.append(feature)
        masks.append(mask)
        actions.append(action)
        log_probabilities.append(float(np.log(probabilities[action])))
        values.append(value)
        rewards.append(reward)
        dones.append(done)
        next_deltas.append(next_feature - feature)
        teacher_actions.append(int(teacher.select(observation, mask)))
        if completed_return is not None:
            completed_returns.append(completed_return)

    final_value = 0.0
    if not dones[-1]:
        final_observation, _ = worker.ready()
        final_feature = encode_observation(final_observation)
        with model_lock, torch.inference_mode():
            _, final_value_tensor, _ = model(
                torch.from_numpy(final_feature).to(device).unsqueeze(0)
            )
            final_value = float(final_value_tensor.item())
    reward_array = np.asarray(rewards, dtype=np.float32)[:, None]
    value_array = np.asarray(values, dtype=np.float32)[:, None]
    done_array = np.asarray(dones, dtype=np.bool_)[:, None]
    advantages, returns = _gae(
        reward_array,
        value_array,
        done_array,
        np.asarray([final_value], dtype=np.float32),
        discount=config.discount,
        gae_lambda=config.gae_lambda,
    )
    return _WorkerRollout(
        np.stack(features),
        np.stack(masks),
        np.asarray(actions, dtype=np.int64),
        np.asarray(log_probabilities, dtype=np.float32),
        np.asarray(values, dtype=np.float32),
        advantages.reshape(-1),
        returns.reshape(-1),
        np.stack(next_deltas),
        np.asarray(teacher_actions, dtype=np.int64),
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
    teacher_counts = torch.bincount(teachers, minlength=model.config.action_count)
    teacher_weights = torch.zeros_like(teacher_counts, dtype=torch.float32)
    present_teacher_actions = teacher_counts > 0
    teacher_weights[present_teacher_actions] = len(teachers) / (
        present_teacher_actions.sum() * teacher_counts[present_teacher_actions]
    )
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
            imitation_loss = F.cross_entropy(
                logits, teachers[indices], weight=teacher_weights
            )
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
