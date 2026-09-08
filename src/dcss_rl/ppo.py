"""Concurrent online masked PPO fine-tuning for the semantic actor-critic."""

from __future__ import annotations

import random
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
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
from dcss_rl.schedule import TrainingSeedSchedule
from dcss_rl.schema import ObservationData
from dcss_rl.units import (
    ActionIndex,
    BatchSize,
    CaseCount,
    DecisionsPerSecond,
    EpisodeIndex,
    EpochCount,
    GameSeed,
    InferenceBatchCount,
    InferenceBatchSize,
    LearningRate,
    LossWeight,
    MeanInferenceBatchSize,
    Probability,
    RewardWeight,
    RolloutLength,
    Seconds,
    StepLimit,
    UpdateCount,
    WorkerCount,
    WorkerIndex,
)
from dcss_rl.webtiles import GameConfig

type FloatArray = NDArray[np.float32]
type BoolArray = NDArray[np.bool_]
type IntArray = NDArray[np.int64]

_DEFAULT_UPDATES = UpdateCount(4)
_DEFAULT_ROLLOUT_LENGTH = RolloutLength(128)
_DEFAULT_WORKERS = WorkerCount(5)
_DEFAULT_INFERENCE_BATCH_SIZE = InferenceBatchSize(64)
_DEFAULT_INFERENCE_BATCH_WAIT = Seconds(0.001)
_DEFAULT_BATCH_SIZE = BatchSize(256)
_DEFAULT_LEARNING_RATE = LearningRate(1e-4)
_DEFAULT_ECHO_WEIGHT = LossWeight(0.1)
_DEFAULT_POLICY_WEIGHT = LossWeight(1.0)
_DEFAULT_VALUE_WEIGHT = LossWeight(0.5)
_DEFAULT_ENTROPY_WEIGHT = LossWeight(0.01)
_DEFAULT_IMITATION_WEIGHT = LossWeight(0.1)
_DEFAULT_TEACHER_BALANCE_EXPONENT = Probability(0.5)
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
    inference_batch_size: InferenceBatchSize = _DEFAULT_INFERENCE_BATCH_SIZE
    inference_batch_wait: Seconds = _DEFAULT_INFERENCE_BATCH_WAIT
    minibatch_size: BatchSize = _DEFAULT_BATCH_SIZE
    learning_rate: LearningRate = _DEFAULT_LEARNING_RATE
    echo_weight: LossWeight = _DEFAULT_ECHO_WEIGHT
    policy_weight: LossWeight = _DEFAULT_POLICY_WEIGHT
    value_weight: LossWeight = _DEFAULT_VALUE_WEIGHT
    entropy_weight: LossWeight = _DEFAULT_ENTROPY_WEIGHT
    imitation_weight: LossWeight = _DEFAULT_IMITATION_WEIGHT
    aggregate_imitation_replay: bool = True
    teacher_balance_exponent: Probability = _DEFAULT_TEACHER_BALANCE_EXPONENT
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
            self.inference_batch_size,
            self.minibatch_size,
            self.epochs_per_update,
        )
        if any(value < 1 for value in positive_integers):
            raise ValueError("PPO counts must be positive")
        if self.learning_rate <= 0:
            raise ValueError("PPO learning rate must be positive")
        if self.inference_batch_wait < 0:
            raise ValueError("inference batch wait cannot be negative")
        if not 0 < self.discount <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("PPO discount and GAE lambda must be probabilities")
        if not 0 < self.clip_ratio < 1:
            raise ValueError("PPO clip ratio must be between zero and one")
        if not 0 <= self.teacher_balance_exponent <= 1:
            raise ValueError("teacher balance exponent must be a probability")
        if any(
            weight < 0
            for weight in (
                self.echo_weight,
                self.policy_weight,
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
    imitation_loss: float
    teacher_agreement: Probability


@dataclass(frozen=True, slots=True)
class PpoUpdateReport:
    update: UpdateCount
    decisions: int
    decision_rate: DecisionsPerSecond
    collection_seconds: Seconds
    optimization_seconds: Seconds
    checkpoint_seconds: Seconds
    inference_batches: InferenceBatchCount
    mean_inference_batch_size: MeanInferenceBatchSize
    completed_episodes: int
    mean_completed_return: float
    policy_loss: float
    value_loss: float
    echo_loss: float
    imitation_loss: float
    teacher_agreement: Probability


@dataclass(frozen=True, slots=True)
class PpoLosses:
    policy: float
    value: float
    echo: float
    imitation: float


@dataclass(slots=True)
class _Worker:
    binary: Path
    suite: EvaluationSuite
    root: Path
    worker_index: WorkerIndex
    seed_schedule: TrainingSeedSchedule
    reward_shaping: RewardShaping
    rng: np.random.Generator
    episode_index: EpisodeIndex = field(default_factory=lambda: EpisodeIndex(0))
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
            self.seed_schedule.case_index(self.worker_index, self.episode_index)
        ]
        episode_index = self.episode_index
        self.episode_index = EpisodeIndex(self.episode_index + 1)
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
    inference_batches: InferenceBatchCount
    mean_inference_batch_size: MeanInferenceBatchSize


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


@dataclass(frozen=True, slots=True)
class _InferenceResult:
    probabilities: FloatArray
    value: float


@dataclass(frozen=True, slots=True)
class _InferenceRequest:
    feature: FloatArray
    mask: BoolArray
    future: Future[_InferenceResult]


class _InferenceBatcher:
    """Combine asynchronous worker requests into bounded GPU forwards."""

    def __init__(
        self,
        model: SemanticActorCritic,
        *,
        device: str,
        batch_size: InferenceBatchSize,
        batch_wait: Seconds,
    ) -> None:
        self._model = model
        self._device = torch.device(device)
        self._batch_size = batch_size
        self._batch_wait = batch_wait
        self._requests: Queue[_InferenceRequest | None] = Queue()
        self._state_lock = Lock()
        self._closed = Event()
        self.request_count = 0
        self.batch_count = 0
        self._thread = Thread(target=self._run, name="ppo-inference", daemon=True)
        self._thread.start()

    @property
    def feature_spec_version(self) -> int:
        return self._model.config.feature_spec_version

    def infer(self, feature: FloatArray, mask: BoolArray) -> _InferenceResult:
        future: Future[_InferenceResult] = Future()
        with self._state_lock:
            if self._closed.is_set():
                raise RuntimeError("inference batcher is closed")
            self._requests.put(_InferenceRequest(feature, mask, future))
        return future.result()

    def close(self) -> None:
        with self._state_lock:
            self._closed.set()
            self._requests.put(None)
        self._thread.join()

    def _run(self) -> None:
        while (first := self._requests.get()) is not None:
            requests = [first]
            deadline = perf_counter() + self._batch_wait
            while len(requests) < self._batch_size:
                remaining = deadline - perf_counter()
                if remaining <= 0:
                    break
                try:
                    request = self._requests.get(timeout=remaining)
                except Empty:
                    break
                if request is None:
                    self._resolve(requests)
                    return
                requests.append(request)
            try:
                self._resolve(requests)
            except BaseException as error:
                for request in requests:
                    request.future.set_exception(error)

    def _resolve(self, requests: list[_InferenceRequest]) -> None:
        self.request_count += len(requests)
        self.batch_count += 1
        features = torch.from_numpy(
            np.stack([request.feature for request in requests])
        ).to(self._device)
        masks = torch.from_numpy(np.stack([request.mask for request in requests])).to(
            self._device
        )
        with torch.inference_mode():
            logits, values, _ = self._model(features)
            probabilities = (
                torch.softmax(logits.masked_fill(~masks, -torch.inf), dim=-1)
                .cpu()
                .numpy()
            )
            host_values = values.cpu().numpy()
        for index, request in enumerate(requests):
            request.future.set_result(
                _InferenceResult(probabilities[index], float(host_values[index]))
            )


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
    seed_schedule = TrainingSeedSchedule(CaseCount(len(suite.cases)), config.workers)
    workers = tuple(
        _Worker(
            binary,
            suite,
            run_root,
            WorkerIndex(index),
            seed_schedule,
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
    losses = PpoLosses(0.0, 0.0, 0.0, 0.0)
    teacher_agreement = Probability(0.0)
    imitation_replay: list[_Rollout] = []
    try:
        with ThreadPoolExecutor(max_workers=config.workers) as executor:
            for update_index in range(config.updates):
                update_started = perf_counter()
                rollout = _collect_rollout(
                    model, workers, executor, teacher, config=config
                )
                collection_finished = perf_counter()
                completed_returns.extend(rollout.completed_returns)
                imitation_replay.append(rollout)
                losses = _ppo_update(
                    model,
                    optimizer,
                    rollout,
                    imitation_replay=tuple(imitation_replay)
                    if config.aggregate_imitation_replay
                    else (rollout,),
                    config=config,
                )
                optimization_finished = perf_counter()
                teacher_agreement = Probability(
                    float(np.mean(rollout.actions == rollout.teacher_actions))
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
                checkpoint_finished = perf_counter()
                if progress is not None:
                    update_decisions = int(config.rollout_length * config.workers)
                    update_returns = rollout.completed_returns
                    progress(
                        PpoUpdateReport(
                            UpdateCount(update_index + 1),
                            update_decisions,
                            DecisionsPerSecond(
                                update_decisions
                                / (checkpoint_finished - update_started)
                            ),
                            Seconds(collection_finished - update_started),
                            Seconds(optimization_finished - collection_finished),
                            Seconds(checkpoint_finished - optimization_finished),
                            rollout.inference_batches,
                            rollout.mean_inference_batch_size,
                            len(update_returns),
                            float(np.mean(update_returns)) if update_returns else 0.0,
                            losses.policy,
                            losses.value,
                            losses.echo,
                            losses.imitation,
                            teacher_agreement,
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
        losses.policy,
        losses.value,
        losses.echo,
        losses.imitation,
        teacher_agreement,
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
        inference_batch_size=config.inference_batch_size,
        inference_batch_wait_seconds=config.inference_batch_wait,
        learning_rate=config.learning_rate,
        echo_weight=config.echo_weight,
        policy_weight=config.policy_weight,
        value_weight=config.value_weight,
        imitation_weight=config.imitation_weight,
        aggregate_imitation_replay=config.aggregate_imitation_replay,
        teacher_balance_exponent=config.teacher_balance_exponent,
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
    batcher = _InferenceBatcher(
        model,
        device=config.device,
        batch_size=config.inference_batch_size,
        batch_wait=config.inference_batch_wait,
    )
    try:
        worker_rollouts = tuple(
            executor.map(
                lambda worker: _collect_worker_rollout(
                    worker,
                    teacher,
                    batcher=batcher,
                    config=config,
                ),
                workers,
            )
        )
    finally:
        batcher.close()
    inference_batches = InferenceBatchCount(batcher.batch_count)
    mean_inference_batch_size = MeanInferenceBatchSize(
        batcher.request_count / batcher.batch_count if batcher.batch_count else 0.0
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
        inference_batches,
        mean_inference_batch_size,
    )


def _collect_worker_rollout(
    worker: _Worker,
    teacher: ScriptedMibePolicy,
    *,
    batcher: _InferenceBatcher,
    config: PpoConfig,
) -> _WorkerRollout:
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
        feature = encode_observation(
            observation, spec_version=batcher.feature_spec_version
        )
        inference = batcher.infer(feature, mask)
        probabilities = inference.probabilities
        value = inference.value
        action = int(worker.rng.choice(len(probabilities), p=probabilities))
        next_observation, reward, done, completed_return = worker.step(
            ActionIndex(action)
        )
        next_feature = encode_observation(
            next_observation, spec_version=batcher.feature_spec_version
        )
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
        final_observation, final_mask = worker.ready()
        final_feature = encode_observation(
            final_observation, spec_version=batcher.feature_spec_version
        )
        final_value = batcher.infer(final_feature, final_mask).value
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
    imitation_replay: tuple[_Rollout, ...],
    config: PpoConfig,
) -> PpoLosses:
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
        )
    )
    features, masks, actions, old_logs, advantages, returns, deltas = tensors
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    replay_features = torch.from_numpy(
        np.concatenate([item.features for item in imitation_replay])
    ).to(device)
    replay_masks = torch.from_numpy(
        np.concatenate([item.masks for item in imitation_replay])
    ).to(device)
    replay_teachers = torch.from_numpy(
        np.concatenate([item.teacher_actions for item in imitation_replay])
    ).to(device)
    teacher_counts = torch.bincount(
        replay_teachers, minlength=model.config.action_count
    )
    teacher_weights = torch.zeros_like(teacher_counts, dtype=torch.float32)
    present_teacher_actions = teacher_counts > 0
    inverse_frequency = len(replay_teachers) / (
        present_teacher_actions.sum() * teacher_counts[present_teacher_actions]
    )
    teacher_weights[present_teacher_actions] = inverse_frequency.pow(
        config.teacher_balance_exponent
    )
    last_losses = PpoLosses(0.0, 0.0, 0.0, 0.0)
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
            replay_indices = torch.randint(
                len(replay_teachers), (len(indices),), device=device
            )
            replay_logits, _, _ = model(replay_features[replay_indices])
            replay_logits = replay_logits.masked_fill(
                ~replay_masks[replay_indices], -torch.inf
            )
            imitation_loss = F.cross_entropy(
                replay_logits,
                replay_teachers[replay_indices],
                weight=teacher_weights,
            )
            loss = (
                config.policy_weight * policy_loss
                + config.value_weight * value_loss
                - config.entropy_weight * distribution.entropy().mean()
                + config.echo_weight * echo_loss
                + config.imitation_weight * imitation_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            last_losses = PpoLosses(
                float(policy_loss.item()),
                float(value_loss.item()),
                float(echo_loss.item()),
                float(imitation_loss.item()),
            )
    return last_losses


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
