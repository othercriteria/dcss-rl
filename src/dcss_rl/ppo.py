"""Concurrent online masked PPO fine-tuning for the semantic actor-critic."""

from __future__ import annotations

import hashlib
import json
import random
import tempfile
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
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import functional as F

from dcss_rl.actions import Action, ActionKind
from dcss_rl.checkpointing import update_checkpoint_path
from dcss_rl.costs import SemanticCycleTracker, training_reward
from dcss_rl.env import ACTION_COUNT, DcssEnv, RewardShaping, action_to_index
from dcss_rl.evaluation import EvaluationSuite
from dcss_rl.features import FEATURE_SPEC_VERSION, encode_observation, feature_count
from dcss_rl.history import encode_action_history
from dcss_rl.learned import (
    LearnedPolicy,
    ModelConfig,
    PpoCheckpointMetadata,
    SemanticActorCritic,
    add_action_history,
    align_action_count,
    align_feature_spec,
    save_checkpoint,
)
from dcss_rl.policy import ActionHistory, ScriptedMibePolicy
from dcss_rl.returns import ReturnBoundaryMode, generalized_advantage_estimate
from dcss_rl.schedule import TrainingSeedSchedule
from dcss_rl.schema import ObservationData
from dcss_rl.training import load_imitation_replay_episode
from dcss_rl.units import (
    ActionHistoryLength,
    ActionIndex,
    BatchSize,
    CaseCount,
    DecisionCost,
    DecisionsPerSecond,
    DecisionWindow,
    EpisodeIndex,
    EpochCount,
    FeatureSpecVersion,
    GameSeed,
    ImitationReplayCacheKey,
    ImitationSampleCount,
    InferenceBatchCount,
    InferenceBatchSize,
    Keycode,
    LearningRate,
    LossWeight,
    MeanInferenceBatchSize,
    Probability,
    RewardWeight,
    RolloutLength,
    Seconds,
    ShortCycleCost,
    StartupAttemptCount,
    StartupAttemptIndex,
    StepLimit,
    TerminalOutcome,
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
_ZERO_UPDATES = UpdateCount(0)
_ZERO_REWARD_WEIGHT = RewardWeight(0.0)
_ZERO_DECISION_COST = DecisionCost(0.0)
_ZERO_SHORT_CYCLE_COST = ShortCycleCost(0.0)
_DEFAULT_SHORT_CYCLE_WINDOW = DecisionWindow(8)
_GAME_START_ATTEMPTS = StartupAttemptCount(3)
_WIN_OUTCOME = TerminalOutcome("won")
_IMITATION_REPLAY_CACHE_SCHEMA = 1
_CACHE_READ_CHUNK_BYTES = 1024 * 1024


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
    return_boundary: ReturnBoundaryMode = ReturnBoundaryMode.EPISODIC
    decision_cost: DecisionCost = _ZERO_DECISION_COST
    short_cycle_cost: ShortCycleCost = _ZERO_SHORT_CYCLE_COST
    short_cycle_window: DecisionWindow = _DEFAULT_SHORT_CYCLE_WINDOW
    action_history_length: ActionHistoryLength | None = None
    new_action_warmup_updates: UpdateCount = _ZERO_UPDATES
    new_action_warmup_menu_keycodes: tuple[Keycode, ...] = ()
    warmup_action_kinds: tuple[ActionKind, ...] = ()
    imitation_trajectories: tuple[Path, ...] = ()
    imitation_replay_cache_directory: Path | None = None
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
        if self.action_history_length is not None and self.action_history_length < 0:
            raise ValueError("action history length cannot be negative")
        if self.new_action_warmup_updates < 0:
            raise ValueError("new-action warmup updates cannot be negative")
        if self.short_cycle_window < 1:
            raise ValueError("short-cycle window must be positive")
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
                self.decision_cost,
                self.short_cycle_cost,
            )
        ):
            raise ValueError("PPO weights and costs cannot be negative")


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
    short_cycles: int


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
    short_cycles: int


@dataclass(frozen=True, slots=True)
class PpoPreparationReport:
    model_setup_seconds: Seconds
    imitation_replay_seconds: Seconds
    imitation_samples: ImitationSampleCount
    imitation_replay_cache_hit: bool


@dataclass(frozen=True, slots=True)
class PpoLosses:
    policy: float
    value: float
    echo: float
    imitation: float


@dataclass(frozen=True, slots=True)
class _ImitationReplay:
    features: FloatArray
    action_histories: FloatArray
    masks: BoolArray
    teacher_actions: IntArray


@dataclass(frozen=True, slots=True)
class _PreparedImitationReplay:
    replay: tuple[_ImitationReplay, ...]
    cache_hit: bool


@dataclass(frozen=True, slots=True)
class _WorkerStep:
    observation: ObservationData
    action_mask: BoolArray
    reward: float
    terminated: bool
    truncated: bool
    completed_return: float | None
    action_history: ActionHistory
    terminal_outcome: TerminalOutcome | None
    short_cycle: bool


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
    action_history: list[ActionIndex] = field(default_factory=list)
    cycle_tracker: SemanticCycleTracker | None = None

    def ready(self) -> tuple[ObservationData, BoolArray]:
        if self.observation is None:
            self._reset()
        if self.observation is None or self.action_mask is None:
            raise RuntimeError("online worker failed to reset")
        return self.observation, self.action_mask

    def history(self) -> ActionHistory:
        return tuple(self.action_history)

    def step(self, action: ActionIndex) -> _WorkerStep:
        if self.env is None:
            raise RuntimeError("online worker is not ready")
        observation, reward, terminated, truncated, info = self.env.step_typed(action)
        done = terminated or truncated
        mask = info.get("action_mask")
        if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
            raise RuntimeError("environment returned an invalid action mask")
        self.episode_return += reward
        completed_return = self.episode_return if done else None
        raw_outcome = info.get("outcome")
        terminal_outcome = (
            TerminalOutcome(raw_outcome) if isinstance(raw_outcome, str) else None
        )
        short_cycle = (
            self.cycle_tracker.observe(observation)
            if self.cycle_tracker is not None
            else False
        )
        self.action_history.append(action)
        next_history = tuple(self.action_history)
        if done:
            self.env.close()
            self.env = None
            self.observation = None
            self.action_mask = None
            self.episode_return = 0.0
            self.action_history.clear()
        else:
            self.observation = observation
            self.action_mask = mask
        return _WorkerStep(
            observation,
            mask,
            reward,
            terminated,
            truncated,
            completed_return,
            next_history,
            terminal_outcome,
            short_cycle,
        )

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
        for raw_attempt in range(_GAME_START_ATTEMPTS):
            attempt = StartupAttemptIndex(raw_attempt)
            run_root = _worker_run_root(
                self.root, self.worker_index, episode_index, attempt
            )
            self.env = DcssEnv(
                self.binary,
                game_config=GameConfig(seed=GameSeed(case.seed)),
                max_steps=StepLimit(self.suite.step_limit),
                run_root=run_root,
                reward_shaping=self.reward_shaping,
            )
            try:
                observation, info = self.env.reset_typed()
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
        if self.cycle_tracker is not None:
            self.cycle_tracker.reset(observation)


def _worker_run_root(
    root: Path,
    worker_index: WorkerIndex,
    episode_index: EpisodeIndex,
    attempt_index: StartupAttemptIndex,
) -> Path:
    """Build a bounded path; suite case IDs already live in trajectory metadata."""
    return (
        root
        / f"worker-{worker_index}"
        / f"episode-{episode_index}-attempt-{attempt_index}"
    )


@dataclass(frozen=True, slots=True)
class _Rollout:
    features: FloatArray
    action_histories: FloatArray
    masks: BoolArray
    actions: IntArray
    log_probabilities: FloatArray
    values: FloatArray
    advantages: FloatArray
    returns: FloatArray
    next_deltas: FloatArray
    teacher_actions: IntArray
    completed_returns: tuple[float, ...]
    short_cycles: int
    inference_batches: InferenceBatchCount
    mean_inference_batch_size: MeanInferenceBatchSize


@dataclass(frozen=True, slots=True)
class _WorkerRollout:
    features: FloatArray
    action_histories: FloatArray
    masks: BoolArray
    actions: IntArray
    log_probabilities: FloatArray
    values: FloatArray
    advantages: FloatArray
    returns: FloatArray
    next_deltas: FloatArray
    teacher_actions: IntArray
    completed_returns: tuple[float, ...]
    short_cycles: int


@dataclass(frozen=True, slots=True)
class _InferenceResult:
    probabilities: FloatArray
    value: float


@dataclass(frozen=True, slots=True)
class _InferenceRequest:
    slot: WorkerIndex
    feature: FloatArray
    action_history: FloatArray
    mask: BoolArray
    future: Future[_InferenceResult]


@dataclass(frozen=True, slots=True)
class _InferenceInputBatch:
    features: FloatArray
    action_histories: FloatArray
    masks: BoolArray


class _InferenceBatcher:
    """Combine asynchronous worker requests into bounded GPU forwards."""

    def __init__(
        self,
        model: SemanticActorCritic,
        *,
        device: str,
        batch_size: InferenceBatchSize,
        batch_wait: Seconds,
        slot_count: WorkerCount,
    ) -> None:
        self._model = model
        self._device = torch.device(device)
        self._batch_size = batch_size
        self._batch_wait = batch_wait
        self._slot_count = slot_count
        self._requests: Queue[_InferenceRequest | None] = Queue()
        self._state_lock = Lock()
        self._closed = Event()
        self.request_count = 0
        self.batch_count = 0
        self._thread = Thread(target=self._run, name="ppo-inference", daemon=True)
        self._thread.start()

    @property
    def feature_spec_version(self) -> FeatureSpecVersion:
        return self._model.config.feature_spec_version

    @property
    def model_config(self) -> ModelConfig:
        return self._model.config

    def infer(
        self,
        slot: WorkerIndex,
        feature: FloatArray,
        action_history: FloatArray,
        mask: BoolArray,
    ) -> _InferenceResult:
        future: Future[_InferenceResult] = Future()
        with self._state_lock:
            if self._closed.is_set():
                raise RuntimeError("inference batcher is closed")
            self._requests.put(
                _InferenceRequest(slot, feature, action_history, mask, future)
            )
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
        inputs = _fixed_inference_inputs(requests, self._slot_count)
        features = torch.from_numpy(inputs.features).to(self._device)
        masks = torch.from_numpy(inputs.masks).to(self._device)
        histories = torch.from_numpy(inputs.action_histories).to(self._device)
        with torch.inference_mode():
            logits, values, _ = self._model(features, histories)
            probabilities = (
                torch.softmax(logits.masked_fill(~masks, -torch.inf), dim=-1)
                .cpu()
                .numpy()
            )
            host_values = values.cpu().numpy()
        for request in requests:
            request.future.set_result(
                _InferenceResult(
                    probabilities[request.slot], float(host_values[request.slot])
                )
            )


def _fixed_inference_inputs(
    requests: list[_InferenceRequest], slot_count: WorkerCount
) -> _InferenceInputBatch:
    """Place workers in stable rows of one reproducible GPU matrix shape."""
    slots = [request.slot for request in requests]
    if (
        not requests
        or len(set(slots)) != len(slots)
        or any(not 0 <= slot < slot_count for slot in slots)
    ):
        raise ValueError("inference requests require unique configured worker slots")
    features = np.zeros((slot_count, len(requests[0].feature)), dtype=np.float32)
    histories = np.zeros(
        (slot_count, len(requests[0].action_history)), dtype=np.float32
    )
    masks = np.zeros((slot_count, len(requests[0].mask)), dtype=np.bool_)
    masks[:, 0] = True
    for request in requests:
        features[request.slot] = request.feature
        histories[request.slot] = request.action_history
        masks[request.slot] = request.mask
    return _InferenceInputBatch(features, histories, masks)


def train_ppo(
    binary: Path,
    suite: EvaluationSuite,
    initial_checkpoint: Path,
    output_checkpoint: Path,
    run_root: Path,
    *,
    config: PpoConfig,
    policy_id: str,
    update_checkpoint_directory: Path | None = None,
    progress: Callable[[PpoUpdateReport], None] | None = None,
    preparation_progress: Callable[[PpoPreparationReport], None] | None = None,
) -> PpoReport:
    """Fine-tune an imitation checkpoint with concurrent on-policy PPO updates."""
    preparation_started = perf_counter()
    _seed_everything(config.seed)
    restored = LearnedPolicy(initial_checkpoint, device=config.device)
    model = restored.model
    established_action_count = restored.checkpoint_action_count
    established_feature_count = int(
        feature_count(restored.checkpoint_feature_spec_version)
    )
    model = align_feature_spec(model, FEATURE_SPEC_VERSION)
    model = align_action_count(model, int(ACTION_COUNT))
    if config.action_history_length is not None:
        model = add_action_history(model, config.action_history_length)
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
            cycle_tracker=SemanticCycleTracker(config.short_cycle_window),
        )
        for index in range(config.workers)
    )
    completed_returns: list[float] = []
    short_cycle_count = 0
    losses = PpoLosses(0.0, 0.0, 0.0, 0.0)
    teacher_agreement = Probability(0.0)
    model_setup_finished = perf_counter()
    prepared_replay = _preloaded_imitation_replay(
        config.imitation_trajectories,
        model=model,
        teacher=teacher,
        cache_directory=config.imitation_replay_cache_directory,
    )
    imitation_replay = list(prepared_replay.replay)
    replay_finished = perf_counter()
    if preparation_progress is not None:
        preparation_progress(
            PpoPreparationReport(
                Seconds(model_setup_finished - preparation_started),
                Seconds(replay_finished - model_setup_finished),
                ImitationSampleCount(
                    sum(len(replay.features) for replay in imitation_replay)
                ),
                prepared_replay.cache_hit,
            )
        )
    try:
        with ThreadPoolExecutor(max_workers=config.workers) as executor:
            for update_index in range(config.updates):
                update_started = perf_counter()
                rollout = _collect_rollout(
                    model, workers, executor, teacher, config=config
                )
                collection_finished = perf_counter()
                completed_returns.extend(rollout.completed_returns)
                short_cycle_count += rollout.short_cycles
                imitation_replay.append(_rollout_imitation_replay(rollout))
                losses = _ppo_update(
                    model,
                    optimizer,
                    rollout,
                    imitation_replay=tuple(imitation_replay)
                    if config.aggregate_imitation_replay
                    else (_rollout_imitation_replay(rollout),),
                    config=config,
                    trainable_action_indices=_warmup_action_indices(
                        established_action_count,
                        model.config.action_count,
                        config.new_action_warmup_menu_keycodes,
                        config.warmup_action_kinds,
                    )
                    if update_index < config.new_action_warmup_updates
                    else None,
                    trainable_feature_indices=tuple(
                        range(
                            established_feature_count,
                            int(feature_count(model.config.feature_spec_version)),
                        )
                    )
                    if update_index < config.new_action_warmup_updates
                    and established_feature_count
                    < feature_count(model.config.feature_spec_version)
                    else None,
                )
                optimization_finished = perf_counter()
                teacher_agreement = Probability(
                    float(np.mean(rollout.actions == rollout.teacher_actions))
                )
                mean_return = (
                    float(np.mean(completed_returns)) if completed_returns else 0.0
                )
                completed_update = UpdateCount(update_index + 1)
                metadata = _checkpoint_metadata(
                    config,
                    updates=completed_update,
                    mean_episode_return=mean_return,
                    action_history_length=model.config.action_history_length,
                )
                save_checkpoint(
                    output_checkpoint,
                    model=model,
                    policy_id=policy_id,
                    training_metadata=metadata,
                )
                if update_checkpoint_directory is not None:
                    save_checkpoint(
                        update_checkpoint_path(
                            update_checkpoint_directory, completed_update
                        ),
                        model=model,
                        policy_id=policy_id,
                        training_metadata=metadata,
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
                            rollout.short_cycles,
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
        short_cycle_count,
    )


def _checkpoint_metadata(
    config: PpoConfig,
    *,
    updates: UpdateCount,
    mean_episode_return: float,
    action_history_length: ActionHistoryLength,
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
        action_history_length=action_history_length,
        new_action_warmup_updates=config.new_action_warmup_updates,
        new_action_warmup_menu_keycodes=tuple(config.new_action_warmup_menu_keycodes),
        warmup_action_kinds=tuple(kind.value for kind in config.warmup_action_kinds),
        imitation_trajectories=tuple(map(str, config.imitation_trajectories)),
        return_boundary=config.return_boundary.value,
        decision_cost=config.decision_cost,
        short_cycle_cost=config.short_cycle_cost,
        short_cycle_window=config.short_cycle_window,
    )


def _preloaded_imitation_replay(
    trajectories: tuple[Path, ...],
    *,
    model: SemanticActorCritic,
    teacher: ScriptedMibePolicy,
    cache_directory: Path | None = None,
) -> _PreparedImitationReplay:
    if not trajectories:
        return _PreparedImitationReplay((), False)
    cache_path = (
        Path(cache_directory)
        / f"{_imitation_replay_cache_key(trajectories, teacher)}.npz"
        if cache_directory is not None
        else None
    )
    if (
        cache_path is not None
        and (cached := _read_imitation_replay_cache(cache_path, model)) is not None
    ):
        return _PreparedImitationReplay((cached,), True)
    episodes = tuple(
        load_imitation_replay_episode(path, teacher=teacher) for path in trajectories
    )
    features = np.concatenate(
        [np.stack(episode.features) for episode in episodes]
    ).astype(np.float32, copy=False)
    masks = np.concatenate([np.stack(episode.masks) for episode in episodes])
    teacher_actions = np.concatenate(
        [np.asarray(episode.actions, dtype=np.int64) for episode in episodes]
    )
    history_width = model.config.action_count * model.config.action_history_length
    replay = _ImitationReplay(
        features,
        np.zeros((len(features), history_width), dtype=np.float32),
        masks,
        teacher_actions,
    )
    if cache_path is not None:
        _write_imitation_replay_cache(cache_path, replay)
    return _PreparedImitationReplay((replay,), False)


def _imitation_replay_cache_key(
    trajectories: tuple[Path, ...], teacher: ScriptedMibePolicy
) -> ImitationReplayCacheKey:
    """Digest every semantic input to the generated replay tensor cache."""
    digest = hashlib.sha256()
    contract = {
        "schema": _IMITATION_REPLAY_CACHE_SCHEMA,
        "feature_spec": int(FEATURE_SPEC_VERSION),
        "action_count": int(ACTION_COUNT),
        "teacher": teacher.policy_id,
        "trajectory_count": len(trajectories),
    }
    digest.update(json.dumps(contract, sort_keys=True).encode())
    for trajectory in trajectories:
        path = Path(trajectory)
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as trajectory_file:
            while chunk := trajectory_file.read(_CACHE_READ_CHUNK_BYTES):
                digest.update(chunk)
    return ImitationReplayCacheKey(digest.hexdigest())


def _read_imitation_replay_cache(
    path: Path, model: SemanticActorCritic
) -> _ImitationReplay | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as cached:
            # NPZ members are materialized as owned ndarrays, not views into the
            # closing ZipFile, so no second 120-MiB feature copy is required here.
            features = cached["features"]
            masks = cached["masks"]
            teacher_actions = cached["teacher_actions"]
    except (OSError, ValueError, KeyError):
        return None
    sample_count = len(features)
    if (
        features.dtype != np.float32
        or features.ndim != 2
        or features.shape[1] != feature_count(FEATURE_SPEC_VERSION)
        or masks.dtype != np.bool_
        or masks.shape != (sample_count, ACTION_COUNT)
        or teacher_actions.dtype != np.int64
        or teacher_actions.shape != (sample_count,)
        or np.any(teacher_actions < 0)
        or np.any(teacher_actions >= ACTION_COUNT)
    ):
        return None
    history_width = model.config.action_count * model.config.action_history_length
    return _ImitationReplay(
        features,
        np.zeros((sample_count, history_width), dtype=np.float32),
        masks,
        teacher_actions,
    )


def _write_imitation_replay_cache(path: Path, replay: _ImitationReplay) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        np.savez(
            temporary,
            features=replay.features,
            masks=replay.masks,
            teacher_actions=replay.teacher_actions,
        )
    temporary_path.replace(path)


def _rollout_imitation_replay(rollout: _Rollout) -> _ImitationReplay:
    return _ImitationReplay(
        rollout.features,
        rollout.action_histories,
        rollout.masks,
        rollout.teacher_actions,
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
        slot_count=config.workers,
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
        np.concatenate([rollout.action_histories for rollout in worker_rollouts]),
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
        sum(rollout.short_cycles for rollout in worker_rollouts),
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
    action_histories: list[FloatArray] = []
    masks: list[BoolArray] = []
    actions: list[int] = []
    log_probabilities: list[float] = []
    values: list[float] = []
    rewards: list[float] = []
    dones: list[bool] = []
    boundary_bootstraps: list[float] = []
    next_deltas: list[FloatArray] = []
    teacher_actions: list[int] = []
    completed_returns: list[float] = []
    short_cycles = 0
    ready_feature: FloatArray | None = None
    for _ in range(config.rollout_length):
        observation, mask = worker.ready()
        feature = ready_feature
        if feature is None:
            feature = encode_observation(
                observation, spec_version=batcher.feature_spec_version
            )
        action_history = encode_action_history(
            worker.history(),
            action_count=batcher.model_config.action_count,
            length=batcher.model_config.action_history_length,
        )
        inference = batcher.infer(worker.worker_index, feature, action_history, mask)
        probabilities = inference.probabilities
        value = inference.value
        action = int(worker.rng.choice(len(probabilities), p=probabilities))
        step = worker.step(ActionIndex(action))
        next_observation = step.observation
        next_feature = encode_observation(
            next_observation, spec_version=batcher.feature_spec_version
        )
        boundary_bootstrap = 0.0
        following_feature = None if step.terminated or step.truncated else next_feature
        if step.truncated and not step.terminated:
            next_history = encode_action_history(
                step.action_history,
                action_count=batcher.model_config.action_count,
                length=batcher.model_config.action_history_length,
            )
            boundary_bootstrap = batcher.infer(
                worker.worker_index, next_feature, next_history, step.action_mask
            ).value
        elif _uses_reset_bootstrap(step, config.return_boundary):
            reset_observation, reset_mask = worker.ready()
            reset_feature = encode_observation(
                reset_observation, spec_version=batcher.feature_spec_version
            )
            reset_history = encode_action_history(
                worker.history(),
                action_count=batcher.model_config.action_count,
                length=batcher.model_config.action_history_length,
            )
            boundary_bootstrap = batcher.infer(
                worker.worker_index, reset_feature, reset_history, reset_mask
            ).value
            following_feature = reset_feature
        features.append(feature)
        action_histories.append(action_history)
        masks.append(mask)
        actions.append(action)
        log_probabilities.append(float(np.log(probabilities[action])))
        values.append(value)
        rewards.append(
            training_reward(
                step.reward,
                decision_cost=config.decision_cost,
                short_cycle_cost=config.short_cycle_cost,
                repeated_state=step.short_cycle,
            )
        )
        short_cycles += int(step.short_cycle)
        dones.append(step.terminated or step.truncated)
        boundary_bootstraps.append(boundary_bootstrap)
        next_deltas.append(next_feature - feature)
        teacher_actions.append(int(teacher.select(observation, mask)))
        if step.completed_return is not None:
            completed_returns.append(step.completed_return)
        ready_feature = following_feature

    final_value = 0.0
    if not dones[-1]:
        _, final_mask = worker.ready()
        if ready_feature is None:
            raise RuntimeError("online worker lost its final encoded observation")
        final_history = encode_action_history(
            worker.history(),
            action_count=batcher.model_config.action_count,
            length=batcher.model_config.action_history_length,
        )
        final_value = batcher.infer(
            worker.worker_index, ready_feature, final_history, final_mask
        ).value
    reward_array = np.asarray(rewards, dtype=np.float32)[:, None]
    value_array = np.asarray(values, dtype=np.float32)[:, None]
    done_array = np.asarray(dones, dtype=np.bool_)[:, None]
    advantages, returns = generalized_advantage_estimate(
        reward_array,
        value_array,
        done_array,
        np.asarray(boundary_bootstraps, dtype=np.float32)[:, None],
        np.asarray([final_value], dtype=np.float32),
        discount=config.discount,
        gae_lambda=config.gae_lambda,
    )
    return _WorkerRollout(
        np.stack(features),
        np.stack(action_histories),
        np.stack(masks),
        np.asarray(actions, dtype=np.int64),
        np.asarray(log_probabilities, dtype=np.float32),
        np.asarray(values, dtype=np.float32),
        advantages.reshape(-1),
        returns.reshape(-1),
        np.stack(next_deltas),
        np.asarray(teacher_actions, dtype=np.int64),
        tuple(completed_returns),
        short_cycles,
    )


def _uses_reset_bootstrap(
    step: _WorkerStep, return_boundary: ReturnBoundaryMode
) -> bool:
    """Keep wins episodic while making other terminal resets continuing."""
    return (
        step.terminated
        and step.terminal_outcome != _WIN_OUTCOME
        and return_boundary is ReturnBoundaryMode.CONTINUING_RESET
    )


def _ppo_update(
    model: SemanticActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: _Rollout,
    *,
    imitation_replay: tuple[_ImitationReplay, ...],
    config: PpoConfig,
    trainable_action_indices: tuple[ActionIndex, ...] | None = None,
    trainable_feature_indices: tuple[int, ...] | None = None,
) -> PpoLosses:
    device = torch.device(config.device)
    tensors = tuple(
        torch.from_numpy(value).to(device)
        for value in (
            rollout.features,
            rollout.action_histories,
            rollout.masks,
            rollout.actions,
            rollout.log_probabilities,
            rollout.advantages,
            rollout.returns,
            rollout.next_deltas,
        )
    )
    features, histories, masks, actions, old_logs, advantages, returns, deltas = tensors
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    replay_features = torch.from_numpy(
        np.concatenate([item.features for item in imitation_replay])
    ).to(device)
    replay_histories = torch.from_numpy(
        np.concatenate([item.action_histories for item in imitation_replay])
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
            logits, values, predicted_deltas = model(
                features[indices], histories[indices]
            )
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
            replay_logits, _, _ = model(
                replay_features[replay_indices], replay_histories[replay_indices]
            )
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
            frozen_weight: Tensor | None = None
            frozen_bias: Tensor | None = None
            frozen_rows: Tensor | None = None
            frozen_encoder_weight: Tensor | None = None
            frozen_feature_columns: Tensor | None = None
            if trainable_action_indices is not None:
                frozen_rows, frozen_feature_columns = _restrict_warmup_gradients(
                    model,
                    trainable_action_indices,
                    trainable_feature_indices=trainable_feature_indices,
                )
                frozen_weight = model.policy_head.weight.detach().clone()
                frozen_bias = model.policy_head.bias.detach().clone()
                if frozen_feature_columns is not None:
                    frozen_encoder_weight = model.input_layer.weight.detach().clone()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            if (
                frozen_weight is not None
                and frozen_bias is not None
                and frozen_rows is not None
            ):
                with torch.no_grad():
                    model.policy_head.weight[frozen_rows].copy_(
                        frozen_weight[frozen_rows]
                    )
                    model.policy_head.bias[frozen_rows].copy_(frozen_bias[frozen_rows])
                    if (
                        frozen_encoder_weight is not None
                        and frozen_feature_columns is not None
                    ):
                        model.input_layer.weight[:, frozen_feature_columns].copy_(
                            frozen_encoder_weight[:, frozen_feature_columns]
                        )
            last_losses = PpoLosses(
                float(policy_loss.item()),
                float(value_loss.item()),
                float(echo_loss.item()),
                float(imitation_loss.item()),
            )
    return last_losses


def _warmup_action_indices(
    established_action_count: int,
    action_count: int,
    menu_keycodes: tuple[Keycode, ...],
    action_kinds: tuple[ActionKind, ...] = (),
) -> tuple[ActionIndex, ...]:
    """Return appended actions and declared rows in their dependent UI flows."""
    return tuple(
        {
            *(
                ActionIndex(index)
                for index in range(established_action_count, action_count)
            ),
            *(
                action_to_index(Action.menu_select(keycode))
                for keycode in menu_keycodes
            ),
            *(action_to_index(Action(kind)) for kind in action_kinds),
        }
    )


def _restrict_warmup_gradients(
    model: SemanticActorCritic,
    action_indices: tuple[ActionIndex, ...],
    *,
    trainable_feature_indices: tuple[int, ...] | None = None,
) -> tuple[Tensor, Tensor | None]:
    """Restrict warmup to declared policy rows and newly appended inputs."""
    for parameter in model.parameters():
        if (
            parameter is not model.policy_head.weight
            and parameter is not model.policy_head.bias
            and not (
                trainable_feature_indices is not None
                and parameter is model.input_layer.weight
            )
        ):
            parameter.grad = None
    frozen_rows = torch.ones(
        model.config.action_count,
        dtype=torch.bool,
        device=model.policy_head.weight.device,
    )
    frozen_rows[list(action_indices)] = False
    if model.policy_head.weight.grad is not None:
        model.policy_head.weight.grad[frozen_rows] = 0
    if model.policy_head.bias.grad is not None:
        model.policy_head.bias.grad[frozen_rows] = 0
    frozen_feature_columns: Tensor | None = None
    if trainable_feature_indices is not None:
        frozen_feature_columns = torch.ones(
            model.input_layer.weight.shape[1],
            dtype=torch.bool,
            device=model.input_layer.weight.device,
        )
        frozen_feature_columns[list(trainable_feature_indices)] = False
        if model.input_layer.weight.grad is not None:
            model.input_layer.weight.grad[:, frozen_feature_columns] = 0
    return frozen_rows, frozen_feature_columns


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
