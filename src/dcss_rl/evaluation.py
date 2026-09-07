"""Fixed-suite evaluation and champion selection for DCSS policies."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import cast

import numpy as np

from dcss_rl.env import DcssEnv
from dcss_rl.policy import Policy
from dcss_rl.schema import JsonObject
from dcss_rl.trajectory import RecordingEnv, TrajectoryWriter
from dcss_rl.units import DecisionsPerSecond, GameSeed, Seconds, StepLimit, WorkerCount
from dcss_rl.webtiles import GameConfig

_DEFAULT_EVALUATION_WORKERS = WorkerCount(1)
_ZERO_SECONDS = Seconds(0.0)
type EvaluationRank = tuple[int, int, int, int, int, float]
RANK_SPEC_VERSION = 2


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    seed: GameSeed


@dataclass(frozen=True, slots=True)
class EvaluationSuite:
    suite_id: str
    step_limit: StepLimit
    cases: tuple[EvaluationCase, ...]


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    case_id: str
    seed: GameSeed
    outcome: str
    total_reward: float
    policy_steps: int
    game_turns: int
    max_depth: int
    max_xl: int
    runes: int
    trajectory: str
    game_directory: str


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    suite_id: str
    policy_id: str
    created_at: str
    episodes: tuple[EpisodeResult, ...]
    wall_seconds: Seconds = _ZERO_SECONDS

    @property
    def rank(self) -> EvaluationRank:
        """Lexicographic champion order, with ascension dominating all milestones."""
        return (
            sum(episode.outcome == "won" for episode in self.episodes),
            sum(episode.runes for episode in self.episodes),
            sum(episode.max_depth for episode in self.episodes),
            sum(episode.max_xl for episode in self.episodes),
            sum(episode.policy_steps for episode in self.episodes),
            sum(episode.total_reward for episode in self.episodes),
        )

    @property
    def decision_rate(self) -> DecisionsPerSecond:
        decisions = sum(episode.policy_steps for episode in self.episodes)
        if self.wall_seconds <= 0:
            return DecisionsPerSecond(0.0)
        return DecisionsPerSecond(decisions / self.wall_seconds)


@dataclass(frozen=True, slots=True)
class RegressionThreshold:
    """Minimum acceptable champion rank for one fixed evaluation suite."""

    suite_id: str
    baseline_policy_id: str
    minimum_rank: EvaluationRank


def load_suite(path: Path) -> EvaluationSuite:
    """Load and validate a checked-in held-out suite at the JSON boundary."""
    decoded: object = json.loads(Path(path).read_text())
    if not isinstance(decoded, dict):
        raise ValueError("evaluation suite must be a JSON object")
    suite_id = decoded.get("suite_id")
    step_limit = decoded.get("step_limit")
    raw_cases = decoded.get("cases")
    if not isinstance(suite_id, str) or not isinstance(step_limit, int):
        raise ValueError("suite_id must be a string and step_limit an integer")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("evaluation suite cases must be a non-empty list")
    cases: list[EvaluationCase] = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise ValueError("each evaluation case must be an object")
        case_id = raw_case.get("case_id")
        seed = raw_case.get("seed")
        if not isinstance(case_id, str) or not isinstance(seed, int):
            raise ValueError("each evaluation case needs a string case_id and int seed")
        cases.append(EvaluationCase(case_id, GameSeed(seed)))
    return EvaluationSuite(suite_id, StepLimit(step_limit), tuple(cases))


def load_regression_threshold(path: Path) -> RegressionThreshold:
    """Load a checked-in aggregate regression floor."""
    decoded: object = json.loads(Path(path).read_text())
    if not isinstance(decoded, dict):
        raise ValueError("regression threshold must be a JSON object")
    suite_id = decoded.get("suite_id")
    baseline_policy_id = decoded.get("baseline_policy_id")
    raw_rank = decoded.get("minimum_rank")
    if not isinstance(suite_id, str) or not isinstance(baseline_policy_id, str):
        raise ValueError("regression threshold needs suite and baseline policy IDs")
    if (
        not isinstance(raw_rank, list)
        or len(raw_rank) != 6
        or any(not isinstance(value, (int, float)) for value in raw_rank)
    ):
        raise ValueError("minimum_rank must contain six numeric metrics")
    return RegressionThreshold(
        suite_id,
        baseline_policy_id,
        (
            int(raw_rank[0]),
            int(raw_rank[1]),
            int(raw_rank[2]),
            int(raw_rank[3]),
            int(raw_rank[4]),
            float(raw_rank[5]),
        ),
    )


def assert_meets_regression_threshold(
    summary: EvaluationSummary, threshold: RegressionThreshold
) -> None:
    """Reject a candidate whose held-out rank falls below the locked baseline."""
    if summary.suite_id != threshold.suite_id:
        raise ValueError(
            f"threshold is for {threshold.suite_id!r}, not {summary.suite_id!r}"
        )
    if summary.rank < threshold.minimum_rank:
        raise RuntimeError(
            f"policy {summary.policy_id!r} rank {summary.rank} is below "
            f"{threshold.baseline_policy_id!r} floor {threshold.minimum_rank}"
        )


def evaluate_policy(
    binary: Path,
    policy: Policy,
    suite: EvaluationSuite,
    output_directory: Path,
    *,
    workers: WorkerCount = _DEFAULT_EVALUATION_WORKERS,
) -> EvaluationSummary:
    """Run one policy over every fixed case and persist auditable artifacts."""
    output_directory.mkdir(parents=True, exist_ok=False)
    if workers < 1:
        raise ValueError("evaluation workers must be positive")
    started = perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        episodes = tuple(
            executor.map(
                lambda case: _run_episode(
                    binary, policy, suite, case, output_directory
                ),
                suite.cases,
            )
        )
    summary = EvaluationSummary(
        suite.suite_id,
        policy.policy_id,
        datetime.now(UTC).isoformat(),
        episodes,
        Seconds(perf_counter() - started),
    )
    _write_json(output_directory / "summary.json", _summary_data(summary))
    return summary


def select_champion(
    candidates: tuple[EvaluationSummary, ...], destination: Path
) -> EvaluationSummary:
    """Select by the documented fixed ordering and write the champion manifest."""
    if not candidates:
        raise ValueError("champion selection requires at least one candidate")
    suite_ids = {candidate.suite_id for candidate in candidates}
    if len(suite_ids) != 1:
        raise ValueError("champion candidates must use the same held-out suite")
    champion = max(
        candidates, key=lambda candidate: (candidate.rank, candidate.policy_id)
    )
    _write_json(
        destination,
        {
            "schema_version": 1,
            "rank_spec_version": RANK_SPEC_VERSION,
            "suite_id": champion.suite_id,
            "policy_id": champion.policy_id,
            "rank": list(champion.rank),
            "summary": _summary_data(champion),
        },
    )
    return champion


def promote_champion(candidate: EvaluationSummary, destination: Path) -> bool:
    """Promote only when a candidate outranks the existing same-suite champion."""
    destination = Path(destination)
    if destination.exists():
        decoded: object = json.loads(destination.read_text())
        if not isinstance(decoded, dict):
            raise ValueError("champion manifest must be a JSON object")
        suite_id = decoded.get("suite_id")
        existing_policy_id = decoded.get("policy_id")
        raw_rank = decoded.get("rank")
        if suite_id != candidate.suite_id:
            raise ValueError(
                f"champion track is for {suite_id!r}, not {candidate.suite_id!r}"
            )
        if not isinstance(raw_rank, list) or len(raw_rank) != 6:
            raise ValueError("champion manifest has an invalid rank")
        existing_rank = (
            _stored_rank(raw_rank)
            if decoded.get("rank_spec_version") == RANK_SPEC_VERSION
            else _migrate_v1_rank(decoded)
        )
        if candidate.rank < existing_rank or (
            candidate.rank == existing_rank
            and existing_policy_id != candidate.policy_id
        ):
            return False
    select_champion((candidate,), destination)
    return True


def _stored_rank(raw_rank: list[object]) -> EvaluationRank:
    return (
        int(_number(raw_rank[0])),
        int(_number(raw_rank[1])),
        int(_number(raw_rank[2])),
        int(_number(raw_rank[3])),
        int(_number(raw_rank[4])),
        float(_number(raw_rank[5])),
    )


def _migrate_v1_rank(manifest: dict[object, object]) -> EvaluationRank:
    """Recompute bounded-survival rank from episode data embedded in v1 manifests."""
    summary = manifest.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("v1 champion manifest lacks an embedded summary")
    episodes = cast(dict[str, object], summary).get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("v1 champion manifest lacks embedded episodes")
    required = (
        "outcome",
        "runes",
        "max_depth",
        "max_xl",
        "policy_steps",
        "total_reward",
    )
    if any(
        not isinstance(episode, dict) or any(field not in episode for field in required)
        for episode in episodes
    ):
        raise ValueError("v1 champion episodes cannot be migrated to rank v2")
    typed_episodes = cast(list[dict[str, object]], episodes)
    return (
        sum(_text(episode["outcome"]) == "won" for episode in typed_episodes),
        sum(int(_number(episode["runes"])) for episode in typed_episodes),
        sum(int(_number(episode["max_depth"])) for episode in typed_episodes),
        sum(int(_number(episode["max_xl"])) for episode in typed_episodes),
        sum(int(_number(episode["policy_steps"])) for episode in typed_episodes),
        sum(float(_number(episode["total_reward"])) for episode in typed_episodes),
    )


def _number(value: object) -> int | float:
    if not isinstance(value, (int, float)):
        raise ValueError("champion rank field must be numeric")
    return value


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("champion outcome must be text")
    return value


def _run_episode(
    binary: Path,
    policy: Policy,
    suite: EvaluationSuite,
    case: EvaluationCase,
    output_directory: Path,
) -> EpisodeResult:
    episode_directory = output_directory / case.case_id
    episode_directory.mkdir()
    trajectory_path = episode_directory / "trajectory.jsonl"
    base = DcssEnv(
        binary,
        game_config=GameConfig(seed=case.seed),
        max_steps=suite.step_limit,
        run_root=episode_directory / "game",
    )
    env = RecordingEnv(
        base,
        TrajectoryWriter(trajectory_path),
        agent_id=policy.policy_id,
        checkpoint_id=policy.checkpoint_id,
    )
    total_reward = 0.0
    outcome = "unknown"
    observation, info = env.reset()
    try:
        while True:
            mask = info.get("action_mask")
            if not isinstance(mask, np.ndarray):
                raise RuntimeError("environment did not provide an ndarray action mask")
            action = policy.select(observation, mask)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                raw_outcome = info.get("outcome")
                outcome = raw_outcome if isinstance(raw_outcome, str) else "truncated"
                break
        player = observation["player"]
        return EpisodeResult(
            case.case_id,
            case.seed,
            outcome,
            total_reward,
            _required_int(info, "steps"),
            player.get("turn", 0),
            _required_int(info, "max_depth"),
            _required_int(info, "max_xl"),
            0,
            str(trajectory_path),
            str(episode_directory / "game"),
        )
    finally:
        env.close()


def _required_int(values: dict[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int):
        raise RuntimeError(f"environment info field {key!r} is not an integer")
    return value


def _summary_data(summary: EvaluationSummary) -> JsonObject:
    return cast(JsonObject, asdict(summary))


def _write_json(path: Path, value: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
