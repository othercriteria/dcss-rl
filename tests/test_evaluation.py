import json
from pathlib import Path

import numpy as np
import pytest

from dcss_rl.evaluation import (
    EpisodeResult,
    EvaluationSummary,
    RegressionThreshold,
    _start_episode,
    activate_champion_track,
    assert_meets_regression_threshold,
    load_regression_threshold,
    load_suite,
    promote_champion,
    select_champion,
)
from dcss_rl.schema import EnvironmentInfo, ObservationData
from dcss_rl.trajectory import RecordingEnv
from dcss_rl.units import (
    DecisionProgressArea,
    DepthWeightedDiscovery,
    GameSeed,
    LevelCount,
)


def result(
    *,
    depth: int,
    xl: int,
    turns: int,
    reward: float = 0.0,
    policy_steps: int = 10,
    discovery: int | None = None,
) -> EpisodeResult:
    return EpisodeResult(
        case_id="case",
        seed=GameSeed(1),
        outcome="dead",
        total_reward=reward,
        policy_steps=policy_steps,
        game_turns=turns,
        depth_progress_area=DecisionProgressArea(max(depth - 1, 0) * policy_steps),
        depth_weighted_discovery=DepthWeightedDiscovery(
            discovery if discovery is not None else depth * 10
        ),
        levels_visited=LevelCount(depth),
        max_depth=depth,
        max_xl=xl,
        runes=0,
        trajectory="trajectory.jsonl",
        game_directory="game",
    )


def test_loads_checked_in_heldout_suite() -> None:
    suite = load_suite(Path("configs/heldout-v1.json"))

    assert suite.suite_id == "mibe-heldout-v1"
    assert len(suite.cases) == 5
    assert len({case.seed for case in suite.cases}) == len(suite.cases)


def test_evaluation_retries_startup_in_isolated_attempt_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts: list[Path] = []

    def reset(self: RecordingEnv) -> tuple[ObservationData, EnvironmentInfo]:
        attempts.append(self.env.run_root or Path())
        if len(attempts) < 3:
            raise TimeoutError("transient startup")
        return (
            {
                "player": {},
                "cells": [],
                "messages": [],
                "menu": None,
                "input_mode": 1,
            },
            {
                "action_mask": np.ones(1, dtype=np.bool_),
                "structured_action": None,
                "emitted_keycodes": (),
                "outcome": None,
                "steps": 0,
                "max_depth": 1,
                "max_xl": 1,
            },
        )

    monkeypatch.setattr(RecordingEnv, "reset", reset)
    suite = load_suite(Path("configs/heldout-v5.json"))
    episode_directory = tmp_path / "case"
    episode_directory.mkdir()

    started = _start_episode(
        tmp_path / "crawl",
        suite,
        suite.cases[0],
        episode_directory,
        agent_id="policy",
        checkpoint_id=None,
    )
    started.env.close()

    assert attempts == [
        episode_directory / "attempt-0/game",
        episode_directory / "attempt-1/game",
        episode_directory / "attempt-2/game",
    ]
    assert started.trajectory_path == episode_directory / "attempt-2/trajectory.jsonl"


def test_broad_training_suite_is_unique_and_disjoint_from_evaluation() -> None:
    training = load_suite(Path("configs/online-train-v2.json"))
    broader_training = load_suite(Path("configs/online-train-v3.json"))
    diagnostic = load_suite(Path("configs/diagnostic-v2.json"))
    heldout = load_suite(Path("configs/heldout-v2.json"))
    next_heldout = load_suite(Path("configs/heldout-v3.json"))
    future_heldout = load_suite(Path("configs/heldout-v4.json"))
    successor_heldout = load_suite(Path("configs/heldout-v5.json"))
    training_seeds = {case.seed for case in training.cases}
    broader_training_seeds = {case.seed for case in broader_training.cases}
    evaluation_seeds = {
        case.seed
        for case in (
            *diagnostic.cases,
            *heldout.cases,
            *next_heldout.cases,
            *future_heldout.cases,
            *successor_heldout.cases,
        )
    }

    assert training.suite_id == "online-train-v2"
    assert training.step_limit == 1000
    assert len(training.cases) == 64
    assert len(training_seeds) == len(training.cases)
    assert training_seeds.isdisjoint(evaluation_seeds)
    assert broader_training.suite_id == "online-train-v3"
    assert broader_training.step_limit == 2000
    assert len(broader_training_seeds) == len(broader_training.cases) == 128
    assert broader_training_seeds.isdisjoint(training_seeds | evaluation_seeds)
    assert {case.seed for case in heldout.cases}.isdisjoint(
        case.seed for case in next_heldout.cases
    )
    assert {case.seed for case in (*heldout.cases, *next_heldout.cases)}.isdisjoint(
        case.seed for case in future_heldout.cases
    )


def test_diagnostic_v2_extends_horizon_without_changing_seeds() -> None:
    previous = load_suite(Path("configs/diagnostic-v1.json"))
    current = load_suite(Path("configs/diagnostic-v2.json"))

    assert current.step_limit == 500
    assert tuple(case.seed for case in current.cases) == tuple(
        case.seed for case in previous.cases
    )


def test_champion_selection_uses_documented_lexicographic_rank(
    tmp_path: Path,
) -> None:
    shallow = EvaluationSummary(
        "suite", "shallow", "now", (result(depth=2, xl=4, turns=50),)
    )
    deep = EvaluationSummary("suite", "deep", "now", (result(depth=3, xl=3, turns=40),))
    manifest = tmp_path / "champion.json"

    champion = select_champion((shallow, deep), manifest)

    assert champion.policy_id == "deep"
    assert json.loads(manifest.read_text())["policy_id"] == "deep"


def test_rejects_comparing_different_suites(tmp_path: Path) -> None:
    first = EvaluationSummary("one", "a", "now", (result(depth=1, xl=1, turns=1),))
    second = EvaluationSummary("two", "b", "now", (result(depth=1, xl=1, turns=1),))

    with pytest.raises(ValueError, match="same held-out suite"):
        select_champion((first, second), tmp_path / "champion.json")


def test_checked_in_regression_threshold_matches_frozen_baseline() -> None:
    threshold = load_regression_threshold(Path("configs/heldout-regression-v1.json"))

    assert threshold.minimum_rank == (0, 0, 0, 2500, 5, 10, 50.0)


def test_rejects_rank_below_regression_threshold() -> None:
    threshold = RegressionThreshold("suite", "baseline", (0, 0, 11, 1, 1, 1, 0.0))
    summary = EvaluationSummary(
        "suite", "candidate", "now", (result(depth=1, xl=20, turns=9999),)
    )

    with pytest.raises(RuntimeError, match=r"below.*floor"):
        assert_meets_regression_threshold(summary, threshold)


def test_rejects_threshold_for_different_suite() -> None:
    threshold = RegressionThreshold("heldout", "baseline", (0, 0, 1, 1, 1, 1, 0.0))
    summary = EvaluationSummary(
        "diagnostic", "candidate", "now", (result(depth=2, xl=2, turns=2),)
    )

    with pytest.raises(ValueError, match="threshold is for"):
        assert_meets_regression_threshold(summary, threshold)


def test_champion_promotion_is_monotonic(tmp_path: Path) -> None:
    destination = tmp_path / "champion.json"
    strong = EvaluationSummary(
        "suite", "strong", "now", (result(depth=3, xl=2, turns=20),)
    )
    weak = EvaluationSummary(
        "suite", "weak", "later", (result(depth=2, xl=9, turns=999),)
    )

    assert promote_champion(strong, destination)
    assert not promote_champion(weak, destination)
    assert json.loads(destination.read_text())["policy_id"] == "strong"


def test_rank_ignores_lingering_policy_steps_and_game_turns() -> None:
    rests = EvaluationSummary(
        "suite",
        "rests",
        "now",
        (result(depth=2, xl=2, turns=5000, policy_steps=5),),
    )
    survives = EvaluationSummary(
        "suite",
        "survives",
        "now",
        (result(depth=2, xl=2, turns=50, policy_steps=10),),
    )

    assert survives.rank == rests.rank


def test_rank_rewards_new_depth_weighted_cells_not_lingering() -> None:
    loops = EvaluationSummary(
        "suite",
        "loops",
        "now",
        (result(depth=3, xl=2, turns=5000, policy_steps=500, discovery=100),),
    )
    explores = EvaluationSummary(
        "suite",
        "explores",
        "now",
        (result(depth=3, xl=2, turns=50, policy_steps=50, discovery=101),),
    )

    assert explores.rank > loops.rank


def test_promoting_same_policy_migrates_older_rank_spec(tmp_path: Path) -> None:
    destination = tmp_path / "champion.json"
    candidate = EvaluationSummary(
        "suite",
        "policy",
        "now",
        (result(depth=2, xl=2, turns=5000, policy_steps=10),),
    )
    select_champion((candidate,), destination)
    manifest = json.loads(destination.read_text())
    manifest["rank_spec_version"] = 2
    manifest["rank"] = [0, 0, 2, 2, 10, 0.0]
    destination.write_text(json.dumps(manifest))

    assert promote_champion(candidate, destination)
    migrated = json.loads(destination.read_text())
    assert migrated["rank_spec_version"] == 5
    assert migrated["rank"][2] == 20


def test_activates_validated_champion_track_and_archives_previous(
    tmp_path: Path,
) -> None:
    old = tmp_path / "champion.json"
    candidate = tmp_path / "candidate.json"
    archive = tmp_path / "archive" / "champion-v1.json"
    old.write_text('{"suite_id": "old"}')
    candidate.write_text(
        json.dumps(
            {
                "suite_id": "new",
                "policy_id": "winner",
                "rank_spec_version": 5,
            }
        )
    )

    activation = activate_champion_track(
        candidate,
        old,
        expected_suite_id="new",
        archive_manifest=archive,
    )

    assert activation.policy_id == "winner"
    assert json.loads(old.read_text())["suite_id"] == "new"
    assert json.loads(archive.read_text())["suite_id"] == "old"
