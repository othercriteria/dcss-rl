import json
from pathlib import Path

import pytest

from dcss_rl.evaluation import (
    EpisodeResult,
    EvaluationSummary,
    RegressionThreshold,
    assert_meets_regression_threshold,
    load_regression_threshold,
    load_suite,
    promote_champion,
    select_champion,
)
from dcss_rl.units import GameSeed


def result(
    *, depth: int, xl: int, turns: int, reward: float = 0.0, policy_steps: int = 10
) -> EpisodeResult:
    return EpisodeResult(
        case_id="case",
        seed=GameSeed(1),
        outcome="dead",
        total_reward=reward,
        policy_steps=policy_steps,
        game_turns=turns,
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
    summary = EvaluationSummary(
        "mibe-heldout-v1",
        "candidate",
        "now",
        (result(depth=5, xl=10, turns=3516, reward=50.0, policy_steps=2500),),
    )

    assert_meets_regression_threshold(summary, threshold)


def test_rejects_rank_below_regression_threshold() -> None:
    threshold = RegressionThreshold("suite", "baseline", (0, 0, 2, 1, 1, 0.0))
    summary = EvaluationSummary(
        "suite", "candidate", "now", (result(depth=1, xl=20, turns=9999),)
    )

    with pytest.raises(RuntimeError, match=r"below.*floor"):
        assert_meets_regression_threshold(summary, threshold)


def test_rejects_threshold_for_different_suite() -> None:
    threshold = RegressionThreshold("heldout", "baseline", (0, 0, 1, 1, 1, 0.0))
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


def test_rank_uses_bounded_policy_survival_not_inflatable_game_turns() -> None:
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

    assert survives.rank > rests.rank


def test_promoting_same_policy_migrates_embedded_v1_rank(tmp_path: Path) -> None:
    destination = tmp_path / "champion.json"
    candidate = EvaluationSummary(
        "suite",
        "policy",
        "now",
        (result(depth=2, xl=2, turns=5000, policy_steps=10),),
    )
    select_champion((candidate,), destination)
    manifest = json.loads(destination.read_text())
    del manifest["rank_spec_version"]
    manifest["rank"][4] = 5000
    destination.write_text(json.dumps(manifest))

    assert promote_champion(candidate, destination)
    migrated = json.loads(destination.read_text())
    assert migrated["rank_spec_version"] == 2
    assert migrated["rank"][4] == 10
