import json
from pathlib import Path

import pytest

from dcss_rl.evaluation import (
    EpisodeResult,
    EvaluationSummary,
    load_suite,
    select_champion,
)
from dcss_rl.units import GameSeed


def result(*, depth: int, xl: int, turns: int) -> EpisodeResult:
    return EpisodeResult(
        case_id="case",
        seed=GameSeed(1),
        outcome="dead",
        total_reward=0.0,
        policy_steps=10,
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
