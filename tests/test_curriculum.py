from dataclasses import replace
from pathlib import Path

import pytest

from dcss_rl.curriculum import collect_ability_curriculum, validate_curriculum_suite
from dcss_rl.evaluation import EvaluationCase, load_suite
from dcss_rl.units import GameSeed, StepLimit, WorkerCount


def test_broader_curriculum_is_training_only_and_disjoint_from_ac1() -> None:
    suite = load_suite(Path("configs/ability-curriculum-v2.json"))
    training = load_suite(Path("configs/training-suite.json"))
    original = load_suite(Path("configs/ability-curriculum-v1.json"))
    validate_curriculum_suite(suite, training)
    assert len(suite.cases) == 48
    assert suite.step_limit == 200
    assert {case.seed for case in suite.cases}.isdisjoint(
        case.seed for case in original.cases
    )


@pytest.mark.parametrize(
    "invalid", ["seed", "duplicate_seed", "duplicate_id", "empty", "horizon"]
)
def test_curriculum_rejects_invalid_suite(invalid: str) -> None:
    suite = load_suite(Path("configs/ability-curriculum-v2.json"))
    training = load_suite(Path("configs/training-suite.json"))
    first = suite.cases[0]
    match invalid:
        case "seed":
            suite = replace(suite, cases=(EvaluationCase("outside", GameSeed(-1)),))
        case "duplicate_seed":
            suite = replace(suite, cases=(first, replace(first, case_id="duplicate")))
        case "duplicate_id":
            suite = replace(
                suite, cases=(first, replace(suite.cases[1], case_id=first.case_id))
            )
        case "empty":
            suite = replace(suite, cases=())
        case "horizon":
            suite = replace(suite, step_limit=StepLimit(0))
    with pytest.raises(ValueError):
        validate_curriculum_suite(suite, training)


def test_invalid_workers_fail_before_loading_or_launching(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="workers"):
        collect_ability_curriculum(
            tmp_path / "missing", tmp_path / "output", workers=WorkerCount(0)
        )
    assert not (tmp_path / "output").exists()
