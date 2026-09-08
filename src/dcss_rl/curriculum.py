"""Deliberate visible ability-menu exposure for training, never promotion."""

from pathlib import Path

import numpy as np

from dcss_rl.actions import Action, ActionKind
from dcss_rl.env import action_to_index
from dcss_rl.evaluation import EvaluationSuite, evaluate_policy, load_suite
from dcss_rl.policy import ActionHistory, ScriptedMibePolicy
from dcss_rl.schema import ObservationData
from dcss_rl.units import ActionIndex, WorkerCount

_DEFAULT_WORKERS = WorkerCount(8)


class AbilityCurriculumPolicy:
    """Occasionally open the ability menu; use the teacher for visible choices.

    Opening during active/cooldown states deliberately collects cancel targets.
    Executed exploration actions are relabeled by the ordinary teacher at training
    time. This collector is not a candidate policy or a tactical safety mask.
    """

    policy_id = "ability-curriculum-v1"
    checkpoint_id = None

    def select(
        self,
        observation: ObservationData,
        action_mask: np.ndarray,
        action_history: ActionHistory = (),
    ) -> ActionIndex:
        abilities = action_to_index(Action(ActionKind.ABILITIES))
        if (
            observation["menu"] is None
            and len(action_history) % 8 == 0
            and action_mask[abilities]
        ):
            return abilities
        return ScriptedMibePolicy().select(observation, action_mask, action_history)


def validate_curriculum_suite(
    suite: EvaluationSuite, training: EvaluationSuite
) -> None:
    if suite.step_limit <= 0 or not suite.cases:
        raise ValueError("ability curriculum requires cases and a positive step limit")
    if len({case.seed for case in suite.cases}) != len(suite.cases) or len(
        {case.case_id for case in suite.cases}
    ) != len(suite.cases):
        raise ValueError("ability curriculum requires unique seeds and case IDs")
    if not {case.seed for case in suite.cases} <= {
        case.seed for case in training.cases
    }:
        raise ValueError("ability curriculum must use only existing training seeds")


def collect_ability_curriculum(
    binary: Path,
    output: Path,
    *,
    suite_path: Path = Path("configs/ability-curriculum-v1.json"),
    workers: WorkerCount = _DEFAULT_WORKERS,
) -> None:
    if workers < 1:
        raise ValueError("ability curriculum workers must be positive")
    suite = load_suite(suite_path)
    training = load_suite(Path("configs/training-suite.json"))
    validate_curriculum_suite(suite, training)
    summary = evaluate_policy(
        binary,
        AbilityCurriculumPolicy(),
        suite,
        output,
        workers=workers,
    )
    print(f"Collected {suite.suite_id}: {len(summary.episodes)} episodes")
