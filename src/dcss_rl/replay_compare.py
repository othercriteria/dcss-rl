"""Compare complete manifest-ordered action sequences, independently of rank."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import NewType

import msgspec

from dcss_rl.units import ActionCount, ActionIndex, CaseId, GameSeed, SuiteId

DecisionIndex = NewType("DecisionIndex", int)


@dataclass(frozen=True, slots=True)
class _EpisodeReference:
    case_id: CaseId
    seed: GameSeed
    trajectory: str


@dataclass(frozen=True, slots=True)
class _Summary:
    suite_id: SuiteId
    episodes: tuple[_EpisodeReference, ...]


@dataclass(frozen=True, slots=True)
class _ActionRecord:
    action_index: ActionIndex


@dataclass(frozen=True, slots=True)
class ActionEdit:
    operation: str
    reference_start: DecisionIndex
    reference_stop: DecisionIndex
    candidate_start: DecisionIndex
    candidate_stop: DecisionIndex
    reference_actions: tuple[ActionIndex, ...]
    candidate_actions: tuple[ActionIndex, ...]


@dataclass(frozen=True, slots=True)
class CaseComparison:
    case_id: CaseId
    reference_decisions: ActionCount
    candidate_decisions: ActionCount
    edits: tuple[ActionEdit, ...]


def _actions(path: Path) -> tuple[ActionIndex, ...]:
    with path.open() as source:
        next(source)  # The first record is trajectory metadata and initial state.
        return tuple(
            msgspec.json.decode(line, type=_ActionRecord).action_index
            for line in source
        )


def compare_actions(
    reference: tuple[ActionIndex, ...], candidate: tuple[ActionIndex, ...]
) -> tuple[ActionEdit, ...]:
    """Report insertions separately, so one cancel does not shift every later row."""
    return tuple(
        ActionEdit(
            operation,
            DecisionIndex(old_start),
            DecisionIndex(old_stop),
            DecisionIndex(new_start),
            DecisionIndex(new_stop),
            reference[old_start:old_stop],
            candidate[new_start:new_stop],
        )
        for operation, old_start, old_stop, new_start, new_stop in SequenceMatcher(
            None, reference, candidate, autojunk=False
        ).get_opcodes()
        if operation != "equal"
    )


def compare_replays(reference: Path, candidate: Path) -> tuple[CaseComparison, ...]:
    before = msgspec.json.decode(reference.read_bytes(), type=_Summary)
    after = msgspec.json.decode(candidate.read_bytes(), type=_Summary)
    if before.suite_id != after.suite_id:
        raise ValueError("action comparisons require the same suite")
    before_ids = {episode.case_id for episode in before.episodes}
    after_cases = {episode.case_id: episode for episode in after.episodes}
    if (
        len(before_ids) != len(before.episodes)
        or len(after_cases) != len(after.episodes)
        or before_ids != after_cases.keys()
    ):
        raise ValueError("action comparisons require identical, unique case sets")
    result: list[CaseComparison] = []
    for episode in before.episodes:
        other = after_cases[episode.case_id]
        if episode.seed != other.seed:
            raise ValueError(f"case {episode.case_id} has mismatched seeds")
        old, new = _actions(Path(episode.trajectory)), _actions(Path(other.trajectory))
        result.append(
            CaseComparison(
                episode.case_id,
                ActionCount(len(old)),
                ActionCount(len(new)),
                compare_actions(old, new),
            )
        )
    return tuple(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="summary.json")
    parser.add_argument("--candidate", type=Path, required=True, help="summary.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-identical", action="store_true")
    args = parser.parse_args()
    comparisons = compare_replays(args.reference, args.candidate)
    report = json.dumps([asdict(case) for case in comparisons], indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output:
            output.write(report)
    print(report, end="")
    if args.require_identical and any(case.edits for case in comparisons):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
