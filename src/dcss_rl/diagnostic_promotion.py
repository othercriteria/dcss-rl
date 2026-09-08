"""Promote a completed current-diagnostic evaluation without rerunning games."""

import argparse
from pathlib import Path

import msgspec

from dcss_rl.evaluation import EvaluationSummary, load_suite, promote_champion


def promote_diagnostic(summary_path: Path) -> bool:
    summary = msgspec.json.decode(summary_path.read_bytes(), type=EvaluationSummary)
    suite = load_suite(Path("configs/diagnostic-suite.json"))
    if summary.suite_id != suite.suite_id or tuple(
        (episode.case_id, episode.seed) for episode in summary.episodes
    ) != tuple((case.case_id, case.seed) for case in suite.cases):
        raise ValueError("promotion requires the complete current diagnostic suite")
    if any(not Path(episode.trajectory).is_file() for episode in summary.episodes):
        raise ValueError("diagnostic promotion requires retained replay files")
    return promote_champion(summary, Path("artifacts/dev-champion.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    promoted = promote_diagnostic(args.summary)
    print(
        f"Diagnostic leader {'promoted' if promoted else 'retained'}: "
        "artifacts/dev-champion.json"
    )


if __name__ == "__main__":
    main()
