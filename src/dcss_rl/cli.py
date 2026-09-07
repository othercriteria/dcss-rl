"""Command-line workflows for evaluation and local observability."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

from dcss_rl.compatibility import run_compatibility_smoke
from dcss_rl.evaluation import (
    assert_meets_regression_threshold,
    evaluate_policy,
    load_regression_threshold,
    load_suite,
    select_champion,
)
from dcss_rl.policy import ScriptedMibePolicy
from dcss_rl.replay import champion_trajectory, watch_replay
from dcss_rl.units import FrameLimit, GameSeed, Seconds, ViewRadius, WorkerCount


def main() -> None:
    parser = argparse.ArgumentParser(prog="dcss-rl")
    commands = parser.add_subparsers(dest="command", required=True)
    compatibility = commands.add_parser("compatibility-smoke")
    compatibility.add_argument(
        "--binary",
        type=Path,
        default=Path("vendor/crawl/crawl-ref/source/crawl"),
    )
    compatibility.add_argument("--seed", type=int, default=1)
    evaluate = commands.add_parser("evaluate-scripted")
    evaluate.add_argument(
        "--binary",
        type=Path,
        default=Path("vendor/crawl/crawl-ref/source/crawl"),
    )
    evaluate.add_argument("--suite", type=Path, default=Path("configs/heldout-v1.json"))
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--workers", type=int, default=5)
    evaluate.add_argument("--threshold", type=Path)
    evaluate.add_argument(
        "--champion", type=Path, default=Path("artifacts/champion.json")
    )
    watch = commands.add_parser("watch-best")
    watch.add_argument("--champion", type=Path, default=Path("artifacts/champion.json"))
    watch.add_argument("--case")
    watch.add_argument("--frame-delay-seconds", type=float, default=0.1)
    watch.add_argument("--view-radius", type=int, default=10)
    watch.add_argument("--frame-limit", type=int)
    watch.add_argument("--no-animate", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "compatibility-smoke":
        report = run_compatibility_smoke(
            arguments.binary, seed=GameSeed(arguments.seed)
        )
        print(
            f"{report.version}: {report.species} D:{report.depth}, "
            f"{report.visible_cells} visible cells, exact replay"
        )
        return
    if arguments.command == "evaluate-scripted":
        _evaluate_scripted(
            binary=arguments.binary,
            suite_path=arguments.suite,
            output=arguments.output,
            champion_path=arguments.champion,
            workers=WorkerCount(arguments.workers),
            threshold_path=arguments.threshold,
        )
        return
    if arguments.command == "watch-best":
        trajectory = champion_trajectory(arguments.champion, arguments.case)
        watch_replay(
            trajectory,
            frame_delay=Seconds(arguments.frame_delay_seconds),
            view_radius=ViewRadius(arguments.view_radius),
            frame_limit=FrameLimit(arguments.frame_limit)
            if arguments.frame_limit is not None
            else None,
            animate=not arguments.no_animate,
        )
        return
    _unreachable(arguments.command)


def _evaluate_scripted(
    *,
    binary: Path,
    suite_path: Path,
    output: Path | None,
    champion_path: Path,
    workers: WorkerCount,
    threshold_path: Path | None,
) -> None:
    policy = ScriptedMibePolicy()
    suite = load_suite(suite_path)
    if output is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output = Path("artifacts/evaluations") / f"{policy.policy_id}-{timestamp}"
    summary = evaluate_policy(binary, policy, suite, output, workers=workers)
    if threshold_path is not None:
        assert_meets_regression_threshold(
            summary, load_regression_threshold(threshold_path)
        )
    select_champion((summary,), champion_path)
    print(f"evaluation: {output / 'summary.json'}")
    print(f"champion: {champion_path}")
    print(f"rank: {summary.rank}")


def _unreachable(value: object) -> NoReturn:
    raise AssertionError(f"unhandled command: {value!r}")


if __name__ == "__main__":
    main()
