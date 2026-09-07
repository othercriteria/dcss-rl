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
    promote_champion,
)
from dcss_rl.policy import Policy, ScriptedMibePolicy
from dcss_rl.replay import champion_trajectory, watch_grid, watch_replay
from dcss_rl.units import (
    BatchSize,
    EpochCount,
    FrameLimit,
    GameSeed,
    LearningRate,
    LossWeight,
    Probability,
    RolloutLength,
    Seconds,
    UpdateCount,
    ViewRadius,
    WorkerCount,
)


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
    learned = commands.add_parser("evaluate-learned")
    learned.add_argument("--checkpoint", type=Path, required=True)
    learned.add_argument(
        "--binary",
        type=Path,
        default=Path("vendor/crawl/crawl-ref/source/crawl"),
    )
    learned.add_argument(
        "--suite", type=Path, default=Path("configs/diagnostic-v1.json")
    )
    learned.add_argument("--output", type=Path)
    learned.add_argument("--workers", type=int, default=5)
    learned.add_argument("--threshold", type=Path)
    learned.add_argument(
        "--champion", type=Path, default=Path("artifacts/champion.json")
    )
    learned.add_argument("--device", default="cuda")
    learned.add_argument("--fallback-scripted", action="store_true")
    learned.add_argument("--confidence-threshold", type=float, default=0.95)
    train = commands.add_parser("train-imitation")
    train.add_argument("trajectories", type=Path, nargs="+")
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--policy-id", default="semantic-bc-echo-v1")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--echo-weight", type=float, default=0.1)
    train.add_argument("--value-weight", type=float, default=0.1)
    train.add_argument("--seed", type=int, default=1)
    train.add_argument("--device", default="cuda")
    train.add_argument("--relabel-scripted", action="store_true")
    ppo = commands.add_parser("train-ppo")
    ppo.add_argument("--initial-checkpoint", type=Path, required=True)
    ppo.add_argument("--checkpoint", type=Path, required=True)
    ppo.add_argument("--policy-id", required=True)
    ppo.add_argument(
        "--binary",
        type=Path,
        default=Path("vendor/crawl/crawl-ref/source/crawl"),
    )
    ppo.add_argument("--suite", type=Path, default=Path("configs/diagnostic-v1.json"))
    ppo.add_argument("--run-root", type=Path, default=Path("artifacts/ppo-runs"))
    ppo.add_argument("--updates", type=int, default=4)
    ppo.add_argument("--rollout-length", type=int, default=128)
    ppo.add_argument("--workers", type=int, default=5)
    ppo.add_argument("--minibatch-size", type=int, default=256)
    ppo.add_argument("--learning-rate", type=float, default=1e-4)
    ppo.add_argument("--echo-weight", type=float, default=0.1)
    ppo.add_argument("--value-weight", type=float, default=0.5)
    ppo.add_argument("--entropy-weight", type=float, default=0.01)
    ppo.add_argument("--imitation-weight", type=float, default=0.1)
    ppo.add_argument("--epochs-per-update", type=int, default=4)
    ppo.add_argument("--clip-ratio", type=float, default=0.2)
    ppo.add_argument("--seed", type=int, default=1)
    ppo.add_argument("--device", default="cuda")
    watch = commands.add_parser("watch-best")
    watch.add_argument("--champion", type=Path, default=Path("artifacts/champion.json"))
    watch.add_argument("--case")
    watch.add_argument("--frame-delay-seconds", type=float, default=0.1)
    watch.add_argument("--view-radius", type=int, default=10)
    watch.add_argument("--frame-limit", type=int)
    watch.add_argument("--no-animate", action="store_true")
    grid = commands.add_parser("watch-grid")
    grid.add_argument("--champion", type=Path, default=Path("artifacts/champion.json"))
    grid.add_argument("--frame-delay-seconds", type=float, default=0.1)
    grid.add_argument("--view-radius", type=int, default=5)
    grid.add_argument("--columns", type=int, default=3)
    grid.add_argument("--frame-limit", type=int)
    grid.add_argument("--no-animate", action="store_true")
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
    if arguments.command == "evaluate-learned":
        from dcss_rl.learned import ConfidenceGatedPolicy, LearnedPolicy

        policy: Policy = LearnedPolicy(arguments.checkpoint, device=arguments.device)
        if arguments.fallback_scripted:
            policy = ConfidenceGatedPolicy(
                policy,
                ScriptedMibePolicy(),
                threshold=Probability(arguments.confidence_threshold),
            )

        _evaluate(
            policy=policy,
            binary=arguments.binary,
            suite_path=arguments.suite,
            output=arguments.output,
            champion_path=arguments.champion,
            workers=WorkerCount(arguments.workers),
            threshold_path=arguments.threshold,
        )
        if isinstance(policy, ConfidenceGatedPolicy):
            print(f"learned decision fraction: {policy.learned_fraction:.3f}")
        return
    if arguments.command == "train-imitation":
        from dcss_rl.training import TrainingConfig, train_imitation

        report = train_imitation(
            tuple(arguments.trajectories),
            arguments.checkpoint,
            config=TrainingConfig(
                seed=arguments.seed,
                epochs=EpochCount(arguments.epochs),
                batch_size=BatchSize(arguments.batch_size),
                learning_rate=LearningRate(arguments.learning_rate),
                echo_weight=LossWeight(arguments.echo_weight),
                value_weight=LossWeight(arguments.value_weight),
                device=arguments.device,
                relabel_with_scripted=arguments.relabel_scripted,
            ),
            policy_id=arguments.policy_id,
        )
        print(
            f"checkpoint: {report.checkpoint}; samples={report.sample_count}; "
            f"validation_accuracy={report.validation_accuracy:.3f}; "
            f"validation_policy_loss={report.validation_policy_loss:.3f}"
        )
        return
    if arguments.command == "train-ppo":
        from dcss_rl.ppo import PpoConfig, PpoUpdateReport, train_ppo

        def report_progress(update: PpoUpdateReport) -> None:
            print(
                f"update {update.update}: decisions={update.decisions}; "
                f"rate={update.decision_rate:.2f}/s; "
                f"episodes={update.completed_episodes}; "
                f"mean_return={update.mean_completed_return:.3f}; "
                f"losses={update.policy_loss:.3f}/{update.value_loss:.3f}/"
                f"{update.echo_loss:.3f}",
                flush=True,
            )

        report = train_ppo(
            arguments.binary,
            load_suite(arguments.suite),
            arguments.initial_checkpoint,
            arguments.checkpoint,
            arguments.run_root,
            config=PpoConfig(
                seed=arguments.seed,
                updates=UpdateCount(arguments.updates),
                rollout_length=RolloutLength(arguments.rollout_length),
                workers=WorkerCount(arguments.workers),
                minibatch_size=BatchSize(arguments.minibatch_size),
                learning_rate=LearningRate(arguments.learning_rate),
                echo_weight=LossWeight(arguments.echo_weight),
                value_weight=LossWeight(arguments.value_weight),
                entropy_weight=LossWeight(arguments.entropy_weight),
                imitation_weight=LossWeight(arguments.imitation_weight),
                epochs_per_update=EpochCount(arguments.epochs_per_update),
                clip_ratio=Probability(arguments.clip_ratio),
                device=arguments.device,
            ),
            policy_id=arguments.policy_id,
            progress=report_progress,
        )
        print(
            f"checkpoint: {report.checkpoint}; decisions={report.decisions}; "
            f"episodes={report.completed_episodes}; "
            f"mean_return={report.mean_episode_return:.3f}; "
            f"policy_loss={report.policy_loss:.3f}; "
            f"value_loss={report.value_loss:.3f}; echo_loss={report.echo_loss:.3f}"
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
    if arguments.command == "watch-grid":
        watch_grid(
            arguments.champion,
            frame_delay=Seconds(arguments.frame_delay_seconds),
            view_radius=ViewRadius(arguments.view_radius),
            columns=arguments.columns,
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
    _evaluate(
        policy=ScriptedMibePolicy(),
        binary=binary,
        suite_path=suite_path,
        output=output,
        champion_path=champion_path,
        workers=workers,
        threshold_path=threshold_path,
    )


def _evaluate(
    *,
    policy: Policy,
    binary: Path,
    suite_path: Path,
    output: Path | None,
    champion_path: Path,
    workers: WorkerCount,
    threshold_path: Path | None,
) -> None:
    suite = load_suite(suite_path)
    if output is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output = Path("artifacts/evaluations") / f"{policy.policy_id}-{timestamp}"
    summary = evaluate_policy(binary, policy, suite, output, workers=workers)
    if threshold_path is not None:
        assert_meets_regression_threshold(
            summary, load_regression_threshold(threshold_path)
        )
    promoted = promote_champion(summary, champion_path)
    print(f"evaluation: {output / 'summary.json'}")
    print(f"champion: {champion_path} ({'promoted' if promoted else 'retained'})")
    print(f"rank: {summary.rank}")
    print(
        f"throughput: {summary.decision_rate:.2f} decisions/s "
        f"over {summary.wall_seconds:.2f}s"
    )


def _unreachable(value: object) -> NoReturn:
    raise AssertionError(f"unhandled command: {value!r}")


if __name__ == "__main__":
    main()
