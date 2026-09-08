"""Command-line workflows for evaluation and local observability."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

from dcss_rl.compatibility import run_compatibility_smoke
from dcss_rl.evaluation import (
    EvaluationProgress,
    activate_champion_track,
    assert_meets_regression_threshold,
    evaluate_policy,
    load_regression_threshold,
    load_suite,
    promote_champion,
)
from dcss_rl.policy import Policy, ScriptedMibePolicy
from dcss_rl.replay import (
    load_champion_manifest,
    replay_identity,
    select_champion_episode,
    watch_grid,
    watch_replay,
)
from dcss_rl.returns import ReturnBoundaryMode
from dcss_rl.units import (
    ActionHistoryLength,
    BatchSize,
    DecisionCost,
    DecisionWindow,
    EpochCount,
    FrameLimit,
    GameSeed,
    GridColumnCount,
    InferenceBatchSize,
    Keycode,
    LearningRate,
    LossWeight,
    Probability,
    RewardWeight,
    RolloutLength,
    Seconds,
    ShortCycleCost,
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
    evaluate.add_argument("--suite", type=Path, default=Path("configs/heldout-v2.json"))
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--workers", type=int, default=5)
    evaluate.add_argument("--threshold", type=Path)
    evaluate.add_argument("--champion", type=Path)
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
    learned.add_argument("--champion", type=Path)
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
    train.add_argument("--teacher-balance-exponent", type=float, default=0.5)
    train.add_argument("--seed", type=int, default=1)
    train.add_argument("--device", default="cuda")
    train.add_argument("--relabel-scripted", action="store_true")
    ppo = commands.add_parser("train-ppo")
    ppo.add_argument("--initial-checkpoint", type=Path, required=True)
    ppo.add_argument("--checkpoint", type=Path, required=True)
    ppo.add_argument("--update-checkpoint-directory", type=Path)
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
    ppo.add_argument("--action-history-length", type=int)
    ppo.add_argument("--new-action-warmup-updates", type=int, default=0)
    ppo.add_argument(
        "--imitation-trajectory-root",
        action="append",
        type=Path,
        default=None,
        help="recursively preload and relabel trajectory.jsonl files for DAgger replay",
    )
    ppo.add_argument(
        "--new-action-warmup-menu-key",
        action="append",
        type=_menu_keycode,
        default=None,
        help="menu key whose policy row joins appended-action warmup",
    )
    ppo.add_argument("--workers", type=int, default=5)
    ppo.add_argument("--inference-batch-size", type=int, default=64)
    ppo.add_argument("--inference-batch-wait-seconds", type=float, default=0.001)
    ppo.add_argument("--minibatch-size", type=int, default=256)
    ppo.add_argument("--learning-rate", type=float, default=1e-4)
    ppo.add_argument("--echo-weight", type=float, default=0.1)
    ppo.add_argument("--policy-weight", type=float, default=1.0)
    ppo.add_argument("--value-weight", type=float, default=0.5)
    ppo.add_argument("--entropy-weight", type=float, default=0.01)
    ppo.add_argument("--imitation-weight", type=float, default=0.1)
    ppo.add_argument(
        "--aggregate-imitation-replay",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ppo.add_argument("--teacher-balance-exponent", type=float, default=0.5)
    ppo.add_argument("--epochs-per-update", type=int, default=4)
    ppo.add_argument("--clip-ratio", type=float, default=0.2)
    ppo.add_argument("--explored-cell-reward", type=float, default=0.0)
    ppo.add_argument("--depth-progress-reward", type=float, default=0.0)
    ppo.add_argument("--experience-progress-reward", type=float, default=0.0)
    ppo.add_argument("--hp-fraction-reward", type=float, default=0.0)
    ppo.add_argument(
        "--return-boundary",
        choices=tuple(ReturnBoundaryMode),
        type=ReturnBoundaryMode,
        default=ReturnBoundaryMode.EPISODIC,
    )
    ppo.add_argument("--decision-cost", type=float, default=0.0)
    ppo.add_argument("--short-cycle-cost", type=float, default=0.0)
    ppo.add_argument("--short-cycle-window", type=int, default=8)
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
    activate = commands.add_parser("activate-champion-track")
    activate.add_argument("--candidate", type=Path, required=True)
    activate.add_argument("--champion", type=Path, required=True)
    activate.add_argument("--suite", type=Path, required=True)
    activate.add_argument("--archive", type=Path)
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
                teacher_balance_exponent=Probability(
                    arguments.teacher_balance_exponent
                ),
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
                f"seconds={update.collection_seconds:.2f}/"
                f"{update.optimization_seconds:.2f}/{update.checkpoint_seconds:.2f}; "
                f"inference_batches={update.inference_batches}; "
                f"mean_batch={update.mean_inference_batch_size:.2f}; "
                f"episodes={update.completed_episodes}; "
                f"mean_return={update.mean_completed_return:.3f}; "
                f"losses={update.policy_loss:.3f}/{update.value_loss:.3f}/"
                f"{update.echo_loss:.3f}/{update.imitation_loss:.3f}; "
                f"teacher_agreement={update.teacher_agreement:.3f}; "
                f"short_cycles={update.short_cycles}",
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
                action_history_length=ActionHistoryLength(
                    arguments.action_history_length
                )
                if arguments.action_history_length is not None
                else None,
                new_action_warmup_updates=UpdateCount(
                    arguments.new_action_warmup_updates
                ),
                new_action_warmup_menu_keycodes=tuple(
                    arguments.new_action_warmup_menu_key or ()
                ),
                imitation_trajectories=tuple(
                    trajectory
                    for root in (arguments.imitation_trajectory_root or ())
                    for trajectory in sorted(root.rglob("trajectory.jsonl"))
                ),
                workers=WorkerCount(arguments.workers),
                inference_batch_size=InferenceBatchSize(arguments.inference_batch_size),
                inference_batch_wait=Seconds(arguments.inference_batch_wait_seconds),
                minibatch_size=BatchSize(arguments.minibatch_size),
                learning_rate=LearningRate(arguments.learning_rate),
                echo_weight=LossWeight(arguments.echo_weight),
                policy_weight=LossWeight(arguments.policy_weight),
                value_weight=LossWeight(arguments.value_weight),
                entropy_weight=LossWeight(arguments.entropy_weight),
                imitation_weight=LossWeight(arguments.imitation_weight),
                aggregate_imitation_replay=arguments.aggregate_imitation_replay,
                teacher_balance_exponent=Probability(
                    arguments.teacher_balance_exponent
                ),
                epochs_per_update=EpochCount(arguments.epochs_per_update),
                clip_ratio=Probability(arguments.clip_ratio),
                explored_cell_reward=RewardWeight(arguments.explored_cell_reward),
                depth_progress_reward=RewardWeight(arguments.depth_progress_reward),
                experience_progress_reward=RewardWeight(
                    arguments.experience_progress_reward
                ),
                hp_fraction_reward=RewardWeight(arguments.hp_fraction_reward),
                return_boundary=arguments.return_boundary,
                decision_cost=DecisionCost(arguments.decision_cost),
                short_cycle_cost=ShortCycleCost(arguments.short_cycle_cost),
                short_cycle_window=DecisionWindow(arguments.short_cycle_window),
                device=arguments.device,
            ),
            policy_id=arguments.policy_id,
            update_checkpoint_directory=arguments.update_checkpoint_directory,
            progress=report_progress,
        )
        print(
            f"checkpoint: {report.checkpoint}; decisions={report.decisions}; "
            f"episodes={report.completed_episodes}; "
            f"mean_return={report.mean_episode_return:.3f}; "
            f"policy_loss={report.policy_loss:.3f}; "
            f"value_loss={report.value_loss:.3f}; echo_loss={report.echo_loss:.3f}; "
            f"imitation_loss={report.imitation_loss:.3f}; "
            f"teacher_agreement={report.teacher_agreement:.3f}; "
            f"short_cycles={report.short_cycles}"
        )
        return
    if arguments.command == "watch-best":
        manifest = load_champion_manifest(arguments.champion)
        episode = select_champion_episode(manifest, arguments.case)
        watch_replay(
            episode.trajectory,
            frame_delay=Seconds(arguments.frame_delay_seconds),
            view_radius=ViewRadius(arguments.view_radius),
            frame_limit=FrameLimit(arguments.frame_limit)
            if arguments.frame_limit is not None
            else None,
            animate=not arguments.no_animate,
            identity=replay_identity(manifest, episode),
        )
        return
    if arguments.command == "watch-grid":
        watch_grid(
            arguments.champion,
            frame_delay=Seconds(arguments.frame_delay_seconds),
            view_radius=ViewRadius(arguments.view_radius),
            columns=GridColumnCount(arguments.columns),
            frame_limit=FrameLimit(arguments.frame_limit)
            if arguments.frame_limit is not None
            else None,
            animate=not arguments.no_animate,
        )
        return
    if arguments.command == "activate-champion-track":
        suite = load_suite(arguments.suite)
        activation = activate_champion_track(
            arguments.candidate,
            arguments.champion,
            expected_suite_id=suite.suite_id,
            archive_manifest=arguments.archive,
        )
        print(
            f"activated {activation.suite_id} champion {activation.policy_id!r} at "
            f"{activation.canonical_manifest}; archived={activation.archived_manifest}"
        )
        return
    _unreachable(arguments.command)


def _evaluate_scripted(
    *,
    binary: Path,
    suite_path: Path,
    output: Path | None,
    champion_path: Path | None,
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
    champion_path: Path | None,
    workers: WorkerCount,
    threshold_path: Path | None,
) -> None:
    suite = load_suite(suite_path)
    if output is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output = Path("artifacts/evaluations") / f"{policy.policy_id}-{timestamp}"
    summary = evaluate_policy(
        binary,
        policy,
        suite,
        output,
        workers=workers,
        progress=_print_evaluation_progress,
    )
    if threshold_path is not None:
        assert_meets_regression_threshold(
            summary, load_regression_threshold(threshold_path)
        )
    print(f"evaluation: {output / 'summary.json'}")
    if champion_path is not None:
        promoted = promote_champion(summary, champion_path)
        print(f"champion: {champion_path} ({'promoted' if promoted else 'retained'})")
    print(f"rank: {summary.rank}")
    print(
        f"throughput: {summary.decision_rate:.2f} decisions/s "
        f"over {summary.wall_seconds:.2f}s"
    )


def _menu_keycode(value: str) -> Keycode:
    if len(value) != 1:
        raise argparse.ArgumentTypeError("menu key must be exactly one character")
    return Keycode(ord(value))


def _unreachable(value: object) -> NoReturn:
    raise AssertionError(f"unhandled command: {value!r}")


def _print_evaluation_progress(progress: EvaluationProgress) -> None:
    print(
        f"case {progress.completed_cases}/{progress.total_cases} "
        f"{progress.case_id}: {progress.outcome}; steps={progress.policy_steps}; "
        f"D:{progress.max_depth}; XL:{progress.max_xl}; "
        f"elapsed={progress.elapsed_seconds:.1f}s; "
        f"completed-rate={progress.decision_rate:.2f}/s",
        flush=True,
    )


if __name__ == "__main__":
    main()
