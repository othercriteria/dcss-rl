import json
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pytest
import torch

from dcss_rl.benchmark_startup_cache import (
    BenchmarkConfig,
    CacheArm,
    benchmark_command,
    compare_checkpoint_tensors,
    parse_ppo_metrics,
    read_reset_cache_timings,
    run_benchmark,
    validate_frozen_checkpoint,
)
from dcss_rl.env import ACTION_COUNT
from dcss_rl.features import FEATURE_SPEC_VERSION
from dcss_rl.learned import ModelConfig, SemanticActorCritic
from dcss_rl.units import ActionHistoryLength, FeatureSpecVersion, Seconds
from dcss_rl.webtiles.cache import (
    CacheDigest,
    CacheMemberCount,
    StaticCachePreparationStage,
    StaticCachePreparationTiming,
    StaticCacheStageTiming,
    StaticCacheTiming,
    StaticDataCache,
    StaticDataIdentity,
)


def configuration(tmp_path: Path) -> BenchmarkConfig:
    return BenchmarkConfig(
        checkpoint=tmp_path / "initial.pt",
        suite=tmp_path / "suite.json",
        static_data_cache=tmp_path / "cache",
        output=tmp_path / "report",
        run_root=tmp_path / "games",
        anchor_roots=(tmp_path / "anchors",),
    )


def test_benchmark_commands_keep_all_training_knobs_matched(tmp_path: Path) -> None:
    config = configuration(tmp_path)
    off = benchmark_command(config, CacheArm.OFF)
    on = benchmark_command(config, CacheArm.ON)
    assert off[:2] == on[:2] == ("poe", "train-ppo")
    off_options = dict(zip(off[2::2], off[3::2], strict=True))
    on_options = dict(zip(on[2::2], on[3::2], strict=True))
    assert off_options["--workers"] == "48"
    assert off_options["--rollout-length"] == "128"
    assert off_options["--updates"] == "1"
    assert "--static-data-cache" not in off_options
    assert on_options.pop("--static-data-cache") == str(config.static_data_cache)
    for key in ("--checkpoint", "--policy-id", "--run-root"):
        assert off_options.pop(key) != on_options.pop(key)
    assert off_options == on_options


def test_timing_benchmark_records_both_arms_only_when_requested(tmp_path: Path) -> None:
    config = configuration(tmp_path)
    for arm in CacheArm:
        assert "--collect-static-cache-timing" not in benchmark_command(config, arm)
        enabled = benchmark_command(
            replace(config, collect_static_cache_timing=True), arm
        )
        assert enabled[-2:] == (
            "--collect-static-cache-timing",
            "--record-rollout-trajectories",
        )


def test_frozen_benchmark_commands_remove_losses_anchors_and_decay_ownership(
    tmp_path: Path,
) -> None:
    config = replace(configuration(tmp_path), anchor_roots=(), frozen_policy=True)
    commands = [benchmark_command(config, arm) for arm in CacheArm]
    flags = {"--record-rollout-trajectories", "--no-aggregate-imitation-replay"}
    options = []
    for command in commands:
        assert flags <= set(command)
        assert not {
            "--warmup-action-kind",
            "--new-action-warmup-menu-key",
            "--warmup-train-value",
            "--imitation-trajectory-root",
        } & set(command)
        tokens = [token for token in command[2:] if token not in flags]
        parsed = dict(zip(tokens[::2], tokens[1::2], strict=True))
        for loss in ("policy", "value", "entropy", "imitation", "echo"):
            assert parsed[f"--{loss}-weight"] == "0"
        assert parsed["--short-cycle-cost"] == "0"
        assert parsed["--updates"] == parsed["--new-action-warmup-updates"] == "1"
        assert parsed["--workers"] == "48" and parsed["--rollout-length"] == "128"
        options.append(parsed)
    options[1].pop("--static-data-cache")
    for parsed in options:
        for key in ("--checkpoint", "--run-root", "--policy-id"):
            parsed.pop(key)
    assert options[0] == options[1]
    with pytest.raises(ValueError, match="must not supply anchor"):
        replace(configuration(tmp_path), frozen_policy=True)


@pytest.mark.parametrize("change", ["none", "actions", "features", "history"])
def test_frozen_preflight_rejects_implicit_expansion_or_unsupported_history(
    tmp_path: Path, change: str
) -> None:
    config = ModelConfig(
        action_count=int(ACTION_COUNT) - (change == "actions"),
        hidden_size=4,
        feature_spec_version=FeatureSpecVersion(5)
        if change == "features"
        else FEATURE_SPEC_VERSION,
        action_history_length=ActionHistoryLength(int(change == "history")),
    )
    model = SemanticActorCritic(config)
    path = tmp_path / "model.pt"
    torch.save(
        {
            "schema_version": 1,
            "feature_spec_version": config.feature_spec_version,
            "model_config": asdict(config),
            "model_state": model.state_dict(),
        },
        path,
    )
    if change == "none":
        validate_frozen_checkpoint(path)
    else:
        with pytest.raises(ValueError, match="current action/features and no history"):
            validate_frozen_checkpoint(path)


def test_empty_warmup_ownership_restores_actual_adamw_decay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dcss_rl.ppo import (
        _restore_warmup_parameters,
        _restrict_warmup_gradients,
        _warmup_action_indices,
    )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = SemanticActorCritic(ModelConfig(int(ACTION_COUNT), hidden_size=4))
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.1)
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    owned = _warmup_action_indices(int(ACTION_COUNT), int(ACTION_COUNT), ())
    assert owned == ()
    rows, columns = _restrict_warmup_gradients(
        model, owned, trainable_feature_indices=()
    )
    optimizer.step()
    assert not torch.equal(model.policy_head.weight, before["policy_head.weight"])
    _restore_warmup_parameters(
        model,
        frozen_rows=rows,
        frozen_policy_weight=before["policy_head.weight"],
        frozen_policy_bias=before["policy_head.bias"],
        frozen_feature_columns=columns,
        frozen_encoder_weight=before["encoder.0.weight"],
    )
    assert all(
        torch.equal(value, before[name]) for name, value in model.state_dict().items()
    )


@dataclass(frozen=True)
class _ParityProbe:
    policy_rollout_matched: bool


@pytest.mark.parametrize("failure", ["none", "weights", "parity"])
def test_frozen_runner_skips_anchors_and_enforces_final_audits_without_games(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    from dcss_rl import benchmark_startup_cache as benchmark

    config = replace(
        configuration(tmp_path),
        anchor_roots=(),
        frozen_policy=True,
        binary=tmp_path / "crawl",
    )
    model = SemanticActorCritic(ModelConfig(int(ACTION_COUNT), hidden_size=4))
    payload = {
        "schema_version": 1,
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "model_config": asdict(model.config),
        "model_state": model.state_dict(),
    }
    torch.save(payload, config.checkpoint)
    identity = StaticDataIdentity(CacheDigest("a" * 64), CacheDigest("b" * 64))
    cache = StaticDataCache(config.static_data_cache, identity, ())
    monkeypatch.setattr(StaticDataCache, "load", lambda *args, **kwargs: cache)
    monkeypatch.setattr(benchmark, "_source_digests", lambda: {})
    monkeypatch.setattr(
        benchmark, "_digest", lambda path: benchmark.FileDigest("fixed")
    )
    monkeypatch.setattr(
        benchmark.subprocess, "check_output", lambda *args, **kwargs: "revision"
    )

    def unexpected_prime(*args: object, **kwargs: object) -> None:
        raise AssertionError("frozen control must not prepare anchors")

    monkeypatch.setattr(
        "dcss_rl.replay_cache.prepare_imitation_replay", unexpected_prime
    )
    launches: list[tuple[str, ...]] = []

    def fake_run(
        command: tuple[str, ...], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        launches.append(command)
        destination = Path(command[command.index("--checkpoint") + 1])
        if failure == "weights":
            with torch.no_grad():
                model.policy_head.bias[0] += 1
        torch.save(payload, destination)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "prepared 0 anchor samples in 0.000s (cache_hit=False)\n"
                "update 1: decisions=6144; rate=100/s; seconds=1.00/0.01/0.01; frozen\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    monkeypatch.setattr(
        "dcss_rl.training_rollout_compare.compare_training_rollouts",
        lambda *args, **kwargs: _ParityProbe(failure != "parity"),
    )
    if failure != "none":
        with pytest.raises(RuntimeError, match=r"changed model|parity failed"):
            run_benchmark(config)
        assert not json.loads((config.output / "report.json").read_text())["completed"]
        assert len(launches) == (1 if failure == "weights" else 2)
    else:
        report = run_benchmark(config)
        assert report.completed and report.policy_rollout_matched
        assert report.prime_seconds is None
        assert set(report.frozen_tensor_comparisons) == set(CacheArm)
        assert all(
            not result.differing for result in report.frozen_tensor_comparisons.values()
        )
        assert len(launches) == 2


@pytest.mark.parametrize("cached", [False, True])
def test_benchmark_reads_typed_per_reset_timings_without_transition_scan(
    tmp_path: Path, cached: bool
) -> None:
    root = tmp_path / "worker-0" / "episode-0-attempt-0"
    root.mkdir(parents=True)
    timing = (
        StaticCachePreparationTiming(
            str(root / "saves"),
            CacheMemberCount(4),
            StaticCacheTiming(Seconds(5), Seconds(2)),
            tuple(
                StaticCacheStageTiming(
                    stage, StaticCacheTiming(Seconds(1), Seconds(0.4))
                )
                for stage in StaticCachePreparationStage
            ),
        )
        if cached
        else None
    )
    path = root / "trajectory.jsonl"
    path.write_text(
        json.dumps(
            {
                "type": "episode",
                "schema_version": 2,
                "metadata": {
                    "static_cache_preparation_timing": asdict(timing)
                    if timing
                    else None
                },
            }
        )
        + "\nthis transition is deliberately not decoded\n"
    )
    records = read_reset_cache_timings(tmp_path, expect_cache=cached)
    assert len(records) == 1
    assert records[0].trajectory == str(path)
    assert records[0].preparation == timing
    with pytest.raises(ValueError, match="unexpected or missing"):
        read_reset_cache_timings(tmp_path, expect_cache=not cached)


def test_benchmark_missing_recordings_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no recorded episode"):
        read_reset_cache_timings(tmp_path, expect_cache=True)


def test_benchmark_extracts_telemetry_and_keeps_full_context() -> None:
    output = (
        "prepared 27333 anchor samples in 0.302s (cache_hit=True)\n"
        "update 1: decisions=6144; rate=157.86/s; seconds=37.99/0.92/0.01; "
        "inference_batches=612; mean_batch=10.15; episodes=21; short_cycles=1007\n"
    )
    metrics = parse_ppo_metrics(output)
    assert metrics.anchor_cache_hit
    assert metrics.anchor_samples == 27333
    assert metrics.decisions == 6144
    assert metrics.collection_seconds == 37.99
    assert metrics.optimization_seconds == 0.92
    assert "episodes=21" in metrics.update_line
    with pytest.raises(ValueError, match="completed update-1"):
        parse_ppo_metrics("prepared 10 anchor samples in 0.1s (cache_hit=False)")


def test_checkpoint_comparison_ignores_metadata_but_detects_weights_and_dtype(
    tmp_path: Path,
) -> None:
    first, second = tmp_path / "first.pt", tmp_path / "second.pt"
    torch.save({"model_state": {"weight": torch.ones(2)}, "policy_id": "off"}, first)
    torch.save({"model_state": {"weight": torch.ones(2)}, "policy_id": "on"}, second)
    comparison = compare_checkpoint_tensors(first, second)
    assert comparison.compared == 1 and not comparison.differing
    torch.save({"model_state": {"weight": torch.ones(2, dtype=torch.float64)}}, second)
    assert compare_checkpoint_tensors(first, second).differing == ("weight",)
    torch.save({"model_state": {"weight": torch.zeros(2)}}, second)
    assert compare_checkpoint_tensors(first, second).differing == ("weight",)


def test_benchmark_refuses_existing_output_before_preparation(tmp_path: Path) -> None:
    config = configuration(tmp_path)
    config.output.mkdir()
    sentinel = config.output / "existing-report.json"
    sentinel.write_text("preserve me")
    with pytest.raises(FileExistsError):
        run_benchmark(config)
    assert sentinel.read_text() == "preserve me"
    assert not config.run_root.exists()
