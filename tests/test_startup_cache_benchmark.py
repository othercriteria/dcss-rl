from pathlib import Path

import pytest
import torch

from dcss_rl.benchmark_startup_cache import (
    BenchmarkConfig,
    CacheArm,
    benchmark_command,
    compare_checkpoint_tensors,
    parse_ppo_metrics,
    run_benchmark,
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
