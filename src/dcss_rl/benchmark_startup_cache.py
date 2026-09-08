"""Matched one-update PPO benchmark of private static-cache prepopulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import resource
import subprocess
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import NewType, cast

from dcss_rl.units import ActionCount, RolloutLength, Seconds, WorkerCount
from dcss_rl.webtiles.cache import StaticDataCache, StaticDataIdentity

BenchmarkSeed = NewType("BenchmarkSeed", int)
ExitStatus = NewType("ExitStatus", int)
FileDigest = NewType("FileDigest", str)
TensorCount = NewType("TensorCount", int)
_DEFAULT_BINARY = Path("vendor/crawl/crawl-ref/source/crawl")
_DEFAULT_WORKERS = WorkerCount(48)
_DEFAULT_ROLLOUT = RolloutLength(128)


class CacheArm(StrEnum):
    OFF = "off"
    ON = "on"


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    checkpoint: Path
    suite: Path
    static_data_cache: Path
    output: Path
    run_root: Path
    anchor_roots: tuple[Path, ...]
    binary: Path = _DEFAULT_BINARY
    imitation_cache: Path = Path(".cache/imitation-replay")
    seed: BenchmarkSeed = BenchmarkSeed(1)
    workers: WorkerCount = _DEFAULT_WORKERS
    rollout_length: RolloutLength = _DEFAULT_ROLLOUT


@dataclass(frozen=True, slots=True)
class PpoMetrics:
    anchor_samples: ActionCount
    anchor_seconds: Seconds
    anchor_cache_hit: bool
    decisions: ActionCount
    collection_seconds: Seconds
    optimization_seconds: Seconds
    checkpoint_seconds: Seconds
    update_line: str


@dataclass(frozen=True, slots=True)
class ArmReport:
    arm: CacheArm
    command: tuple[str, ...]
    returncode: ExitStatus
    wall_seconds: Seconds
    child_cpu_seconds: Seconds
    metrics: PpoMetrics | None
    source_unchanged: bool


@dataclass(frozen=True, slots=True)
class TensorComparison:
    compared: TensorCount
    differing: tuple[str, ...]


@dataclass(slots=True)
class BenchmarkReport:
    config: BenchmarkConfig
    revision: str
    source_digests: dict[str, FileDigest]
    input_digests: dict[str, FileDigest]
    static_identity: StaticDataIdentity
    prime_seconds: Seconds | None = None
    arms: list[ArmReport] = field(default_factory=list)
    tensor_comparison: TensorComparison | None = None
    completed: bool = False
    caveats: tuple[str, ...] = (
        "Single off/on pair; fixed ordering has no replicate error estimate.",
        "CLI preparation timing is rounded to .001s; update timings to .01s.",
        "Child CPU includes awaited trainer and DCSS descendants, not only startup.",
        "Anchor-cache priming is separate; source digests must remain unchanged.",
        "Tensor equality checks model state, not a claim of raw protocol equality.",
    )


def benchmark_command(config: BenchmarkConfig, arm: CacheArm) -> tuple[str, ...]:
    command = [
        "poe",
        "train-ppo",
        "--initial-checkpoint",
        str(config.checkpoint),
        "--checkpoint",
        str(config.output / f"{arm}.pt"),
        "--policy-id",
        f"startup-cache-benchmark-{arm}",
        "--binary",
        str(config.binary),
        "--suite",
        str(config.suite),
        "--run-root",
        str(config.run_root / arm),
        "--seed",
        str(config.seed),
        "--updates",
        "1",
        "--workers",
        str(config.workers),
        "--rollout-length",
        str(config.rollout_length),
        "--inference-batch-size",
        "64",
        "--inference-batch-wait-seconds",
        "0.001",
        "--epochs-per-update",
        "4",
        "--learning-rate",
        "0.0001",
        "--policy-weight",
        "1",
        "--value-weight",
        "0.5",
        "--entropy-weight",
        "0.01",
        "--imitation-weight",
        "0.1",
        "--teacher-balance-exponent",
        "0.5",
        "--echo-weight",
        "0",
        "--return-boundary",
        "continuing-reset",
        "--decision-cost",
        "0.01",
        "--device",
        "cuda",
        "--imitation-cache-directory",
        str(config.imitation_cache),
    ]
    for root in config.anchor_roots:
        command.extend(("--imitation-trajectory-root", str(root)))
    if arm is CacheArm.ON:
        command.extend(("--static-data-cache", str(config.static_data_cache)))
    return tuple(command)


def parse_ppo_metrics(output: str) -> PpoMetrics:
    prepared = re.search(
        r"prepared (\d+) anchor samples in ([\d.]+)s \(cache_hit=(True|False)\)", output
    )
    update = re.search(
        r"^update 1: decisions=(\d+); rate=[\d.]+/s; "
        r"seconds=([\d.]+)/([\d.]+)/([\d.]+);[^\n]*",
        output,
        re.MULTILINE,
    )
    if prepared is None or update is None:
        raise ValueError("benchmark needs preparation and completed update-1 telemetry")
    return PpoMetrics(
        ActionCount(int(prepared[1])),
        Seconds(float(prepared[2])),
        prepared[3] == "True",
        ActionCount(int(update[1])),
        Seconds(float(update[2])),
        Seconds(float(update[3])),
        Seconds(float(update[4])),
        update[0],
    )


def compare_checkpoint_tensors(first: Path, second: Path) -> TensorComparison:
    import torch

    def state(path: Path) -> dict[str, torch.Tensor]:
        loaded: object = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(loaded, dict):
            raise ValueError("checkpoint is not an object")
        values = loaded.get("model_state")
        if not isinstance(values, dict) or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in values.items()
        ):
            raise ValueError("checkpoint has no tensor model_state")
        return cast(dict[str, torch.Tensor], values)

    left, right = state(first), state(second)
    names = sorted(left.keys() | right.keys())
    differing = tuple(
        name
        for name in names
        if name not in left
        or name not in right
        or left[name].dtype != right[name].dtype
        or not torch.equal(left[name], right[name])
    )
    return TensorComparison(TensorCount(len(names)), differing)


def run_benchmark(config: BenchmarkConfig) -> BenchmarkReport:
    from dcss_rl.policy import ScriptedMibePolicy
    from dcss_rl.replay_cache import prepare_imitation_replay

    if config.workers < 1 or config.rollout_length < 1:
        raise ValueError("worker and rollout counts must be positive")
    if config.output.resolve() == config.run_root.resolve():
        raise ValueError("report output and game run root must be distinct")
    for path in (config.output, config.run_root):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    cache = StaticDataCache.load(config.static_data_cache, binary=config.binary)
    trajectories = tuple(
        path
        for root in config.anchor_roots
        for path in sorted(root.rglob("trajectory.jsonl"))
    )
    if not trajectories:
        raise ValueError("matched benchmark requires nonempty anchor replay")
    source_digests = _source_digests()
    report = BenchmarkReport(
        config=config,
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        source_digests=source_digests,
        input_digests={
            str(path): _digest(path)
            for path in (
                config.binary,
                Path("pyproject.toml"),
                config.checkpoint,
                config.suite,
                *trajectories,
                config.static_data_cache / "manifest.json",
            )
        },
        static_identity=cache.identity,
    )
    config.output.mkdir(parents=True)
    config.run_root.mkdir(parents=True)
    _record(report)
    print(f"Priming {len(trajectories)} anchor trajectories", flush=True)
    started = perf_counter()
    prepare_imitation_replay(
        trajectories,
        teacher=ScriptedMibePolicy(),
        cache_directory=config.imitation_cache,
    )
    report.prime_seconds = Seconds(perf_counter() - started)
    _record(report)
    for arm in CacheArm:
        if _source_digests() != source_digests:
            raise RuntimeError("source changed before benchmark arm")
        if any(
            _digest(Path(path)) != digest
            for path, digest in report.input_digests.items()
        ):
            raise RuntimeError("benchmark input changed before arm")
        command = benchmark_command(config, arm)
        print(f"Starting cache-{arm}: {' '.join(command)}", flush=True)
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        started = perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True)
        elapsed = Seconds(perf_counter() - started)
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        output = completed.stdout + completed.stderr
        (config.output / f"{arm}.log").write_text(output)
        metrics = parse_ppo_metrics(output) if completed.returncode == 0 else None
        result = ArmReport(
            arm,
            command,
            ExitStatus(completed.returncode),
            elapsed,
            Seconds(
                after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
            ),
            metrics,
            _source_digests() == source_digests,
        )
        report.arms.append(result)
        _record(report)
        print(output, flush=True)
        print(
            f"Finished cache-{arm}: wall={elapsed:.3f}s; "
            f"child_cpu={result.child_cpu_seconds:.3f}s",
            flush=True,
        )
        if completed.returncode or not result.source_unchanged:
            raise RuntimeError("benchmark arm failed or source changed; stopping pair")
        if (
            metrics is None
            or metrics.decisions != config.workers * config.rollout_length
        ):
            raise RuntimeError("benchmark transition budget mismatch")
        if not metrics.anchor_cache_hit:
            raise RuntimeError(
                "anchor replay was not warm after priming; pair confounded"
            )
    report.tensor_comparison = compare_checkpoint_tensors(
        config.output / "off.pt", config.output / "on.pt"
    )
    report.completed = True
    _record(report)
    print(f"Tensor comparison: {report.tensor_comparison}", flush=True)
    return report


def _source_digests() -> dict[str, FileDigest]:
    root = Path(__file__).parent
    return {str(path): _digest(path) for path in sorted(root.rglob("*.py"))}


def _digest(path: Path) -> FileDigest:
    with path.open("rb") as source:
        return FileDigest(hashlib.file_digest(source, "sha256").hexdigest())


def _json_default(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported report value: {type(value)}")


def _record(report: BenchmarkReport) -> None:
    (report.config.output / "report.json").write_text(
        json.dumps(asdict(report), indent=2, default=_json_default) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--static-data-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--imitation-trajectory-root", type=Path, action="append", required=True
    )
    parser.add_argument("--binary", type=Path, default=_DEFAULT_BINARY)
    arguments = parser.parse_args()
    run_benchmark(
        BenchmarkConfig(
            checkpoint=arguments.checkpoint,
            suite=arguments.suite,
            static_data_cache=arguments.static_data_cache,
            output=arguments.output,
            run_root=arguments.run_root,
            anchor_roots=tuple(arguments.imitation_trajectory_root),
            binary=arguments.binary,
        )
    )


if __name__ == "__main__":
    main()
