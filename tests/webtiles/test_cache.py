import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

import pytest

from dcss_rl.webtiles.cache import (
    StaticCachePreparationStage,
    StaticDataCache,
    static_data_identity,
)
from dcss_rl.webtiles.process import ManagedGame


def cache_source(tmp_path: Path) -> tuple[Path, Path]:
    binary = tmp_path / "crawl"
    binary.write_bytes(b"test executable")
    data = tmp_path / "dat"
    data.mkdir()
    (data / "maps.des").write_text("static upstream data")
    saves = tmp_path / "closed" / "saves"
    for name, contents in (
        ("db/descriptions.db", b"database"),
        ("des/test.idx", b"index"),
        ("des/test.dsc", b"map"),
        ("des/test.lux", b"prelude"),
        ("des/test.lk", b"lock"),
        ("player.cs", b"private player"),
        ("start-ns.prf", b"private preferences"),
    ):
        path = saves / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    return binary, saves


def test_static_cache_copies_only_static_files_with_private_inodes(
    tmp_path: Path,
) -> None:
    binary, saves = cache_source(tmp_path)
    cache = StaticDataCache.capture(
        tmp_path / "snapshot",
        binary=binary,
        closed_save_directory=saves,
        source_identity=static_data_identity(binary),
    )
    restored = StaticDataCache.load(cache.directory, binary=binary)
    first = tmp_path / "first"
    second = tmp_path / "second"
    restored.populate(first, binary=binary)
    restored.populate(second, binary=binary)
    assert sorted(
        str(path.relative_to(first)) for path in first.rglob("*") if path.is_file()
    ) == ["db/descriptions.db", "des/test.dsc", "des/test.idx", "des/test.lux"]
    for member in restored.members:
        paths = [
            saves / member.name,
            cache.directory / member.name,
            first / member.name,
            second / member.name,
        ]
        assert len({path.stat().st_ino for path in paths}) == 4
    (first / "des/test.dsc").write_bytes(b"private mutation")
    assert (second / "des/test.dsc").read_bytes() == b"map"
    assert (cache.directory / "des/test.dsc").read_bytes() == b"map"


@pytest.mark.parametrize("changed", ["binary", "data-content", "data-mtime", "cache"])
@pytest.mark.parametrize("collect_timing", [False, True])
def test_static_cache_rejects_stale_or_corrupt_snapshot(
    tmp_path: Path, changed: str, collect_timing: bool
) -> None:
    binary, saves = cache_source(tmp_path)
    cache = StaticDataCache.capture(
        tmp_path / "snapshot",
        binary=binary,
        closed_save_directory=saves,
        source_identity=static_data_identity(binary),
    )
    if changed == "binary":
        binary.write_bytes(b"new executable")
    elif changed == "data-content":
        (tmp_path / "dat/maps.des").write_text("changed data")
    elif changed == "data-mtime":
        data = tmp_path / "dat/maps.des"
        stamp = data.stat().st_mtime_ns
        os.utime(data, ns=(stamp, stamp + 1_000_000_000))
    else:
        (cache.directory / "des/test.idx").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match=r"does not match|corrupt"):
        cache.populate(
            tmp_path / "destination", binary=binary, collect_timing=collect_timing
        )
    assert not (tmp_path / "destination").exists()


@pytest.mark.parametrize(
    "unsafe_name", ["../player.cs", "db/../player.db", "/tmp/bad.db", "db/player.cs"]
)
def test_static_cache_manifest_rejects_nonstatic_paths(
    tmp_path: Path, unsafe_name: str
) -> None:
    binary, saves = cache_source(tmp_path)
    cache = StaticDataCache.capture(
        tmp_path / "snapshot",
        binary=binary,
        closed_save_directory=saves,
        source_identity=static_data_identity(binary),
    )
    manifest = cache.directory / "manifest.json"
    text = manifest.read_text().replace('"db/descriptions.db"', json.dumps(unsafe_name))
    manifest.write_text(text)
    with pytest.raises(ValueError, match="not a static"):
        StaticDataCache.load(cache.directory, binary=binary)


def test_static_cache_rejects_symlinked_members_and_existing_targets(
    tmp_path: Path,
) -> None:
    binary, saves = cache_source(tmp_path)
    cache = StaticDataCache.capture(
        tmp_path / "snapshot",
        binary=binary,
        closed_save_directory=saves,
        source_identity=static_data_identity(binary),
    )
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "db").mkdir()
    with pytest.raises(FileExistsError):
        cache.populate(destination, binary=binary)
    member = cache.directory / "des/test.dsc"
    member.unlink()
    member.symlink_to(saves / "des/test.dsc")
    with pytest.raises(ValueError, match="corrupt"):
        cache.populate(tmp_path / "other", binary=binary)


@pytest.mark.parametrize("collect_timing", [False, True])
def test_managed_game_static_cache_is_explicitly_opt_in(
    tmp_path: Path, collect_timing: bool
) -> None:
    binary, saves = cache_source(tmp_path)
    cache = StaticDataCache.capture(
        tmp_path / "snapshot",
        binary=binary,
        closed_save_directory=saves,
        source_identity=static_data_identity(binary),
    )
    default = ManagedGame(binary, run_root=tmp_path / "default")
    default._prepare()
    assert list(default.save_path.iterdir()) == []
    assert default.static_cache_preparation_timing is None
    warmed = ManagedGame(
        binary,
        run_root=tmp_path / "warm",
        static_cache=cache,
        collect_static_cache_timing=collect_timing,
    )
    warmed._prepare()
    assert (warmed.save_path / "des/test.dsc").read_bytes() == b"map"
    assert (warmed.static_cache_preparation_timing is not None) is collect_timing
    exported = warmed.export_static_cache(tmp_path / "exported")
    assert exported.identity == cache.identity


def test_managed_game_rejects_export_before_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    game = ManagedGame(tmp_path / "crawl", run_root=tmp_path / "run")
    monkeypatch.setattr(game, "process", object())
    with pytest.raises(RuntimeError, match="close the DCSS game"):
        game.export_static_cache(tmp_path / "snapshot")


def test_static_cache_export_requires_recorded_unchanged_source_identity(
    tmp_path: Path,
) -> None:
    binary, saves = cache_source(tmp_path)
    default = ManagedGame(binary, run_root=saves.parent)
    with pytest.raises(RuntimeError, match="enable capture_static_cache"):
        default.export_static_cache(tmp_path / "missing-provenance")
    capture = ManagedGame(binary, run_root=saves.parent, capture_static_cache=True)
    capture._prepare()
    (tmp_path / "dat/maps.des").write_text("source changed after initialization")
    with pytest.raises(ValueError, match="changed since the source game started"):
        capture.export_static_cache(tmp_path / "stale-provenance")


def test_static_cache_default_timing_reads_no_clocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary, saves = cache_source(tmp_path)
    cache = StaticDataCache.capture(
        tmp_path / "snapshot",
        binary=binary,
        closed_save_directory=saves,
        source_identity=static_data_identity(binary),
    )

    def unexpected_clock() -> float:
        raise AssertionError("default cache preparation must not read timing clocks")

    monkeypatch.setattr("dcss_rl.webtiles.cache.time.perf_counter", unexpected_clock)
    monkeypatch.setattr("dcss_rl.webtiles.cache.time.thread_time", unexpected_clock)
    cache.validate(binary=binary)
    assert cache.populate(tmp_path / "destination", binary=binary) is None


def test_static_cache_timing_preserves_copy_and_verification_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dcss_rl.webtiles import cache as cache_module

    binary, saves = cache_source(tmp_path)
    cache = StaticDataCache.capture(
        tmp_path / "snapshot",
        binary=binary,
        closed_save_directory=saves,
        source_identity=static_data_identity(binary),
    )
    operations: list[tuple[str, str]] = []
    original_digest = cache_module._digest
    original_copy = shutil.copy2

    def digest(path: Path) -> cache_module.CacheDigest:
        operations.append(("hash", path.name))
        return original_digest(path)

    def copy(source: Path, destination: Path) -> Path:
        operations.append(("copy", source.name))
        return Path(original_copy(source, destination))

    monkeypatch.setattr(cache_module, "_digest", digest)
    monkeypatch.setattr(cache_module.shutil, "copy2", copy)
    first, second = tmp_path / "untimed", tmp_path / "timed"
    assert cache.populate(first, binary=binary) is None
    untimed_operations = operations.copy()
    operations.clear()
    report = cache.populate(second, binary=binary, collect_timing=True)
    assert operations == untimed_operations
    assert report is not None
    assert report.destination == str(second)
    assert report.members == len(cache.members)
    assert [span.stage for span in report.stages] == list(StaticCachePreparationStage)
    assert (
        0
        < sum(s.elapsed.wall_seconds for s in report.stages)
        <= report.total.wall_seconds
    )
    assert (
        0
        <= sum(s.elapsed.thread_cpu_seconds for s in report.stages)
        <= report.total.thread_cpu_seconds
    )
    serialized = json.loads(json.dumps(asdict(report)))
    assert serialized["stages"][0]["stage"] == "identity_validation"
    for member in cache.members:
        left, right = first / member.name, second / member.name
        assert left.read_bytes() == right.read_bytes()
        assert left.stat().st_mtime_ns == right.stat().st_mtime_ns
        assert left.stat().st_ino != right.stat().st_ino


@pytest.mark.parametrize("collect_timing", [False, True])
def test_static_cache_timing_preserves_destination_corruption_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, collect_timing: bool
) -> None:
    binary, saves = cache_source(tmp_path)
    cache = StaticDataCache.capture(
        tmp_path / "snapshot",
        binary=binary,
        closed_save_directory=saves,
        source_identity=static_data_identity(binary),
    )
    original_copy = shutil.copy2

    def corrupt_copy(source: Path, destination: Path) -> Path:
        copied = Path(original_copy(source, destination))
        copied.write_bytes(b"corrupt destination")
        return copied

    monkeypatch.setattr("dcss_rl.webtiles.cache.shutil.copy2", corrupt_copy)
    destination = tmp_path / "destination"
    with pytest.raises(ValueError, match="static cache changed while copying"):
        cache.populate(destination, binary=binary, collect_timing=collect_timing)
    assert list(destination.iterdir()) == []
