import json
import os
from pathlib import Path

import pytest

from dcss_rl.webtiles.cache import StaticDataCache, static_data_identity
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
def test_static_cache_rejects_stale_or_corrupt_snapshot(
    tmp_path: Path, changed: str
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
        cache.populate(tmp_path / "destination", binary=binary)
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


def test_managed_game_static_cache_is_explicitly_opt_in(tmp_path: Path) -> None:
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
    warmed = ManagedGame(binary, run_root=tmp_path / "warm", static_cache=cache)
    warmed._prepare()
    assert (warmed.save_path / "des/test.dsc").read_bytes() == b"map"
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
