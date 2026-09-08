import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dcss_rl.units import Seconds
from dcss_rl.webtiles.cache import StaticDataCache, static_data_identity
from dcss_rl.webtiles.identity_validation import (
    ExternalIdentityValidator,
    ValidationConcurrency,
)


def source_cache(tmp_path: Path) -> tuple[Path, StaticDataCache]:
    binary = tmp_path / "crawl"
    binary.write_bytes(b"unmodified fixture binary")
    data = tmp_path / "dat"
    data.mkdir()
    (data / "one.des").write_bytes(b"source data")
    (data / "two.des").write_bytes(b"more source data")
    saves = tmp_path / "closed"
    for name in ("db/text.db", "des/map.idx", "des/map.dsc"):
        path = saves / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    cache = StaticDataCache.capture(
        tmp_path / "cache",
        binary=binary,
        closed_save_directory=saves,
        source_identity=static_data_identity(binary),
    )
    return binary, cache


def response(binary: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        (),
        0,
        json.dumps(
            {
                "schema": 1,
                "binary": str(binary.resolve()),
                "identity": asdict(static_data_identity(binary)),
                "validation_wall_seconds": 0.001,
                "validation_cpu_seconds": 0.001,
                "helper_process_cpu_seconds": 0.01,
            }
        ),
        "",
    )


def test_real_helper_matches_identity_and_preserves_private_copy(
    tmp_path: Path,
) -> None:
    binary, cache = source_cache(tmp_path)
    validator = ExternalIdentityValidator()
    result = validator.validate(binary)
    assert result.identity == static_data_identity(binary)
    assert result.external_timing.request_wall_seconds > 0
    assert (
        result.external_timing.helper_process_cpu_seconds
        >= result.external_timing.validation_cpu_seconds
    )
    destination = tmp_path / "destination"
    timing = cache.populate(
        destination, binary=binary, collect_timing=True, identity_validator=validator
    )
    assert timing is not None and timing.external_identity_validation is not None
    for member in cache.members:
        source, copied = cache.directory / member.name, destination / member.name
        assert source.read_bytes() == copied.read_bytes()
        assert source.stat().st_ino != copied.stat().st_ino


@pytest.mark.parametrize(
    "mutation", ["binary", "same-mtime-data", "added", "removed", "mtime"]
)
def test_real_helper_rechecks_contents_and_freshness_every_request(
    tmp_path: Path, mutation: str
) -> None:
    binary, cache = source_cache(tmp_path)
    validator = ExternalIdentityValidator()
    original = validator.validate(binary).identity
    data = tmp_path / "dat/one.des"
    stamp = data.stat()
    if mutation == "binary":
        binary.write_bytes(b"changed binary")
    elif mutation == "same-mtime-data":
        data.write_bytes(b"changed data")
        os.utime(data, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    elif mutation == "added":
        (tmp_path / "dat/three.des").write_bytes(b"new")
    elif mutation == "removed":
        (tmp_path / "dat/two.des").unlink()
    else:
        os.utime(data, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1_000_000_000))
    with pytest.raises(ValueError, match="does not match"):
        cache.populate(
            tmp_path / "destination", binary=binary, identity_validator=validator
        )
    assert original == cache.identity
    assert not (tmp_path / "destination").exists()


@pytest.mark.parametrize(
    "failure", ["nonzero", "malformed", "wrong-path", "digest", "negative", "timeout"]
)
def test_helper_failures_fail_closed_and_release_permit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    binary, cache = source_cache(tmp_path)
    good = response(binary)
    bad = response(binary)
    if failure == "nonzero":
        bad.returncode = 1
    elif failure == "malformed":
        bad.stdout = "not json"
    elif failure != "timeout":
        value = json.loads(bad.stdout)
        if failure == "wrong-path":
            value["binary"] = "other"
        elif failure == "digest":
            value["identity"]["binary"] = "invalid"
        else:
            value["validation_cpu_seconds"] = -1
        bad.stdout = json.dumps(value)
    first = subprocess.TimeoutExpired("helper", 0.01) if failure == "timeout" else bad
    run = MagicMock(side_effect=[first, good])
    monkeypatch.setattr("dcss_rl.webtiles.identity_validation.subprocess.run", run)
    validator = ExternalIdentityValidator(max_concurrency=ValidationConcurrency(1))
    with pytest.raises(TimeoutError if failure == "timeout" else ValueError):
        cache.populate(
            tmp_path / "destination", binary=binary, identity_validator=validator
        )
    assert not (tmp_path / "destination").exists()
    assert validator.validate(binary).identity == cache.identity
    assert run.call_args.kwargs["shell"] is False
    assert 0 < run.call_args.kwargs["timeout"] <= 30


def test_validation_concurrency_is_bounded_and_requests_are_not_shared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary, cache = source_cache(tmp_path)
    good = response(binary)
    entered = threading.Barrier(3)
    release = threading.Event()
    lock = threading.Lock()
    active = peak = calls = 0

    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal active, peak, calls
        with lock:
            active += 1
            calls += 1
            index = calls
            peak = max(peak, active)
        if index <= 2:
            entered.wait(timeout=2)
        assert release.wait(timeout=2)
        with lock:
            active -= 1
        return good

    monkeypatch.setattr("dcss_rl.webtiles.identity_validation.subprocess.run", run)
    validator = ExternalIdentityValidator(max_concurrency=ValidationConcurrency(2))
    with ThreadPoolExecutor(max_workers=4) as executor:
        pending = [executor.submit(validator.validate, binary) for _ in range(4)]
        try:
            entered.wait(timeout=2)
            assert peak == 2
        finally:
            release.set()
        assert all(future.result().identity == cache.identity for future in pending)
    assert calls == 4 and peak == 2


def test_queue_deadline_prevents_launch_and_does_not_leak_permit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary, cache = source_cache(tmp_path)
    run = MagicMock(return_value=response(binary))
    monkeypatch.setattr("dcss_rl.webtiles.identity_validation.subprocess.run", run)
    validator = ExternalIdentityValidator(
        max_concurrency=ValidationConcurrency(1), timeout=Seconds(0.02)
    )
    assert validator._permits.acquire(blocking=False)
    try:
        with pytest.raises(TimeoutError, match="queue"):
            validator.validate(binary)
        run.assert_not_called()
    finally:
        validator._permits.release()
    assert validator.validate(binary).identity == cache.identity


def test_real_timeout_kills_and_reaps_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary, _ = source_cache(tmp_path)
    original_popen = subprocess.Popen
    children: list[subprocess.Popen[str]] = []

    def sleeping_child(*args: object, **kwargs: object) -> subprocess.Popen[str]:
        child = original_popen(
            (sys.executable, "-c", "import time; time.sleep(10)"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        children.append(child)
        return child

    monkeypatch.setattr(
        "dcss_rl.webtiles.identity_validation.subprocess.Popen", sleeping_child
    )
    validator = ExternalIdentityValidator(timeout=Seconds(0.03))
    with pytest.raises(TimeoutError, match="helper timed out"):
        validator.validate(binary)
    assert len(children) == 1
    assert children[0].poll() is not None
    assert validator._permits.acquire(blocking=False)
    validator._permits.release()


def test_external_identity_does_not_bypass_member_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary, cache = source_cache(tmp_path)
    monkeypatch.setattr(
        "dcss_rl.webtiles.identity_validation.subprocess.run",
        MagicMock(return_value=response(binary)),
    )
    (cache.directory / "des/map.idx").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt static cache member"):
        cache.populate(
            tmp_path / "destination",
            binary=binary,
            identity_validator=ExternalIdentityValidator(),
        )
    assert not (tmp_path / "destination").exists()


@pytest.mark.parametrize(
    "concurrency,timeout", [(0, 1), (1, 0), (1, float("inf")), (1, float("nan"))]
)
def test_invalid_validator_limits_rejected(concurrency: int, timeout: float) -> None:
    with pytest.raises(ValueError):
        ExternalIdentityValidator(
            max_concurrency=ValidationConcurrency(concurrency), timeout=Seconds(timeout)
        )
