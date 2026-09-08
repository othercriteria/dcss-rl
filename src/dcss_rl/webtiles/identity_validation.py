"""Opt-in bounded, fresh-process validation of static DCSS executable/data identity."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NewType

import msgspec

from dcss_rl.units import Seconds
from dcss_rl.webtiles.cache import (
    ExternalIdentityTiming,
    StaticDataIdentity,
    StaticIdentityValidation,
    static_data_identity,
)

ValidationConcurrency = NewType("ValidationConcurrency", int)
_DEFAULT_CONCURRENCY = ValidationConcurrency(2)
_DEFAULT_TIMEOUT = Seconds(30.0)
_SCHEMA = 1


@dataclass(frozen=True, slots=True)
class _Response:
    schema: int
    binary: str
    identity: StaticDataIdentity
    validation_wall_seconds: Seconds
    validation_cpu_seconds: Seconds
    helper_process_cpu_seconds: Seconds


class ExternalIdentityValidator:
    """Run-owned concurrency limiter; no persistent children or cached identities.

    Each call starts a new interpreter, hashes the actual requested tree, and waits
    for/reaps that child. A timeout or malformed response never falls back to a prior
    result. This preserves the existing validation-to-launch race, not atomicity.
    """

    def __init__(
        self,
        *,
        max_concurrency: ValidationConcurrency = _DEFAULT_CONCURRENCY,
        timeout: Seconds = _DEFAULT_TIMEOUT,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("validation concurrency must be positive")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("validation timeout must be finite and positive")
        self.max_concurrency = max_concurrency
        self.timeout = timeout
        self._permits = threading.BoundedSemaphore(max_concurrency)

    def validate(self, binary: Path) -> StaticIdentityValidation:
        started = time.perf_counter()
        deadline = started + self.timeout
        binary = Path(binary).resolve()
        remaining = deadline - time.perf_counter()
        if remaining <= 0 or not self._permits.acquire(timeout=remaining):
            raise TimeoutError("external identity validation queue timed out")
        try:
            acquired = time.perf_counter()
            remaining = deadline - acquired
            if remaining <= 0:
                raise TimeoutError("external identity validation deadline expired")
            try:
                completed = subprocess.run(
                    (
                        sys.executable,
                        "-m",
                        "dcss_rl.webtiles.identity_validation",
                        "--binary",
                        str(binary),
                    ),
                    capture_output=True,
                    text=True,
                    check=False,
                    shell=False,
                    timeout=remaining,
                )
            except subprocess.TimeoutExpired as error:
                # subprocess.run kills and waits for its direct child on timeout.
                raise TimeoutError(
                    "external identity validation helper timed out"
                ) from error
            if completed.returncode:
                raise ValueError("external identity validation helper failed")
            try:
                response = msgspec.json.decode(completed.stdout, type=_Response)
            except (msgspec.DecodeError, TypeError) as error:
                raise ValueError(
                    "malformed external identity validation response"
                ) from error
            durations = (
                response.validation_wall_seconds,
                response.validation_cpu_seconds,
                response.helper_process_cpu_seconds,
            )
            if (
                response.schema != _SCHEMA
                or response.binary != str(binary)
                or any(
                    re.fullmatch(r"[0-9a-f]{64}", digest) is None
                    for digest in (response.identity.binary, response.identity.data)
                )
                or any(not math.isfinite(value) or value < 0 for value in durations)
            ):
                raise ValueError("invalid external identity validation response")
            finished = time.perf_counter()
            if finished > deadline:
                raise TimeoutError("external identity validation deadline expired")
            return StaticIdentityValidation(
                response.identity,
                ExternalIdentityTiming(
                    Seconds(acquired - started),
                    Seconds(finished - started),
                    response.validation_wall_seconds,
                    response.validation_cpu_seconds,
                    response.helper_process_cpu_seconds,
                ),
            )
        finally:
            self._permits.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    arguments = parser.parse_args()
    binary = arguments.binary.resolve()
    wall, cpu = time.perf_counter(), time.process_time()
    identity = static_data_identity(binary)
    response = _Response(
        _SCHEMA,
        str(binary),
        identity,
        Seconds(time.perf_counter() - wall),
        Seconds(time.process_time() - cpu),
        Seconds(time.process_time()),
    )
    print(json.dumps(asdict(response), separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
