"""Lifecycle management for isolated local DCSS WebTiles processes."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import BinaryIO

from dcss_rl.units import GameSeed, Seconds
from dcss_rl.webtiles.transport import ObservationBatch, WebtilesTransport

_DEFAULT_GAME_TIMEOUT = Seconds(15.0)
_AUTOMATIC_COMMAND_QUIET_PERIOD = Seconds(0.5)
_PROCESS_SHUTDOWN_TIMEOUT = Seconds(3.0)


@dataclass(frozen=True, slots=True)
class GameConfig:
    """Player-visible configuration for one reproducible game process."""

    name: str = "dcss-rl"
    species: str = "Mi"
    background: str = "Be"
    seed: GameSeed | None = None


class ManagedGame:
    """An upstream DCSS process and its direct WebTiles connection.

    Every instance gets private mutable directories. The upstream source tree is used
    only for immutable game data and the compiled executable.
    """

    def __init__(
        self,
        binary: Path,
        *,
        config: GameConfig | None = None,
        timeout: Seconds = _DEFAULT_GAME_TIMEOUT,
        run_root: Path | None = None,
    ) -> None:
        self.binary = Path(binary).resolve()
        self.config = config or GameConfig()
        self.timeout = timeout
        self._owned_root: tempfile.TemporaryDirectory[str] | None = None
        if run_root is None:
            self._owned_root = tempfile.TemporaryDirectory(prefix="dcss-rl-game-")
            run_root = Path(self._owned_root.name)
        self.run_root = Path(run_root).resolve()
        self.socket_path = self.run_root / "game.sock"
        self.transport: WebtilesTransport | None = None
        self.process: subprocess.Popen[bytes] | None = None
        self._log_handle: BinaryIO | None = None

    @property
    def log_path(self) -> Path:
        return self.run_root / "crawl.log"

    @property
    def morgue_path(self) -> Path:
        return self.run_root / "morgue"

    @property
    def save_path(self) -> Path:
        return self.run_root / "saves"

    def _prepare(self) -> Path:
        if not self.binary.is_file():
            raise FileNotFoundError(f"DCSS binary not found: {self.binary}")
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.morgue_path.mkdir(exist_ok=True)
        self.save_path.mkdir(exist_ok=True)
        (self.run_root / "macros").mkdir(exist_ok=True)
        rc_path = self.run_root / "crawl.rc"
        rc_path.write_text(
            "\n".join(
                (
                    f"save_dir = {self.save_path}",
                    f"morgue_dir = {self.morgue_path}",
                    "restart_after_game = false",
                    "show_more = false",
                    "",
                )
            )
        )
        return rc_path

    def start(self) -> ObservationBatch:
        """Start DCSS, attach as its primary controller, and return initial output."""
        if self.process is not None:
            raise RuntimeError("DCSS game is already started")
        rc_path = self._prepare()
        arguments = [
            str(self.binary),
            "-name",
            self.config.name,
            "-species",
            self.config.species,
            "-background",
            self.config.background,
            "-rc",
            str(rc_path),
            "-macro",
            str(self.run_root / "macros"),
            "-morgue",
            str(self.morgue_path),
            "-webtiles-socket",
            str(self.socket_path),
            "-await-connection",
            "-no-player-bones",
        ]
        if self.config.seed is not None:
            arguments.extend(("-seed", str(self.config.seed)))

        self._log_handle = self.log_path.open("wb")
        self.process = subprocess.Popen(
            arguments,
            cwd=self.binary.parent,
            stdin=subprocess.DEVNULL,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.transport = WebtilesTransport(
            self.socket_path,
            timeout=self.timeout,
        )
        try:
            self.transport.connect()
            return self.transport.receive_until_flush()
        except BaseException:
            self.close()
            raise

    def send_key(self, key: str | int) -> ObservationBatch:
        """Apply one primitive input and collect the resulting state delta."""
        if self.transport is None:
            raise RuntimeError("DCSS game is not started")
        self.transport.send_key(key)
        quiet_period = _AUTOMATIC_COMMAND_QUIET_PERIOD if key in {"o", "5"} else None
        return self.transport.receive_until_flush(quiet_period=quiet_period)

    def close(self) -> None:
        """Close transport and stop DCSS, escalating only if it fails to exit."""
        if self.transport is not None:
            self.transport.close()
            self.transport = None
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=_PROCESS_SHUTDOWN_TIMEOUT)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=_PROCESS_SHUTDOWN_TIMEOUT)
            self.process = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        self.socket_path.unlink(missing_ok=True)
        if self._owned_root is not None:
            self._owned_root.cleanup()
            self._owned_root = None

    def preserve(self, destination: Path) -> None:
        """Copy this episode's runtime artifacts to a durable location."""
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copytree(self.run_root, destination)

    def __enter__(self) -> ManagedGame:
        self.start()
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()
