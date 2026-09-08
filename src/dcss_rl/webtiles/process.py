"""Lifecycle management for isolated local DCSS WebTiles processes."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import BinaryIO

from dcss_rl.units import GameSeed, Keycode, Seconds, UnixSocketPathBytes
from dcss_rl.webtiles.cache import (
    StaticCachePreparationTiming,
    StaticDataCache,
    StaticDataIdentity,
    static_data_identity,
)
from dcss_rl.webtiles.transport import (
    FlushBoundary,
    ObservationBatch,
    WebtilesTransport,
)

_DEFAULT_GAME_TIMEOUT = Seconds(15.0)
_AUTOMATIC_COMMAND_QUIET_PERIOD = Seconds(0.5)
_PROCESS_SHUTDOWN_TIMEOUT = Seconds(3.0)
_MAX_UNIX_SOCKET_PATH_BYTES = UnixSocketPathBytes(107)


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
        static_cache: StaticDataCache | None = None,
        capture_static_cache: bool = False,
        collect_static_cache_timing: bool = False,
    ) -> None:
        self.binary = Path(binary).resolve()
        self.config = config or GameConfig()
        self.timeout = timeout
        self.static_cache = static_cache
        self._capture_static_cache = capture_static_cache
        self.collect_static_cache_timing = collect_static_cache_timing
        self.static_cache_preparation_timing: StaticCachePreparationTiming | None = None
        self._static_cache_identity: StaticDataIdentity | None = None
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
        socket_path_bytes = len(bytes(self.socket_path))
        if socket_path_bytes > _MAX_UNIX_SOCKET_PATH_BYTES:
            raise ValueError(
                f"DCSS socket path is {socket_path_bytes} bytes; Linux permits at most "
                f"{_MAX_UNIX_SOCKET_PATH_BYTES}. Choose a shorter run/output path."
            )
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.morgue_path.mkdir(exist_ok=True)
        self.save_path.mkdir(exist_ok=True)
        if self._capture_static_cache:
            self._static_cache_identity = static_data_identity(self.binary)
        if self.static_cache is not None:
            self.static_cache_preparation_timing = self.static_cache.populate(
                self.save_path,
                binary=self.binary,
                collect_timing=self.collect_static_cache_timing,
            )
            self._static_cache_identity = self.static_cache.identity
        (self.run_root / "macros").mkdir(exist_ok=True)
        rc_path = self.run_root / "crawl.rc"
        rc_path.write_text(
            "\n".join(
                (
                    f"save_dir = {self.save_path}",
                    f"morgue_dir = {self.morgue_path}",
                    "restart_after_game = false",
                    "show_more = false",
                    "view_delay = 0",
                    "travel_delay = -1",
                    "rest_delay = -1",
                    "use_animations =",
                    "",
                )
            )
        )
        return rc_path

    def start(self, *, initial_keycode: Keycode | None = None) -> ObservationBatch:
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
            # Headless WebTiles still includes fd 0 in its blocking pselect. A
            # DEVNULL fd is permanently readable and makes an otherwise idle DCSS
            # process busy-spin; an unwritten pipe lets the socket wait block.
            stdin=subprocess.PIPE,
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
            if initial_keycode is not None:
                self.transport.send_key(initial_keycode)
            return self.transport.receive_until_flush(
                quiet_period=_AUTOMATIC_COMMAND_QUIET_PERIOD
                if initial_keycode is not None
                else None,
                boundary=FlushBoundary.QUIESCENCE,
            )
        except BaseException:
            self.close()
            raise

    def send_key(
        self,
        key: str | int,
        *,
        level_transition: bool = False,
        ui_continuation: bool = False,
    ) -> ObservationBatch:
        """Apply one primitive input and collect the resulting state delta."""
        if level_transition and ui_continuation:
            raise ValueError("input boundary requests must be mutually exclusive")
        if self.transport is None:
            raise RuntimeError("DCSS game is not started")
        self.transport.send_key(key)
        quiet_period = _AUTOMATIC_COMMAND_QUIET_PERIOD if key in {"o", "5"} else None
        if quiet_period is None and not self.transport.output_available():
            # Some supported releases emit no delta or flush for a command that has
            # no visible effect (notably WAIT in 0.33). This upstream message is
            # processed when DCSS next enters its input loop and guarantees a
            # full-state response without modifying the game.
            self.transport.request_full_state()
        return self.transport.receive_until_flush(
            quiet_period=quiet_period,
            boundary=FlushBoundary.LEVEL_TRANSITION
            if level_transition
            else FlushBoundary.UI_CONTINUATION
            if ui_continuation
            else FlushBoundary.INPUT_READY_OR_QUIESCENCE
            if quiet_period is not None
            else FlushBoundary.QUIESCENCE,
        )

    def close(self) -> None:
        """Close transport and stop DCSS, escalating only if it fails to exit."""
        if self.transport is not None:
            self.transport.close()
            self.transport = None
        if self.process is not None:
            if self.process.stdin is not None:
                self.process.stdin.close()
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

    def export_static_cache(self, destination: Path) -> StaticDataCache:
        """Snapshot upstream static data only after this game's process is closed."""
        if self.process is not None:
            raise RuntimeError("close the DCSS game before exporting static caches")
        if self._static_cache_identity is None:
            raise RuntimeError(
                "enable capture_static_cache before starting the source game"
            )
        return StaticDataCache.capture(
            destination,
            binary=self.binary,
            closed_save_directory=self.save_path,
            source_identity=self._static_cache_identity,
        )

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
