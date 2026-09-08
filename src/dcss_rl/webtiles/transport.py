"""Direct transport for the Unix-datagram protocol spoken by a DCSS process.

This intentionally bypasses the browser-facing Tornado server. DCSS remains an
unmodified upstream binary, while rollout workers avoid HTTP/WebSocket overhead and
account-management machinery that is irrelevant to local training.
"""

from __future__ import annotations

import json
import select
import socket
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum, StrEnum, auto
from pathlib import Path
from typing import cast

from dcss_rl.schema import JsonObject
from dcss_rl.units import Seconds

_DEFAULT_TRANSPORT_TIMEOUT = Seconds(10.0)
_DEFAULT_INPUT_QUIET_PERIOD = Seconds(0.01)
_SOCKET_APPEARANCE_POLL_INTERVAL = Seconds(0.01)
_BLOCKING_UI_QUIET_PERIOD = Seconds(0.5)
_LEVEL_INPUT_MODES = frozenset({1, 2, 3, 4, 5, 7, 8})
_BLOCKING_UI_TYPES = frozenset(
    {
        "describe-generic",
        "describe-feature-wide",
        "describe-item",
        "describe-spell",
        "describe-monster",
        "describe-god",
        "describe-cards",
        "msgwin-get-line",
        "version",
        "game-over",
        "formatted-scroller",
        "newgame-random-combo",
        "seed-selection",
        "newgame-choice",
    }
)


class FlushBoundary(StrEnum):
    """Evidence required to treat a protocol flush as a policy boundary."""

    QUIESCENCE = "quiescence"
    INPUT_READY_OR_QUIESCENCE = "input-ready-or-quiescence"
    LEVEL_TRANSITION = "level-transition"


class _InputReadiness(Enum):
    UNKNOWN = auto()
    BUSY = auto()
    READY = auto()


@dataclass
class _LevelBoundaryEvidence:
    """Current-exchange input evidence, not a remembered previous ready state."""

    input_mode: int | None = None
    ui_stack: list[bool] = field(default_factory=list)
    line_input: bool = False

    def apply(self, payload: JsonObject) -> None:
        kind = payload.get("msg")
        if kind == "input_mode":
            mode = payload.get("mode")
            self.input_mode = mode if isinstance(mode, int) else None
        elif kind == "menu":
            self.ui_stack.append(True)
        elif kind == "ui-push":
            ui_type = payload.get("type")
            self.ui_stack.append(
                isinstance(ui_type, str) and ui_type in _BLOCKING_UI_TYPES
            )
        elif kind in {"close_menu", "ui-pop"}:
            if self.ui_stack:
                self.ui_stack.pop()
        elif kind == "close_all_menus":
            self.ui_stack.clear()
        elif kind == "ui-stack":
            self.ui_stack.clear()
            items = payload.get("items")
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        self.apply(item)
        elif kind == "init_input":
            self.line_input = True
        elif kind == "close_input":
            self.line_input = False

    @property
    def explicit_input(self) -> bool:
        return self.input_mode in _LEVEL_INPUT_MODES

    @property
    def blocking_ui(self) -> bool:
        if self.ui_stack and not self.ui_stack[-1]:
            return False
        return self.line_input or bool(self.ui_stack and self.ui_stack[-1])


@dataclass(frozen=True, slots=True)
class Message:
    """One decoded message emitted by DCSS.

    Control messages have a leading ``*`` on the wire and concern the transport or
    server (for example, ``flush_messages``). Other messages are player-visible game
    observations and are suitable inputs to a policy.
    """

    payload: JsonObject
    control: bool = False

    @property
    def kind(self) -> str:
        value = self.payload.get("msg")
        return value if isinstance(value, str) else ""


@dataclass(frozen=True, slots=True)
class ObservationBatch:
    """Messages produced atomically before DCSS next waits for player input."""

    observations: tuple[Message, ...]
    controls: tuple[Message, ...]
    ordered: tuple[Message, ...] = ()

    @property
    def messages(self) -> tuple[Message, ...]:
        """Return messages in wire order, with a fallback for legacy fixtures."""
        return self.ordered or (*self.observations, *self.controls)


class WebtilesTransport:
    """Blocking, single-owner connection to one local DCSS process."""

    def __init__(
        self,
        game_socket: Path,
        *,
        timeout: Seconds = _DEFAULT_TRANSPORT_TIMEOUT,
        input_quiet_period: Seconds = _DEFAULT_INPUT_QUIET_PERIOD,
        client_directory: Path | None = None,
    ) -> None:
        self.game_socket = Path(game_socket)
        self.timeout = timeout
        self.input_quiet_period = input_quiet_period
        self._owned_directory: tempfile.TemporaryDirectory[str] | None = None
        if client_directory is None:
            self._owned_directory = tempfile.TemporaryDirectory(
                prefix="dcss-rl-webtiles-"
            )
            client_directory = Path(self._owned_directory.name)
        else:
            client_directory.mkdir(parents=True, exist_ok=True)

        self.client_socket = client_directory / "client.sock"
        self._socket: socket.socket | None = None
        self._fragment_buffer = bytearray()
        self._input_readiness = _InputReadiness.UNKNOWN

    def connect(self, *, primary: bool = True) -> None:
        """Wait for DCSS's socket, bind locally, and send the attach handshake."""
        if self._socket is not None:
            raise RuntimeError("WebTiles transport is already connected")

        deadline = time.monotonic() + self.timeout
        while not self.game_socket.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"DCSS socket did not appear: {self.game_socket}")
            time.sleep(_SOCKET_APPEARANCE_POLL_INTERVAL)

        transport = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        transport.settimeout(self.timeout)
        transport.bind(str(self.client_socket))
        self._socket = transport
        self.send({"msg": "attach", "primary": primary})

    def send(self, payload: Mapping[str, object]) -> None:
        """Send one JSON control message to DCSS."""
        if self._socket is None:
            raise RuntimeError("WebTiles transport is not connected")
        encoded = json.dumps(payload, separators=(",", ":")).encode()
        self._socket.sendto(encoded, str(self.game_socket))

    def send_key(self, key: str | int) -> None:
        """Send one Unicode character or explicit DCSS keycode."""
        if isinstance(key, str):
            if len(key) != 1:
                raise ValueError("send_key expects exactly one character")
            keycode = ord(key)
        else:
            keycode = key
        self.send({"msg": "key", "keycode": keycode})

    def output_available(self, *, within: Seconds | None = None) -> bool:
        """Report whether DCSS emitted output within a bounded interval."""
        if self._socket is None:
            raise RuntimeError("WebTiles transport is not connected")
        wait = self.input_quiet_period if within is None else within
        readable, _, _ = select.select([self._socket], [], [], wait)
        return bool(readable)

    def request_full_state(self) -> None:
        """Queue a full-state response as an input-loop synchronization probe."""
        self.send({"msg": "spectator_joined"})

    def receive(self) -> Message:
        """Receive and decode one newline-terminated, possibly fragmented message."""
        if self._socket is None:
            raise RuntimeError("WebTiles transport is not connected")

        while True:
            fragment = self._socket.recv(128 * 1024)
            self._fragment_buffer.extend(fragment)
            if not self._fragment_buffer.endswith(b"\n"):
                continue

            wire_message = bytes(self._fragment_buffer[:-1])
            self._fragment_buffer.clear()
            control = wire_message.startswith(b"*")
            if control:
                wire_message = wire_message[1:]
            decoded: object = json.loads(wire_message)
            if not isinstance(decoded, dict):
                raise ValueError("DCSS WebTiles message must be a JSON object")
            if not all(isinstance(key, str) for key in decoded):
                raise ValueError("DCSS WebTiles object keys must be strings")
            return Message(cast(JsonObject, decoded), control=control)

    def receive_until_flush(
        self,
        *,
        quiet_period: Seconds | None = None,
        boundary: FlushBoundary = FlushBoundary.QUIESCENCE,
    ) -> ObservationBatch:
        """Collect deltas until DCSS flushes and becomes quiescent for input.

        DCSS also flushes after each automatic travel/rest turn. Those are rendering
        boundaries, not policy-action boundaries, so consecutive ready batches are
        coalesced until the process stops emitting output.
        """
        if self._socket is None:
            raise RuntimeError("WebTiles transport is not connected")
        settle = self.input_quiet_period if quiet_period is None else quiet_period
        observations: list[Message] = []
        controls: list[Message] = []
        ordered: list[Message] = []
        level_evidence = _LevelBoundaryEvidence()
        while True:
            message = self.receive()
            ordered.append(message)
            target = controls if message.control else observations
            target.append(message)
            if boundary is FlushBoundary.LEVEL_TRANSITION:
                level_evidence.apply(message.payload)
                if message.kind == "exit":
                    return ObservationBatch(
                        tuple(observations), tuple(controls), tuple(ordered)
                    )
            if message.kind == "input_mode":
                mode = message.payload.get("mode")
                if isinstance(mode, int):
                    self._input_readiness = (
                        _InputReadiness.READY if mode == 1 else _InputReadiness.BUSY
                    )
            if message.control and message.kind == "flush_messages":
                if boundary is FlushBoundary.LEVEL_TRANSITION:
                    if level_evidence.explicit_input:
                        return ObservationBatch(
                            tuple(observations), tuple(controls), tuple(ordered)
                        )
                    if not level_evidence.blocking_ui:
                        # Level generation emits mode 0 + flush before doing CPU
                        # work. Silence here is not permission to send another key.
                        # The socket timeout fails closed if no input evidence arrives.
                        continue
                    settle = _BLOCKING_UI_QUIET_PERIOD
                if (
                    boundary is FlushBoundary.INPUT_READY_OR_QUIESCENCE
                    and self._input_readiness is _InputReadiness.READY
                ):
                    return ObservationBatch(
                        tuple(observations), tuple(controls), tuple(ordered)
                    )
                readable, _, _ = select.select([self._socket], [], [], settle)
                if not readable:
                    return ObservationBatch(
                        tuple(observations), tuple(controls), tuple(ordered)
                    )

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self.client_socket.unlink(missing_ok=True)
        if self._owned_directory is not None:
            self._owned_directory.cleanup()
            self._owned_directory = None

    def __enter__(self) -> WebtilesTransport:
        self.connect()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
