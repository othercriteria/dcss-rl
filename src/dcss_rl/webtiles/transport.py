"""Direct transport for the Unix-datagram protocol spoken by a DCSS process.

This intentionally bypasses the browser-facing Tornado server. DCSS remains an
unmodified upstream binary, while rollout workers avoid HTTP/WebSocket overhead and
account-management machinery that is irrelevant to local training.
"""

from __future__ import annotations

import json
import socket
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from dcss_rl.schema import JsonObject


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
        timeout: float = 10.0,
        client_directory: Path | None = None,
    ) -> None:
        self.game_socket = Path(game_socket)
        self.timeout = timeout
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

    def connect(self, *, primary: bool = True) -> None:
        """Wait for DCSS's socket, bind locally, and send the attach handshake."""
        if self._socket is not None:
            raise RuntimeError("WebTiles transport is already connected")

        deadline = time.monotonic() + self.timeout
        while not self.game_socket.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"DCSS socket did not appear: {self.game_socket}")
            time.sleep(0.01)

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

    def receive_until_flush(self) -> ObservationBatch:
        """Collect the complete state delta emitted before the next input boundary."""
        observations: list[Message] = []
        controls: list[Message] = []
        ordered: list[Message] = []
        while True:
            message = self.receive()
            ordered.append(message)
            target = controls if message.control else observations
            target.append(message)
            if message.control and message.kind == "flush_messages":
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
