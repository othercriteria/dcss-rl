import json
import select
import socket
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Never

import pytest

from dcss_rl.units import Seconds
from dcss_rl.webtiles import FlushBoundary, WebtilesTransport


@pytest.fixture
def game_socket(tmp_path: Path) -> Iterator[socket.socket]:
    server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    server.bind(str(tmp_path / "game.sock"))
    try:
        yield server
    finally:
        server.close()


def test_connect_and_send_key(game_socket: socket.socket, tmp_path: Path) -> None:
    game_path = Path(game_socket.getsockname())
    transport = WebtilesTransport(game_path, client_directory=tmp_path / "client")
    transport.connect()

    attach, client_address = game_socket.recvfrom(4096)
    assert json.loads(attach) == {"msg": "attach", "primary": True}

    transport.send_key("h")
    key, second_address = game_socket.recvfrom(4096)
    assert client_address == second_address
    assert json.loads(key) == {"msg": "key", "keycode": ord("h")}
    transport.close()


def test_receive_until_flush_reassembles_fragments(
    game_socket: socket.socket, tmp_path: Path
) -> None:
    game_path = Path(game_socket.getsockname())
    transport = WebtilesTransport(game_path, client_directory=tmp_path / "client")
    transport.connect()
    _, client_address = game_socket.recvfrom(4096)

    def emit() -> None:
        game_socket.sendto(b'{"msg":"pla', client_address)
        game_socket.sendto(b'yer","hp":12}\n', client_address)
        game_socket.sendto(b'*{"msg":"flush_messages"}\n', client_address)

    sender = threading.Thread(target=emit)
    sender.start()
    batch = transport.receive_until_flush()
    sender.join()
    transport.close()

    assert [message.kind for message in batch.observations] == ["player"]
    assert batch.observations[0].payload["hp"] == 12
    assert [message.kind for message in batch.controls] == ["flush_messages"]
    assert [message.kind for message in batch.messages] == ["player", "flush_messages"]


def test_receive_until_flush_coalesces_automatic_turns(
    game_socket: socket.socket, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    game_path = Path(game_socket.getsockname())
    transport = WebtilesTransport(
        game_path,
        input_quiet_period=Seconds(0.05),
        client_directory=tmp_path / "client",
    )
    transport.connect()
    _, client_address = game_socket.recvfrom(4096)

    def emit() -> None:
        game_socket.sendto(b'{"msg":"input_mode","mode":0}\n', client_address)
        game_socket.sendto(b'{"msg":"player","turn":1}\n', client_address)
        game_socket.sendto(b'*{"msg":"flush_messages"}\n', client_address)
        game_socket.sendto(b'{"msg":"player","turn":2}\n', client_address)
        game_socket.sendto(b'{"msg":"input_mode","mode":1}\n', client_address)
        game_socket.sendto(b'*{"msg":"flush_messages"}\n', client_address)

    sender = threading.Thread(target=emit)
    sender.start()
    select_calls = 0
    original_select = select.select

    def counting_select(rlist, wlist, xlist, timeout=None):
        nonlocal select_calls
        select_calls += 1
        return original_select(rlist, wlist, xlist, timeout)

    monkeypatch.setattr(select, "select", counting_select)
    batch = transport.receive_until_flush(
        boundary=FlushBoundary.INPUT_READY_OR_QUIESCENCE
    )
    sender.join()
    transport.close()

    assert [
        message.payload.get("turn")
        for message in batch.observations
        if message.kind == "player"
    ] == [1, 2]
    assert [message.kind for message in batch.controls] == [
        "flush_messages",
        "flush_messages",
    ]
    assert select_calls == 1


def test_input_ready_boundary_persists_across_batches(
    game_socket: socket.socket, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = WebtilesTransport(
        Path(game_socket.getsockname()), client_directory=tmp_path / "client"
    )
    transport.connect()
    _, client_address = game_socket.recvfrom(4096)
    game_socket.sendto(b'{"msg":"input_mode","mode":1}\n', client_address)
    game_socket.sendto(b'*{"msg":"flush_messages"}\n', client_address)
    first = transport.receive_until_flush(
        boundary=FlushBoundary.INPUT_READY_OR_QUIESCENCE
    )

    game_socket.sendto(b'{"msg":"player","turn":2}\n', client_address)
    game_socket.sendto(b'*{"msg":"flush_messages"}\n', client_address)

    def unexpected_select(*args: object) -> Never:
        raise AssertionError(
            f"ready input unexpectedly waited for quiescence: {args!r}"
        )

    monkeypatch.setattr(select, "select", unexpected_select)
    second = transport.receive_until_flush(
        boundary=FlushBoundary.INPUT_READY_OR_QUIESCENCE
    )
    transport.close()

    assert [message.kind for message in first.observations] == ["input_mode"]
    assert [message.payload.get("turn") for message in second.observations] == [2]


def test_send_key_rejects_strings_that_are_not_one_character(
    game_socket: socket.socket, tmp_path: Path
) -> None:
    transport = WebtilesTransport(
        Path(game_socket.getsockname()), client_directory=tmp_path / "client"
    )
    transport.connect()
    game_socket.recvfrom(4096)

    with pytest.raises(ValueError, match="exactly one character"):
        transport.send_key("wait")
    transport.close()


def test_full_state_probe_uses_upstream_spectator_message(
    game_socket: socket.socket, tmp_path: Path
) -> None:
    game_path = Path(game_socket.getsockname())
    transport = WebtilesTransport(game_path, client_directory=tmp_path / "client")
    transport.connect()
    game_socket.recvfrom(4096)

    assert not transport.output_available()
    transport.request_full_state()

    payload, _ = game_socket.recvfrom(4096)
    assert json.loads(payload) == {"msg": "spectator_joined"}
    transport.close()
