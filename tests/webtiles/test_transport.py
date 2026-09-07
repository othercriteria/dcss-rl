import json
import socket
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from dcss_rl.webtiles import WebtilesTransport


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
