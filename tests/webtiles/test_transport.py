import json
import select
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Never

import pytest

from dcss_rl.schema import JsonObject
from dcss_rl.units import Seconds
from dcss_rl.webtiles import FlushBoundary, ManagedGame, WebtilesTransport


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


def test_level_transition_waits_past_busy_flush_and_old_ready_state(
    game_socket: socket.socket, tmp_path: Path
) -> None:
    with WebtilesTransport(
        Path(game_socket.getsockname()), client_directory=tmp_path / "client"
    ) as transport:
        _, address = game_socket.recvfrom(4096)
        game_socket.sendto(b'{"msg":"input_mode","mode":1}\n', address)
        game_socket.sendto(b'*{"msg":"flush_messages"}\n', address)
        transport.receive_until_flush(boundary=FlushBoundary.INPUT_READY_OR_QUIESCENCE)

        def emit() -> None:
            game_socket.sendto(b'{"msg":"input_mode","mode":0}\n', address)
            game_socket.sendto(b'*{"msg":"flush_messages"}\n', address)
            time.sleep(0.04)  # Four times the ordinary 10 ms quiet interval.
            game_socket.sendto(b'{"msg":"player","depth":2}\n', address)
            game_socket.sendto(b'{"msg":"input_mode","mode":1}\n', address)
            game_socket.sendto(b'*{"msg":"flush_messages"}\n', address)

        sender = threading.Thread(target=emit)
        sender.start()
        batch = transport.receive_until_flush(boundary=FlushBoundary.LEVEL_TRANSITION)
        sender.join()
        assert [m.kind for m in batch.messages] == [
            "input_mode",
            "flush_messages",
            "player",
            "input_mode",
            "flush_messages",
        ]
        assert batch.observations[1].payload["depth"] == 2


@pytest.mark.parametrize("mode", [1, 2, 3, 4, 5, 7, 8])
def test_level_transition_accepts_explicit_input_modes(
    game_socket: socket.socket, tmp_path: Path, mode: int
) -> None:
    with WebtilesTransport(
        Path(game_socket.getsockname()), client_directory=tmp_path / "client"
    ) as transport:
        _, address = game_socket.recvfrom(4096)
        game_socket.sendto(
            json.dumps({"msg": "input_mode", "mode": mode}).encode() + b"\n", address
        )
        game_socket.sendto(b'*{"msg":"flush_messages"}\n', address)
        batch = transport.receive_until_flush(boundary=FlushBoundary.LEVEL_TRANSITION)
        assert batch.observations[0].payload["mode"] == mode


@pytest.mark.parametrize(
    "payloads",
    [
        [{"msg": "menu", "type": "crt", "tag": "skills"}],
        [{"msg": "menu", "tag": "inventory"}],
        [{"msg": "ui-push", "type": "describe-item"}],
        [{"msg": "init_input"}],
        [{"msg": "ui-stack", "items": [{"msg": "menu", "type": "crt"}]}],
        [
            {"msg": "menu", "type": "crt"},
            {"msg": "ui-push", "type": "progress-bar"},
            {"msg": "ui-pop"},
        ],
    ],
)
def test_level_transition_accepts_current_blocking_ui(
    game_socket: socket.socket,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payloads: list[JsonObject],
) -> None:
    monkeypatch.setattr(
        "dcss_rl.webtiles.transport._BLOCKING_UI_QUIET_PERIOD", Seconds(0.001)
    )
    with WebtilesTransport(
        Path(game_socket.getsockname()), client_directory=tmp_path / "client"
    ) as transport:
        _, address = game_socket.recvfrom(4096)
        for payload in [{"msg": "input_mode", "mode": 0}, *payloads]:
            game_socket.sendto(json.dumps(payload).encode() + b"\n", address)
        game_socket.sendto(b'*{"msg":"flush_messages"}\n', address)
        batch = transport.receive_until_flush(boundary=FlushBoundary.LEVEL_TRANSITION)
        assert len(batch.observations) == len(payloads) + 1


@pytest.mark.parametrize(
    "payloads",
    [
        [],
        [{"msg": "input_mode", "mode": 6}],
        [{"msg": "ui-push", "type": "progress-bar"}],
        [{"msg": "ui-push", "type": "unknown-future-ui"}],
        [{"msg": "menu", "type": "crt"}, {"msg": "close_menu"}],
        [{"msg": "menu", "type": "crt"}, {"msg": "close_all_menus"}],
        [{"msg": "init_input"}, {"msg": "close_input"}],
        [{"msg": "init_input"}, {"msg": "ui-push", "type": "progress-bar"}],
        [{"msg": "menu", "type": "crt"}, {"msg": "ui-push", "type": "progress-bar"}],
        [{"msg": "menu", "type": "crt"}, {"msg": "ui-stack", "items": []}],
        [{"msg": "ui-state", "type": "crt"}],
    ],
)
def test_level_transition_fails_closed_without_current_input_evidence(
    game_socket: socket.socket, tmp_path: Path, payloads: list[JsonObject]
) -> None:
    with WebtilesTransport(
        Path(game_socket.getsockname()),
        timeout=Seconds(0.02),
        client_directory=tmp_path / "client",
    ) as transport:
        _, address = game_socket.recvfrom(4096)
        for payload in [{"msg": "input_mode", "mode": 0}, *payloads]:
            game_socket.sendto(json.dumps(payload).encode() + b"\n", address)
        game_socket.sendto(b'*{"msg":"flush_messages"}\n', address)
        with pytest.raises(TimeoutError):
            transport.receive_until_flush(boundary=FlushBoundary.LEVEL_TRANSITION)


def test_level_transition_returns_terminal_exit_without_input_mode_or_flush(
    game_socket: socket.socket, tmp_path: Path
) -> None:
    with WebtilesTransport(
        Path(game_socket.getsockname()), client_directory=tmp_path / "client"
    ) as transport:
        _, address = game_socket.recvfrom(4096)
        game_socket.sendto(b'{"msg":"input_mode","mode":0}\n', address)
        game_socket.sendto(b'*{"msg":"exit","reason":"dead"}\n', address)
        batch = transport.receive_until_flush(boundary=FlushBoundary.LEVEL_TRANSITION)
        assert batch.messages[-1].kind == "exit"


def test_silent_noop_stairs_collects_full_state_probe(
    game_socket: socket.socket, tmp_path: Path
) -> None:
    game = ManagedGame(tmp_path / "crawl", run_root=tmp_path / "run")
    with WebtilesTransport(
        Path(game_socket.getsockname()), client_directory=tmp_path / "client"
    ) as transport:
        game.transport = transport
        game_socket.recvfrom(4096)
        received: list[object] = []

        def emit() -> None:
            # A supported no-op command may emit nothing until the existing
            # spectator probe requests a forced current input_mode/full state.
            key, address = game_socket.recvfrom(4096)
            received.append(json.loads(key))
            probe, _ = game_socket.recvfrom(4096)
            received.append(json.loads(probe))
            game_socket.sendto(b'{"msg":"player","depth":1}\n', address)
            game_socket.sendto(b'{"msg":"ui-stack","items":[]}\n', address)
            game_socket.sendto(b'{"msg":"input_mode","mode":1}\n', address)
            game_socket.sendto(b'*{"msg":"flush_messages"}\n', address)

        sender = threading.Thread(target=emit)
        sender.start()
        batch = game.send_key(">", level_transition=True)
        sender.join()
        assert received == [{"msg": "key", "keycode": 62}, {"msg": "spectator_joined"}]
        assert batch.observations[0].payload["depth"] == 1
        assert batch.observations[-1].payload["mode"] == 1
