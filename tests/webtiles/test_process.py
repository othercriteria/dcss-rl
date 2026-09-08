from pathlib import Path

import pytest

from dcss_rl.observation import ObservationReducer
from dcss_rl.units import GameSeed, Keycode, Seconds
from dcss_rl.webtiles import (
    FlushBoundary,
    GameConfig,
    ManagedGame,
    ObservationBatch,
    WebtilesTransport,
)

_DCSS_BINARY = Path("vendor/crawl/crawl-ref/source/crawl")


def test_missing_binary_fails_before_start(tmp_path: Path) -> None:
    game = ManagedGame(tmp_path / "missing-crawl", run_root=tmp_path / "run")

    with pytest.raises(FileNotFoundError, match="DCSS binary not found"):
        game.start()


def test_prepare_isolates_mutable_game_paths(tmp_path: Path) -> None:
    binary = tmp_path / "crawl"
    binary.touch()
    game = ManagedGame(binary, run_root=tmp_path / "run")

    rc_path = game._prepare()

    assert game.save_path.is_dir()
    assert game.morgue_path.is_dir()
    assert f"save_dir = {game.save_path}" in rc_path.read_text()
    assert f"morgue_dir = {game.morgue_path}" in rc_path.read_text()
    assert "view_delay = 0" in rc_path.read_text()
    assert "travel_delay = -1" in rc_path.read_text()
    assert "rest_delay = -1" in rc_path.read_text()
    assert "use_animations =\n" in rc_path.read_text()


def test_rejects_overlong_unix_socket_path_before_start(tmp_path: Path) -> None:
    binary = tmp_path / "crawl"
    binary.touch()
    game = ManagedGame(binary, run_root=tmp_path / ("long-path-" * 12))

    with pytest.raises(ValueError, match="Choose a shorter run/output path"):
        game.start()


@pytest.mark.parametrize(
    "key,level_transition",
    [
        ("<", True),
        (">", True),
        (60, True),
        (62, True),
        ("<", False),
        (">", False),
        (60, False),
        (62, False),
        ("h", False),
        ("o", False),
        ("5", False),
    ],
)
@pytest.mark.parametrize("has_output", [False, True])
def test_send_key_scopes_level_boundary_and_preserves_silent_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str | int,
    level_transition: bool,
    has_output: bool,
) -> None:
    game = ManagedGame(tmp_path / "crawl", run_root=tmp_path / "run")
    transport = WebtilesTransport(
        tmp_path / "game.sock", client_directory=tmp_path / "client"
    )
    game.transport = transport
    sent: list[str | int] = []
    probes: list[bool] = []
    boundaries: list[tuple[Seconds | None, FlushBoundary]] = []
    expected = ObservationBatch((), ())

    def receive(
        *,
        quiet_period: Seconds | None = None,
        boundary: FlushBoundary = FlushBoundary.QUIESCENCE,
    ) -> ObservationBatch:
        boundaries.append((quiet_period, boundary))
        return expected

    monkeypatch.setattr(transport, "send_key", sent.append)
    monkeypatch.setattr(transport, "output_available", lambda: has_output)
    monkeypatch.setattr(transport, "request_full_state", lambda: probes.append(True))
    monkeypatch.setattr(transport, "receive_until_flush", receive)
    assert game.send_key(key, level_transition=level_transition) is expected
    assert sent == [key]
    automatic = key in {"o", "5"}
    assert probes == ([True] if not has_output and not automatic else [])
    expected_boundary = (
        FlushBoundary.LEVEL_TRANSITION
        if level_transition
        else FlushBoundary.INPUT_READY_OR_QUIESCENCE
        if automatic
        else FlushBoundary.QUIESCENCE
    )
    assert boundaries == [(Seconds(0.5) if automatic else None, expected_boundary)]


@pytest.mark.integration
@pytest.mark.skipif(
    not _DCSS_BINARY.is_file(), reason="local DCSS binary has not been built"
)
def test_trunk_reaches_a_webtiles_input_boundary(tmp_path: Path) -> None:
    game = ManagedGame(
        _DCSS_BINARY,
        config=GameConfig(name="integration", seed=GameSeed(1)),
        run_root=tmp_path / "episode",
    )
    try:
        initial = game.start(initial_keycode=Keycode(ord("c")))
        kinds = {message.kind for message in initial.observations}
        control_kinds = {message.kind for message in initial.controls}

        assert "flush_messages" in control_kinds
        assert {"version", "options", "layout", "player", "map"}.issubset(kinds)
        assert game.process is not None
        assert game.process.poll() is None

        reducer = ObservationReducer()
        in_game = reducer.apply(initial)
        assert in_game.player["species"] == "Minotaur"
        assert in_game.player["depth"] == 1
        assert any(cell.get("g") == "@" for cell in in_game.cells)
        assert in_game.menu_type is None
    finally:
        game.close()

    assert not game.socket_path.exists()
