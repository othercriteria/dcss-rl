from pathlib import Path

import pytest

from dcss_rl.observation import ObservationReducer
from dcss_rl.units import GameSeed, Keycode
from dcss_rl.webtiles import GameConfig, ManagedGame

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
