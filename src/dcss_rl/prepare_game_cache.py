"""Prepare an opt-in upstream static cache from one isolated, closed MiBe game."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict

from dcss_rl.observation import ObservationReducer
from dcss_rl.schema import ObservationData, RawMessageData
from dcss_rl.units import GameSeed, Keycode
from dcss_rl.webtiles.cache import StaticDataCache
from dcss_rl.webtiles.process import GameConfig, ManagedGame

_DEFAULT_BOOTSTRAP_SEED = GameSeed(3001)


class StaticCacheBootstrapData(TypedDict):
    binary: str
    seed: int
    setup_keycode: int
    raw_messages: list[RawMessageData]
    observation: ObservationData


def prepare_game_cache(
    *,
    binary: Path,
    output: Path,
    run_root: Path,
    seed: GameSeed = _DEFAULT_BOOTSTRAP_SEED,
) -> StaticDataCache:
    """Retain initial protocol evidence, close DCSS, then export only db/des files."""
    for path in (output, run_root):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    game = ManagedGame(
        binary,
        config=GameConfig(seed=seed),
        run_root=run_root,
        capture_static_cache=True,
    )
    try:
        initial = game.start(initial_keycode=Keycode(ord("c")))
        observation = ObservationReducer().apply(initial).to_dict()
        evidence = StaticCacheBootstrapData(
            binary=str(game.binary),
            seed=int(seed),
            setup_keycode=ord("c"),
            raw_messages=[
                {"control": message.control, "payload": message.payload}
                for message in initial.messages
            ],
            observation=observation,
        )
        (game.run_root / "initial.json").write_text(
            json.dumps(evidence, indent=2) + "\n"
        )
    finally:
        game.close()
    return game.export_static_cache(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=int(_DEFAULT_BOOTSTRAP_SEED))
    arguments = parser.parse_args()
    cache = prepare_game_cache(
        binary=arguments.binary,
        output=arguments.output,
        run_root=arguments.run_root,
        seed=GameSeed(arguments.seed),
    )
    print(f"Prepared {len(cache.members)} static files at {cache.directory}")


if __name__ == "__main__":
    main()
