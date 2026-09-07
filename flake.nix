{
  description = "Reinforcement learning research environment for Dungeon Crawl Stone Soup";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            # Python research tooling. GPU frameworks come from the uv lockfile;
            # this lets their wheels track host-driver-compatible CUDA runtimes.
            python313
            uv
            ruff
            poethepoet
            pre-commit

            # DCSS console/WebTiles build dependencies.
            git
            gcc
            gnumake
            pkg-config
            perl
            flex
            bison
            ncurses
            lua5_4
            sqlite
            zlib
            pcre2
            python313Packages.pyyaml

            # Inspection and process-control utilities used by adapters/tests.
            jq
            tmux
            which
          ];

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc
            pkgs.ncurses
            pkgs.lua5_4
            pkgs.sqlite
            pkgs.zlib
            pkgs.pcre2
          ];

          shellHook = ''
            export UV_CACHE_DIR="$PWD/.cache/uv"
            export XDG_CACHE_HOME="$PWD/.cache/xdg"
            export PYTHONNOUSERSITE=1
            unset PYTHONPATH PYTHONHOME

            if [ -d .venv ]; then
              export VIRTUAL_ENV="$PWD/.venv"
              export PATH="$VIRTUAL_ENV/bin:$PATH"
            fi

            # PyTorch/vLLM wheels need the host NVIDIA driver, not a second
            # driver supplied by Nix.
            if [ -d /run/opengl-driver/lib ]; then
              export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
            fi

            echo "DCSS RL environment ready (Python $(python --version 2>&1))."
            echo "Run 'uv sync' and then 'poe check'."
          '';
        };

        formatter = pkgs.nixpkgs-fmt;
      });
}
