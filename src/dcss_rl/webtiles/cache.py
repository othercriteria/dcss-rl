"""Opt-in snapshots of DCSS's static db/des caches, copied into private saves."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NewType, cast

CacheDigest = NewType("CacheDigest", str)
CacheMemberName = NewType("CacheMemberName", str)
_SCHEMA = 1
_SUFFIXES = {"db": frozenset({".db"}), "des": frozenset({".idx", ".dsc", ".lux"})}


@dataclass(frozen=True, slots=True)
class StaticDataIdentity:
    binary: CacheDigest
    data: CacheDigest


@dataclass(frozen=True, slots=True)
class StaticCacheMember:
    name: CacheMemberName
    digest: CacheDigest


@dataclass(frozen=True, slots=True)
class StaticDataCache:
    """A verified snapshot; no live player files or shared writable cache files.

    Only the bundled ``binary.parent/dat`` layout and direct saves/db + saves/des
    cache layout are supported. A different build/data tree requires a new snapshot.
    Source games must be closed before capture; ManagedGame enforces this boundary.
    """

    directory: Path
    identity: StaticDataIdentity
    members: tuple[StaticCacheMember, ...]

    @classmethod
    def capture(
        cls,
        directory: Path,
        *,
        binary: Path,
        closed_save_directory: Path,
        source_identity: StaticDataIdentity,
    ) -> StaticDataCache:
        """Snapshot only known static file formats from a closed game's saves."""
        directory = Path(directory).resolve()
        if directory.exists():
            raise FileExistsError(directory)
        identity = static_data_identity(binary)
        if identity != source_identity:
            raise ValueError("DCSS binary/data changed since the source game started")
        sources: list[Path] = []
        for folder, suffixes in _SUFFIXES.items():
            source_directory = closed_save_directory / folder
            if source_directory.is_symlink() or not source_directory.is_dir():
                raise ValueError(
                    f"missing private static cache directory: {source_directory}"
                )
            sources.extend(
                path
                for path in sorted(source_directory.iterdir())
                if path.suffix in suffixes
            )
        members: list[StaticCacheMember] = []
        directory.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="static-cache-", dir=directory.parent
        ) as temporary:
            staged = Path(temporary) / "snapshot"
            staged.mkdir()
            for source in sources:
                if source.is_symlink() or not source.is_file():
                    raise ValueError(
                        f"static cache member must be a regular file: {source}"
                    )
                name = CacheMemberName(
                    source.relative_to(closed_save_directory).as_posix()
                )
                destination = staged / name
                destination.parent.mkdir(exist_ok=True)
                shutil.copy2(source, destination)
                members.append(StaticCacheMember(name, _digest(destination)))
            _validate_members(tuple(members))
            if static_data_identity(binary) != identity:
                raise ValueError(
                    "DCSS binary/data changed while capturing static cache"
                )
            manifest = {
                "schema": _SCHEMA,
                "identity": asdict(identity),
                "members": [asdict(member) for member in members],
            }
            (staged / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            staged.rename(directory)
        return cls(directory, identity, tuple(members))

    @classmethod
    def load(cls, directory: Path, *, binary: Path) -> StaticDataCache:
        """Validate the snapshot's manifest, source identity, and all copied bytes."""
        directory = Path(directory).resolve()
        try:
            decoded: object = json.loads((directory / "manifest.json").read_text())
            manifest = _object(decoded)
            if manifest.get("schema") != _SCHEMA:
                raise ValueError("unsupported static cache manifest")
            raw_identity = _object(manifest.get("identity"))
            identity = StaticDataIdentity(
                CacheDigest(_string(raw_identity.get("binary"))),
                CacheDigest(_string(raw_identity.get("data"))),
            )
            raw_members = manifest.get("members")
            if not isinstance(raw_members, list):
                raise ValueError("invalid static cache members")
            members = tuple(
                StaticCacheMember(
                    CacheMemberName(_string(_object(item).get("name"))),
                    CacheDigest(_string(_object(item).get("digest"))),
                )
                for item in raw_members
            )
        except (OSError, TypeError) as error:
            raise ValueError("unreadable static cache manifest") from error
        cache = cls(directory, identity, members)
        cache.validate(binary=binary)
        return cache

    def validate(self, *, binary: Path) -> None:
        _validate_members(self.members)
        if static_data_identity(binary) != self.identity:
            raise ValueError("static cache does not match DCSS binary/data")
        for member in self.members:
            source = self.directory / member.name
            if (
                source.parent.is_symlink()
                or source.is_symlink()
                or not source.is_file()
                or _digest(source) != member.digest
            ):
                raise ValueError(f"corrupt static cache member: {member.name}")

    def populate(self, save_directory: Path, *, binary: Path) -> None:
        """Copy into new private db/des directories; never overwrite or link files."""
        self.validate(binary=binary)
        for folder in _SUFFIXES:
            if (save_directory / folder).exists() or (
                save_directory / folder
            ).is_symlink():
                raise FileExistsError(save_directory / folder)
        save_directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="static-copy-", dir=save_directory
        ) as temporary:
            staged = Path(temporary)
            for folder in _SUFFIXES:
                (staged / folder).mkdir()
            for member in self.members:
                destination = staged / member.name
                shutil.copy2(self.directory / member.name, destination)
                if _digest(destination) != member.digest:
                    raise ValueError(
                        f"static cache changed while copying: {member.name}"
                    )
            for folder in _SUFFIXES:
                (staged / folder).rename(save_directory / folder)


def static_data_identity(binary: Path) -> StaticDataIdentity:
    """Hash executable plus all bundled data contents and upstream freshness stamps."""
    binary = Path(binary).resolve()
    data = binary.parent / "dat"
    if not data.is_dir():
        raise ValueError(f"static cache requires bundled data directory: {data}")
    digest = hashlib.sha256()
    files = sorted(path for path in data.rglob("*") if path.is_file())
    if not files:
        raise ValueError("static cache requires nonempty bundled DCSS data")
    for path in files:
        # DCSS embeds source mtimes in both database stamps and compiled maps.
        contract = (
            path.relative_to(data).as_posix(),
            path.stat().st_mtime_ns,
            _digest(path),
        )
        digest.update(json.dumps(contract, separators=(",", ":")).encode())
    return StaticDataIdentity(_digest(binary), CacheDigest(digest.hexdigest()))


def _validate_members(members: tuple[StaticCacheMember, ...]) -> None:
    names = {member.name for member in members}
    if len(names) != len(members):
        raise ValueError("duplicate static cache member")
    for name in names:
        path = Path(name)
        if (
            len(path.parts) != 2
            or path.parts[0] not in _SUFFIXES
            or path.suffix not in _SUFFIXES[path.parts[0]]
        ):
            raise ValueError(f"not a static db/des cache member: {name}")
    if not any(Path(name).parts[0] == "db" for name in names):
        raise ValueError("static cache has no databases")
    indices = {
        str(Path(name).with_suffix("")) for name in names if Path(name).suffix == ".idx"
    }
    maps = {
        str(Path(name).with_suffix("")) for name in names if Path(name).suffix == ".dsc"
    }
    preludes = {
        str(Path(name).with_suffix("")) for name in names if Path(name).suffix == ".lux"
    }
    if not indices or indices != maps or not preludes <= indices:
        raise ValueError("static map cache needs matching index/data members")


def _digest(path: Path) -> CacheDigest:
    with path.open("rb") as source:
        return CacheDigest(hashlib.file_digest(source, "sha256").hexdigest())


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("static cache manifest must contain objects")
    return cast(dict[str, object], value)


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("static cache manifest fields must be strings")
    return value
