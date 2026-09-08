"""Explicit checkpoint migration for a zero-update preprocessing experiment."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from dcss_rl.features import FEATURE_SPEC_VERSION
from dcss_rl.learned import (
    FeatureMigrationMetadata,
    LearnedPolicy,
    align_feature_spec,
    save_checkpoint,
)
from dcss_rl.units import FeatureSpecVersion


def migrate_features(
    source: Path, output: Path, *, target: FeatureSpecVersion, policy_id: str
) -> FeatureMigrationMetadata:
    if output.exists():
        raise FileExistsError(f"migration output already exists: {output}")
    policy = LearnedPolicy(source)
    metadata = FeatureMigrationMetadata(
        "preprocessing-only",
        policy.checkpoint_id,
        policy.model.config.feature_spec_version,
        target,
        tuple(
            (
                name,
                hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest(),
            )
            for name in (
                "features.py",
                "terrain.py",
                "learned.py",
                "feature_migration.py",
            )
        ),
    )
    save_checkpoint(
        output,
        model=align_feature_spec(policy.model, target),
        policy_id=policy_id,
        training_metadata=metadata,
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--feature-spec", type=int, default=FEATURE_SPEC_VERSION)
    args = parser.parse_args()
    metadata = migrate_features(
        args.checkpoint,
        args.output,
        target=FeatureSpecVersion(args.feature_spec),
        policy_id=args.policy_id,
    )
    print(json.dumps(asdict(metadata), indent=2))


if __name__ == "__main__":
    main()
