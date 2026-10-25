"""Train NuGAT from explicit cached-feature manifests; see docs/usage.md."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from nugat.training import train_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate", action="store_true", help="Report validation Nash loss, not M.J.")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        parser.error("The config must be a YAML mapping")
    summary = train_runs(config, args.manifest, args.output, validation=args.validate)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
