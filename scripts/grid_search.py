from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from nugat.training import grid_search, load_metric_callback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric-callback")
    parser.add_argument("--metric-name")
    parser.add_argument("--maximize", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    grid = yaml.safe_load(args.grid.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(grid, dict):
        parser.error("Config and grid must be YAML mappings")
    callback = load_metric_callback(args.metric_callback) if args.metric_callback else None
    result = grid_search(config, args.manifest, grid, args.output,
                         metric_callback=callback, metric_name=args.metric_name,
                         maximize=args.maximize)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
