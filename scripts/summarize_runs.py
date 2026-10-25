"""Summarize independently scored runs of one dataset/model/protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nugat.evaluation import summarize_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="Per-seed JSON results from evaluate.py score")
    parser.add_argument("--expected-runs", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.runs]
    result = summarize_runs(records, expected_runs=args.expected_runs)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
