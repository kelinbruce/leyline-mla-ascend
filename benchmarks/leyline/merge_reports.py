#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Merge reports collected across mutually exclusive TP4 server modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = [json.loads(path.read_text()) for path in args.reports]
    merged_cases: dict[str, dict[str, Any]] = {}
    for document in documents:
        for case in document.get("cases", []):
            target = merged_cases.setdefault(
                case["id"],
                {
                    "id": case["id"],
                    "category": case["category"],
                    "prompt_tokens": case["prompt_tokens"],
                    "oracle": case["oracle"],
                    "arms": {},
                    "counterfactuals": [],
                },
            )
            target["arms"].update(case.get("arms", {}))
            if case.get("counterfactuals"):
                target["counterfactuals"] = case["counterfactuals"]

    for case in merged_cases.values():
        arms = case["arms"]
        full_ok = arms.get("full", {}).get("matches_oracle", False)
        honest_ok = arms.get("honest_edited", {}).get("matches_oracle", False)
        counterfactual_ok = all(item["matches_oracle"] for item in case["counterfactuals"])
        declared = case["category"] in {"admissible", "counterfactual_admissible"}
        case["gates"] = {
            "full_matches": full_ok,
            "honest_edited_matches": honest_ok,
            "counterfactuals_match": counterfactual_ok,
            "admitted": bool(declared and full_ok and honest_ok and counterfactual_ok),
            "leyline_matches": arms.get("leyline", {}).get("matches_oracle", False),
        }

    merged = {
        "schema_version": 1,
        "sources": [str(path.resolve()) for path in args.reports],
        "environments": [doc["environment"] for doc in documents if "environment" in doc],
        "cases": list(merged_cases.values()),
        "performance": [item for doc in documents for item in doc.get("performance", [])],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
