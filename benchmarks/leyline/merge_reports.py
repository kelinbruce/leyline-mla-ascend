#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Merge reports collected across mutually exclusive TP4 server modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.leyline.run_validation import evaluate_case_results


def _stable_identity(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _stable_identity(item)
            for key, item in value.items()
            if key != "hashing_seconds"
        }
    if isinstance(value, list):
        return [_stable_identity(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def merge_documents(documents: list[dict[str, Any]], sources: list[str]) -> dict[str, Any]:
    if not documents or any(document.get("schema_version") != 2 for document in documents):
        raise ValueError("only schema-v2 reports can be merged")
    contracts = [document.get("evaluation_contract") for document in documents]
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise ValueError("reports use mixed evaluation contracts")
    identities = [_stable_identity(document.get("checkpoint_identity")) for document in documents]
    if not identities[0] or any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("reports have missing or mixed checkpoint identities")
    preflights = [document.get("preflight") for document in documents]
    first_preflight_passed = bool(preflights[0] and preflights[0].get("passed"))
    if any(
        bool(item and item.get("passed")) != first_preflight_passed
        for item in preflights[1:]
    ):
        raise ValueError("reports disagree on baseline preflight status")
    preflight_contracts = [
        {
            "required": item.get("required") if item else None,
            "passed": item.get("passed") if item else None,
            "prompt_format": item.get("prompt_format") if item else None,
            "oracle": item.get("oracle") if item else None,
            "prompt_token_ids": item.get("prompt_token_ids") if item else None,
            "structured": ((item.get("result") or {}).get("structured") if item else None),
            "output_token_ids": ((item.get("result") or {}).get("output_token_ids") if item else None),
        }
        for item in preflights
    ]
    if any(item != preflight_contracts[0] for item in preflight_contracts[1:]):
        raise ValueError("reports use mixed preflight evidence")
    merged_cases: dict[str, dict[str, Any]] = {}
    for document in documents:
        for case in document.get("cases", []):
            target = merged_cases.setdefault(
                case["id"],
                {
                    "id": case["id"],
                    "category": case["category"],
                    "family": case.get("family"),
                    "claim_type": case.get("claim_type"),
                    "prompt_tokens": case["prompt_tokens"],
                    "oracle": case.get("oracle"),
                    "expected_completion": case.get("expected_completion"),
                    "target_token_ids": case.get("target_token_ids", []),
                    "evaluation": case["evaluation"],
                    "arms": {},
                    "counterfactuals": [],
                },
            )
            target["arms"].update(case.get("arms", {}))
            if case.get("counterfactuals"):
                target["counterfactuals"] = case["counterfactuals"]

    contract = contracts[0]
    for case in merged_cases.values():
        evaluation = evaluate_case_results(
            case,
            case["arms"],
            case["counterfactuals"],
            mode=contract["mode"],
            reference_tokens=int(case["evaluation"].get("reference_tokens") or 1),
            preflight_passed=bool(preflights[0] and preflights[0].get("passed")),
            target_token_ids=tuple(case.get("target_token_ids", [])),
        )
        case["gates"] = evaluation["gates"]
        case["leyline_execution"] = evaluation["leyline_execution"]
        case["pairwise"] = evaluation["pairwise"]

    return {
        "schema_version": 2,
        "sources": sources,
        "evaluation_contract": contract,
        "checkpoint_identity": identities[0],
        "preflight": preflights[0],
        "environments": [doc["environment"] for doc in documents if "environment" in doc],
        "cases": list(merged_cases.values()),
        "performance": [item for doc in documents for item in doc.get("performance", [])],
    }


def main() -> None:
    args = parse_args()
    documents = [json.loads(path.read_text()) for path in args.reports]
    merged = merge_documents(documents, [str(path.resolve()) for path in args.reports])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
