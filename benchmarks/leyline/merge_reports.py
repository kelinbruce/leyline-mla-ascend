#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Merge reports collected across mutually exclusive TP4 server modes."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.leyline.evidence import identity_conflicts  # noqa: E402
from benchmarks.leyline.run_validation import (  # noqa: E402
    _stability_summary,
    evaluate_case_results,
)


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
    contracts = [document.get("evaluation_contract") or {} for document in documents]
    semantic_contracts = [
        {
            key: contract.get(key)
            for key in ("mode", "prompt_format", "score_evidence")
            if key in contract
        }
        for contract in contracts
    ]
    if any(contract != semantic_contracts[0] for contract in semantic_contracts[1:]):
        raise ValueError("reports use mixed evaluation contracts")
    identities = [_stable_identity(document.get("checkpoint_identity")) for document in documents]
    if not identities[0] or any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("reports have missing or mixed checkpoint identities")
    for document in documents[1:]:
        conflicts = identity_conflicts(
            documents[0],
            document,
            allow_missing=True,
            allow_cache_mode_difference=True,
        )
        if conflicts:
            raise ValueError(f"reports have mixed evidence identity: {conflicts}")
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
            target = merged_cases.get(case["id"])
            if target is None:
                merged_cases[case["id"]] = deepcopy(case)
                continue
            target_runs = target.get("repetitions") or [target]
            incoming_runs = case.get("repetitions") or [case]
            if len(target_runs) != len(incoming_runs):
                raise ValueError(f"reports disagree on repetitions for {case['id']}")
            for target_run, incoming_run in zip(target_runs, incoming_runs):
                incoming_arms = incoming_run.get("arms", {})
                target_run.setdefault("arms", {}).update(deepcopy(incoming_arms))
                incoming_counterfactuals = incoming_run.get("counterfactuals") or []
                if incoming_counterfactuals:
                    # Cache-off is independent baseline evidence. Preserve it
                    # without replacing the connector-on variants.
                    if "cache_off" in incoming_arms:
                        target_run["cache_off_counterfactuals"] = deepcopy(
                            incoming_counterfactuals
                        )
                    else:
                        if target_run.get("counterfactuals") and len(
                            target_run["counterfactuals"]
                        ) != len(incoming_counterfactuals):
                            raise ValueError(
                                f"reports disagree on counterfactuals for {case['id']}"
                            )
                        target_run["counterfactuals"] = deepcopy(incoming_counterfactuals)

    contract = contracts[0]
    for case in merged_cases.values():
        runs = case.get("repetitions") or [case]
        for run in runs:
            evaluation = evaluate_case_results(
                case,
                run["arms"],
                run.get("counterfactuals") or [],
                mode=contract["mode"],
                reference_tokens=int(case["evaluation"].get("reference_tokens") or 1),
                preflight_passed=bool(preflights[0] and preflights[0].get("passed")),
                target_token_ids=tuple(case.get("target_token_ids", [])),
            )
            run["gates"] = evaluation["gates"]
            run["leyline_execution"] = evaluation["leyline_execution"]
            run["pairwise"] = evaluation["pairwise"]
        if case.get("repetitions"):
            case.update(
                {
                    "arms": deepcopy(runs[0]["arms"]),
                    "counterfactuals": deepcopy(runs[0].get("counterfactuals") or []),
                    "cache_off_counterfactuals": deepcopy(
                        runs[0].get("cache_off_counterfactuals") or []
                    ),
                    "gates": deepcopy(runs[0]["gates"]),
                    "leyline_execution": deepcopy(runs[0]["leyline_execution"]),
                    "pairwise": deepcopy(runs[0]["pairwise"]),
                    "stability": _stability_summary(runs),
                }
            )

    return {
        "schema_version": 2,
        "sources": sources,
        "evaluation_contract": contract,
        "checkpoint_identity": identities[0],
        "preflight": preflights[0],
        "environments": [doc["environment"] for doc in documents if "environment" in doc],
        "cases": list(merged_cases.values()),
        "evidence_identity": documents[0].get("evidence_identity"),
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
