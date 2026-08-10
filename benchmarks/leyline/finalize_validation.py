#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Finalize immutable Leyline evidence without issuing inference requests."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.leyline.evidence import (  # noqa: E402
    FINAL_REPORT_SCHEMA_VERSION,
    align_cache_off,
    auxiliary_request_ids,
    cache_comparison_errors,
    checkpoint_identity,
    classify_case,
    environment_blockers,
    file_evidence,
    first_determining_gate,
    identity_conflicts,
    parse_validation_request_id,
    request_id_matches,
    request_index,
    retained_context_summary,
    rollback_report_errors,
    source_baseline_errors,
    stable_value,
)


def _read(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"evidence must be a JSON object: {path}")
    return value


def _source_entry(name: str, path: Path | None) -> dict[str, Any] | None:
    return {"name": name, **file_evidence(path)} if path is not None else None


def finalize_documents(
    correctness: dict[str, Any],
    *,
    environment: dict[str, Any] | None = None,
    cache_comparison: dict[str, Any] | None = None,
    first_token_logits: dict[str, Any] | None = None,
    divergence_logits: dict[str, Any] | None = None,
    cache_off: dict[str, Any] | None = None,
    rollback: dict[str, Any] | None = None,
    source_entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if correctness.get("schema_version") not in {2, 3}:
        raise ValueError("only schema-v2/v3 correctness reports can be finalized")
    source_requests = request_index(correctness)
    provenance_errors: list[Any] = []
    source_run_identity = correctness.get("evidence_identity") or {}
    if (
        source_run_identity.get("evidence_schema_version") != 1
        or not source_run_identity.get("run_id")
    ):
        provenance_errors.append("source_run_identity_invalid")
    effective_environment = environment or correctness.get("environment")
    if effective_environment is None:
        provenance_errors.append("environment_manifest_missing")
    elif correctness.get("environment") and environment and correctness["environment"] != environment:
        provenance_errors.append("environment_manifest_differs_from_source")
    if cache_off is not None:
        cache_off_run_identity = cache_off.get("evidence_identity") or {}
        if (
            cache_off_run_identity.get("evidence_schema_version") != 1
            or not cache_off_run_identity.get("run_id")
        ):
            provenance_errors.append("cache_off_run_identity_invalid")
        provenance_errors.extend(
            identity_conflicts(
                correctness,
                cache_off,
                allow_missing=False,
                allow_cache_mode_difference=True,
            )
        )
    if rollback is not None:
        rollback_identity = rollback.get("evidence_identity") or {}
        source_identity = checkpoint_identity(correctness)
        for field in (
            "checkpoint_identity",
            "imported_modules",
            "repositories",
            "runtime",
            "topology",
        ):
            observed = stable_value(
                rollback_identity.get(field), excluded_keys={"hashing_seconds"}
            )
            expected = stable_value(
                source_identity.get(field), excluded_keys={"hashing_seconds"}
            )
            if observed is None or expected is None:
                provenance_errors.append(f"rollback_identity_missing:{field}")
            elif observed != expected:
                provenance_errors.append(
                    {
                        "source": "rollback",
                        "field": field,
                        "expected": expected,
                        "observed": observed,
                    }
                )
    for label, document in (
        ("cache_comparison", cache_comparison),
        ("first_token_logits", first_token_logits),
        ("divergence_logits", divergence_logits),
    ):
        if document is None:
            continue
        observed_request_ids = auxiliary_request_ids(document)
        unknown = {
            observed
            for observed in observed_request_ids
            if not any(
                request_id_matches(observed, expected) for expected in source_requests
            )
        }
        diagnostic_run_id = (document.get("evidence_identity") or {}).get("run_id")
        linked_diagnostics = set(document.get("linked_diagnostic_request_ids") or [])
        valid_diagnostics = set()
        for request_id in linked_diagnostics:
            parsed = parse_validation_request_id(request_id)
            if parsed is not None and parsed["run_id"] == diagnostic_run_id:
                valid_diagnostics.update(
                    observed
                    for observed in unknown
                    if request_id_matches(observed, request_id)
                )
            elif request_id not in source_requests:
                provenance_errors.append(
                    {"source": label, "invalid_diagnostic_request_id": request_id}
                )
        unknown -= valid_diagnostics
        if unknown:
            provenance_errors.append({"source": label, "unknown_request_ids": sorted(unknown)})

    cache_off_aligned: dict[str, Any] = {}
    cache_off_errors: list[str] = []
    if cache_off is not None:
        cache_off_aligned, cache_off_errors = align_cache_off(correctness, cache_off)
    provenance_errors.extend(cache_off_errors)
    baseline_errors = source_baseline_errors(correctness)

    env_errors = environment_blockers(effective_environment)
    environment_state = "failed" if env_errors or provenance_errors else "passed"
    numerical_errors = (
        [] if cache_comparison is None else cache_comparison_errors(cache_comparison)
    )
    expected_tp_size = int(
        ((checkpoint_identity(correctness).get("runtime") or {}).get("tensor_parallel_size"))
        or 0
    )
    if cache_comparison is not None and expected_tp_size:
        observed_ranks = {
            int(capture["rank"])
            for capture in cache_comparison.get("captures") or []
            if capture.get("rank") is not None
        }
        if observed_ranks != set(range(expected_tp_size)):
            numerical_errors.append("cache_comparison_tp_rank_set_mismatch")
    rollback_errors = [] if rollback is None else rollback_report_errors(rollback)
    numerical_state = (
        "missing"
        if cache_comparison is None
        else ("passed" if not numerical_errors else "failed")
    )
    rollback_state = (
        "missing"
        if rollback is None
        else ("passed" if not rollback_errors else "failed")
    )
    cache_off_state = "missing" if cache_off is None else ("failed" if cache_off_errors else "passed")
    execution_passed = bool(correctness.get("cases")) and all(
        case.get("gates", {}).get("leyline_execution_valid") is True
        for case in correctness.get("cases", [])
        if case.get("claim_type") not in {"negative_control"}
    )
    baseline_passed = cache_off_state == "passed" and not baseline_errors and all(
        (not case.get("gates", {}).get("admitted"))
        or case.get("stability", {}).get("execution_stable", True)
        for case in correctness.get("cases", [])
    )
    leyline_target_passed = all(
        case.get("gates", {}).get("leyline_accepted") is True
        for case in correctness.get("cases", [])
        if case.get("gates", {}).get("admitted") is True
    )
    gates = {
        "environment": environment_state,
        "execution": "passed" if execution_passed else "failed",
        "numerical": numerical_state,
        "rollback": rollback_state,
        "baselines": (
            "passed"
            if baseline_passed
            else ("missing" if cache_off_state == "missing" else "failed")
        ),
        "leyline_target": "passed" if leyline_target_passed else "failed",
    }
    case_classifications = {
        case["id"]: classify_case(
            case,
            environment_state=environment_state,
            numerical_state=numerical_state,
            rollback_state=rollback_state,
            baseline_state=gates["baselines"],
        )
        for case in correctness.get("cases", [])
    }
    retained = retained_context_summary(correctness)
    logit_by_case: dict[str, list[dict[str, Any]]] = {}
    for comparison in (first_token_logits or {}).get("comparisons", []):
        logit_by_case.setdefault(str(comparison.get("case")), []).append(
            deepcopy(comparison)
        )
    for entry in retained["cases"]:
        entry["raw_logit_comparisons"] = logit_by_case.get(entry["case_id"], [])
    determining = first_determining_gate(gates)
    passed = determining is None
    return {
        "schema_version": FINAL_REPORT_SCHEMA_VERSION,
        "report_type": "leyline_joined_qualification",
        "evidence_identity": deepcopy(correctness.get("evidence_identity")),
        "sources": source_entries or [],
        "source_correctness": deepcopy(correctness),
        "cache_off_evidence": cache_off_aligned,
        "numerical_validation": deepcopy(cache_comparison),
        "raw_logit_validation": {
            "first_token": deepcopy(first_token_logits),
            "divergence": deepcopy(divergence_logits),
        },
        "rollback_validation": deepcopy(rollback),
        "retained_context_diagnostics": retained,
        "qualification": {
            "passed": passed,
            "determining_gate": determining,
            "gates": gates,
            "environment_blockers": env_errors,
            "provenance_errors": provenance_errors,
            "baseline_errors": baseline_errors,
            "numerical_errors": numerical_errors,
            "rollback_errors": rollback_errors,
            "case_classifications": case_classifications,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--correctness", type=Path, required=True)
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--cache-comparison", type=Path)
    parser.add_argument("--first-token-logits", type=Path)
    parser.add_argument("--divergence-logits", type=Path)
    parser.add_argument("--cache-off", type=Path)
    parser.add_argument("--rollback", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    named_paths = [
        ("correctness", args.correctness),
        ("environment", args.environment),
        ("cache_comparison", args.cache_comparison),
        ("first_token_logits", args.first_token_logits),
        ("divergence_logits", args.divergence_logits),
        ("cache_off", args.cache_off),
        ("rollback", args.rollback),
    ]
    report = finalize_documents(
        _read(args.correctness) or {},
        environment=_read(args.environment),
        cache_comparison=_read(args.cache_comparison),
        first_token_logits=_read(args.first_token_logits),
        divergence_logits=_read(args.divergence_logits),
        cache_off=_read(args.cache_off),
        rollback=_read(args.rollback),
        source_entries=[
            entry
            for name, path in named_paths
            if (entry := _source_entry(name, path)) is not None
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["qualification"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
