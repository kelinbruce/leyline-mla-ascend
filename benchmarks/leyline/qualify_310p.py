#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Combine all 310P Leyline hardware gates into one fail-closed record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TARGET_BASELINE = "05e095a202bdcfef4da61168eae34bfd3b99da13"
EXPECTED_RANKS = {0, 1, 2, 3}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _e2e_passed(report: dict[str, Any]) -> bool:
    admissible = [case for case in report.get("cases", []) if case.get("category") == "admissible"]
    return bool(
        admissible
        and all(
            case.get("gates", {}).get("admitted")
            and case.get("gates", {}).get("leyline_matches")
            for case in admissible
        )
    )


def build_qualification(
    probes: list[dict[str, Any]],
    numerical: dict[str, Any],
    e2e: dict[str, Any],
    rollback: dict[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    if not probes:
        blockers.append("operator_probes_missing")
        first_probe: dict[str, Any] = {}
    else:
        first_probe = probes[0]

    ranks = {
        int(probe.get("topology", {}).get("rank"))
        for probe in probes
        if probe.get("topology", {}).get("rank") is not None
    }
    if ranks != EXPECTED_RANKS:
        blockers.append("operator_probes_incomplete_tp4")
    if any(probe.get("status") != "passed" for probe in probes):
        blockers.append("operator_probe_failed")

    identity_fields = [
        "vllm_ascend_baseline",
        "checkout_head",
        "vllm_checkout_head",
        "model",
        "topology",
        "cache",
        "torch",
        "torch_npu",
        "cann",
    ]
    inconsistent = [
        field
        for field in identity_fields
        if probes and any(probe.get(field) != first_probe.get(field) for probe in probes[1:])
    ]
    # Rank is expected to differ; compare topology without it separately.
    if "topology" in inconsistent:
        topologies = [
            {key: value for key, value in probe.get("topology", {}).items() if key != "rank"}
            for probe in probes
        ]
        if topologies and all(value == topologies[0] for value in topologies[1:]):
            inconsistent.remove("topology")
    if inconsistent:
        blockers.append("operator_probe_identity_mismatch:" + ",".join(inconsistent))

    if first_probe.get("vllm_ascend_baseline") != TARGET_BASELINE:
        blockers.append("unexpected_vllm_ascend_baseline")
    if any(probe.get("checkout_status") for probe in probes):
        blockers.append("vllm_ascend_checkout_dirty")
    if not first_probe.get("vllm_checkout_head"):
        blockers.append("vllm_checkout_head_missing")
    if not first_probe.get("cann"):
        blockers.append("cann_version_missing")
    model = first_probe.get("model") or {}
    if not model.get("revision") or str(model.get("revision")).startswith("PIN_"):
        blockers.append("model_revision_not_pinned")
    if not model.get("tokenizer_revision") or str(model.get("tokenizer_revision")).startswith("PIN_"):
        blockers.append("tokenizer_revision_not_pinned")
    if not numerical.get("passed"):
        blockers.append("fp16_numerical_gate_failed")
    if not _e2e_passed(e2e):
        blockers.append("e2e_correctness_gate_failed")
    if rollback.get("passed") is not True:
        blockers.append("rollback_gate_failed")

    gates = {
        "operator_probes_tp4": bool(probes) and ranks == EXPECTED_RANKS and all(
            probe.get("status") == "passed" for probe in probes
        ),
        "fp16_numerical": numerical.get("passed") is True,
        "e2e_correctness": _e2e_passed(e2e),
        "transaction_rollback": rollback.get("passed") is True,
    }
    return {
        "schema_version": 1,
        "status": "passed" if not blockers else "failed",
        "vllm_ascend_baseline": TARGET_BASELINE,
        "checkout_head": first_probe.get("checkout_head"),
        "vllm_checkout_head": first_probe.get("vllm_checkout_head"),
        "model": first_probe.get("model"),
        "topology": {
            key: value
            for key, value in first_probe.get("topology", {}).items()
            if key != "rank"
        },
        "cache": first_probe.get("cache"),
        "qualified_ranks": sorted(ranks),
        "gates": gates,
        "blockers": blockers,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, action="append", required=True)
    parser.add_argument("--numerical", type=Path, required=True)
    parser.add_argument("--e2e", type=Path, required=True)
    parser.add_argument("--rollback", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_qualification(
        [_load(path) for path in args.probe],
        _load(args.numerical),
        _load(args.e2e),
        _load(args.rollback),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
