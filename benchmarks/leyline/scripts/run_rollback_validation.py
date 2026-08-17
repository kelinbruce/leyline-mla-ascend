#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run a guarded post-write Leyline rollback validation on a dedicated server."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.leyline.common.evidence import (  # noqa: E402
    ensure_run_id,
    environment_blockers,
    file_evidence,
    report_identity,
    validation_request_id,
)
from benchmarks.leyline.scripts.run_validation import (  # noqa: E402
    RAW,
    build_prompt_plan,
    expand_workload_cases,
    request_completion,
)


def _matches_target(result: dict[str, Any], target: tuple[int, ...]) -> bool:
    output = result.get("output_token_ids") or []
    return bool(target and output[: len(target)] == list(target))


def run_rollback(
    config: dict[str, Any],
    workloads: dict[str, Any],
    environment: dict[str, Any],
    tokenizer: Any,
    *,
    case_id: str,
    fail_rank: int,
    fail_after_layer: int,
) -> dict[str, Any]:
    if fail_rank < 0 or fail_after_layer < 0:
        raise ValueError("failure rank and layer must be non-negative")
    run_id = ensure_run_id(config)
    cases = {case["id"]: case for case in expand_workload_cases(workloads["cases"])}
    if case_id not in cases:
        raise ValueError(f"unknown rollback case: {case_id}")
    case = cases[case_id]
    plan = build_prompt_plan(
        tokenizer,
        case,
        prompt_format=RAW,
        block_size=int(config.get("block_size", 128)),
    )
    endpoints = config.get("arms") or {}
    honest_endpoint = endpoints.get("honest_edited") or endpoints.get("full")
    leyline_endpoint = endpoints.get("leyline")
    if not honest_endpoint or not leyline_endpoint:
        raise ValueError("rollback config requires honest/full and leyline endpoints")
    common = {
        "api_key": config.get("api_key"),
        "max_tokens": int(config.get("max_tokens", 64)),
        "top_logprobs": int(config.get("diagnostics", {}).get("top_logprobs", 0)),
    }
    honest = request_completion(
        honest_endpoint,
        config["model"],
        plan.edited,
        cache_salt=f"rollback-honest-{run_id}",
        request_id=validation_request_id(run_id, case_id, "rollback-honest", 0),
        **common,
    )
    session_id = f"rollback-{run_id}-{case_id}"
    cache_salt = f"rollback-transaction-{run_id}"
    record = request_completion(
        leyline_endpoint,
        config["model"],
        plan.full,
        cache_salt=cache_salt,
        kv_transfer_params={
            "leyline": {"version": 1, "action": "record", "session_id": session_id}
        },
        request_id=validation_request_id(run_id, case_id, "rollback-record", 0),
        **common,
    )
    amortize = request_completion(
        leyline_endpoint,
        config["model"],
        plan.edited,
        cache_salt=cache_salt,
        kv_transfer_params={
            "leyline": {
                "version": 1,
                "action": "amortize",
                "session_id": session_id,
                "delete": {"start": plan.delete_start, "end": plan.delete_end},
                "fault_injection": {
                    "rank": fail_rank,
                    "layer": fail_after_layer,
                    "stage": "after_layer_write",
                },
            }
        },
        request_id=validation_request_id(run_id, case_id, "rollback-amortize", 0),
        **common,
    )
    record_meta = ((record.get("kv_transfer_params") or {}).get("leyline") or {})
    metadata = ((amortize.get("kv_transfer_params") or {}).get("leyline") or {})
    cleanup = metadata.get("cleanup") or {}
    accounting_complete = (
        int(metadata.get("local_apc_tokens", 0) or 0)
        + int(metadata.get("normal_prefill_tokens", 0) or 0)
        == len(plan.edited)
    )
    conditions = {
        "server_validation_opt_in": os.environ.get(
            "VLLM_ASCEND_LEYLINE_FAULT_INJECTION"
        )
        == "validation-only",
        "environment_valid": not environment_blockers(environment),
        "recorded": record_meta.get("recorded") is True,
        "injection_reached": metadata.get("injection_reached") is True,
        "injection_rank": metadata.get("injected_rank") == fail_rank,
        "injection_layer": metadata.get("injected_layer") == fail_after_layer,
        "post_write": int(metadata.get("destination_writes", 0) or 0) > 0,
        "not_applied": metadata.get("applied") is False,
        "transform_failed": metadata.get("fallback_reason") == "transform_failed",
        "destination_blocks_invalidated": int(
            metadata.get("invalidated_destination_blocks", 0) or 0
        )
        > 0,
        "honest_reprefill_accounting": accounting_complete,
        "honest_target": _matches_target(honest, plan.target_token_ids),
        "fallback_target": _matches_target(amortize, plan.target_token_ids),
        "cleanup_complete": bool(cleanup) and all(value == 0 for value in cleanup.values()),
    }
    identity = report_identity(
        {
            "config": config,
            "environment": environment,
            "checkpoint_identity": environment.get("model"),
            "corpus_id": workloads.get("corpus_id"),
            "workload_version": workloads.get("version"),
            "evaluation_contract": {
                "mode": config.get("evaluation", {}).get("mode"),
                "prompt_format": config.get("prompt_format"),
            },
        }
    )
    identity["run_id"] = run_id
    return {
        "schema_version": 1,
        "report_type": "leyline_rollback_validation",
        "evidence_identity": identity,
        "case_id": case_id,
        "injection": {
            "rank": fail_rank,
            "layer": fail_after_layer,
            "stage": "after_layer_write",
        },
        "request_ids": {
            "honest": honest.get("request_id"),
            "record": record.get("request_id"),
            "amortize": amortize.get("request_id"),
        },
        "conditions": conditions,
        "passed": all(conditions.values()),
        "record_metadata": record_meta,
        "amortize_metadata": metadata,
        "target_token_ids": list(plan.target_token_ids),
        "honest_output_token_ids": honest.get("output_token_ids"),
        "fallback_output_token_ids": amortize.get("output_token_ids"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    resource_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument(
        "--workloads",
        type=Path,
        default=resource_root / "workloads" / "workloads.base.json",
    )
    parser.add_argument("--case", required=True)
    parser.add_argument("--fail-rank", type=int, required=True)
    parser.add_argument("--fail-after-layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    environment = json.loads(args.environment.read_text())
    workloads = json.loads(args.workloads.read_text())
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["tokenizer"],
        revision=config.get("tokenizer_revision"),
        trust_remote_code=bool(config.get("trust_remote_code", False)),
    )
    report = run_rollback(
        config,
        workloads,
        environment,
        tokenizer,
        case_id=args.case,
        fail_rank=args.fail_rank,
        fail_after_layer=args.fail_after_layer,
    )
    report["sources"] = [
        {"name": "config", **file_evidence(args.config)},
        {"name": "environment", **file_evidence(args.environment)},
        {"name": "workloads", **file_evidence(args.workloads)},
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
