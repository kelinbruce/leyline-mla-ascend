#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Join Leyline raw first-token logits with a correctness report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.leyline.evidence import (  # noqa: E402
    parse_validation_request_id,
    request_id_matches,
)


def compare_logit_vectors(left: np.ndarray, right: np.ndarray, top_k: int = 10) -> dict[str, Any]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("logit vectors must be one-dimensional with matching shapes")
    if left.size < 2:
        raise ValueError("logit vectors must contain at least two values")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    left_ranked = np.argsort(left)[::-1]
    right_ranked = np.argsort(right)[::-1]
    left_top = left_ranked[:top_k]
    right_top = right_ranked[:top_k]
    left_shifted = left - left.max()
    right_shifted = right - right.max()
    left_prob = np.exp(left_shifted)
    right_prob = np.exp(right_shifted)
    left_prob /= left_prob.sum()
    right_prob /= right_prob.sum()
    midpoint = (left_prob + right_prob) / 2
    left_mask = left_prob > 0
    right_mask = right_prob > 0
    left_kl = np.sum(
        left_prob[left_mask]
        * np.log(left_prob[left_mask] / midpoint[left_mask]))
    right_kl = np.sum(
        right_prob[right_mask]
        * np.log(right_prob[right_mask] / midpoint[right_mask]))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return {
        "max_abs_difference": float(np.max(np.abs(left - right))),
        "cosine_similarity": float(np.dot(left, right) / denominator) if denominator else None,
        "jensen_shannon_divergence": float((left_kl + right_kl) / 2),
        "topk_overlap_token_ids": sorted(set(left_top.tolist()) & set(right_top.tolist())),
        "left_selected_token_id": int(left_top[0]),
        "right_selected_token_id": int(right_top[0]),
        "left_top1_top2_margin": float(left[left_ranked[0]] - left[left_ranked[1]]),
        "right_top1_top2_margin": float(right[right_ranked[0]] - right[right_ranked[1]]),
    }


def _load_captures(
    root: Path, request_id: str, decode_step: int = 0
) -> dict[int, tuple[np.ndarray, dict[str, Any]]]:
    suffix = "first-token-logits" if decode_step == 0 else f"step{decode_step:04d}-logits"
    matches = sorted(root.glob(f"*{request_id}*.rank*.{suffix}.npz"))
    if not matches:
        raise FileNotFoundError(f"raw logits missing for request {request_id}")
    captures: dict[int, tuple[np.ndarray, dict[str, Any]]] = {}
    for path in matches:
        with np.load(path) as capture:
            metadata = json.loads(str(capture["metadata_json"]))
            metadata["legacy_first_token"] = "decode_step" not in metadata
            metadata.setdefault("decode_step", 0)
            if not request_id_matches(str(metadata.get("request_id")), request_id):
                raise ValueError(
                    f"raw-logit metadata request mismatch for {path.name}"
                )
            if int(metadata["decode_step"]) != decode_step:
                raise ValueError(
                    f"raw-logit metadata decode-step mismatch for {path.name}"
                )
            structured = parse_validation_request_id(request_id)
            if structured is not None:
                for field in ("run_id", "case_id", "variant", "repetition", "arm"):
                    metadata.setdefault(field, structured[field])
                    if metadata[field] != structured[field]:
                        raise ValueError(
                            f"raw-logit metadata {field} mismatch for {path.name}"
                        )
            rank = int(metadata["rank"])
            if rank in captures:
                raise ValueError(f"duplicate raw-logit capture for request {request_id} rank {rank}")
            captures[rank] = (capture["logits"], metadata)
    return captures


def compare_report(
    report: dict[str, Any],
    capture_dir: Path,
    top_k: int = 10,
    divergence_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if divergence_plan is not None:
        report_run_id = (report.get("evidence_identity") or {}).get("run_id")
        target_run_id = divergence_plan.get("target_run_id")
        if report_run_id is not None and target_run_id != report_run_id:
            raise ValueError("diagnostic report does not match divergence-plan target run")
    comparisons = []
    missing_rank_pairs = []
    plan_entries = {
        (entry["case_id"], int(entry["repetition"])): entry
        for entry in (divergence_plan or {}).get("entries", [])
    }
    linked_source_ids = {
        request_id
        for entry in plan_entries.values()
        for request_id in entry.get("source_request_ids", {}).values()
        if request_id
    }
    diagnostic_request_ids = {
        result.get("request_id")
        for case in report.get("cases", [])
        for run in (case.get("repetitions") or [case])
        for result in (run.get("arms") or {}).values()
        if result.get("request_id")
    }
    correlatable_entries = []
    environment = report.get("environment") or {}
    runtime = environment.get("runtime_config") or environment.get("runtime") or {}
    tensor_parallel_size = int(runtime.get("tensor_parallel_size", 0) or 0)
    expected_ranks = set(range(tensor_parallel_size)) if tensor_parallel_size else None
    for case in report.get("cases", []):
        runs = case.get("repetitions") or [case]
        for repetition, run in enumerate(runs):
            arms = run.get("arms", {})
            full_id = (arms.get("full") or {}).get("request_id")
            if not full_id:
                continue
            plan_entry = plan_entries.get((case["id"], repetition))
            decode_step = int(plan_entry["decode_step"]) if plan_entry else 0
            current_pair = run.get("pairwise", {}).get("full_leyline", {})
            correlatable = True
            if plan_entry:
                correlatable = bool(
                    current_pair.get("common_prefix_tokens") == decode_step
                    and (current_pair.get("first_divergence") or {}).get("index")
                    == decode_step
                    and (run.get("arms", {}).get("full", {}).get("output_token_ids") or [])[
                        :decode_step
                    ]
                    == plan_entry.get("source_prefix_token_ids", [])
                )
                correlatable_entries.append(
                    {
                        "case": case["id"],
                        "repetition": repetition,
                        "decode_step": decode_step,
                        "correlatable": correlatable,
                    }
                )
            try:
                full_captures = _load_captures(capture_dir, full_id, decode_step)
            except FileNotFoundError:
                if plan_entry and not correlatable:
                    continue
                raise
            for arm in ("honest_edited", "leyline"):
                request_id = (arms.get(arm) or {}).get("request_id")
                if not request_id:
                    continue
                other_captures = _load_captures(capture_dir, request_id, decode_step)
                observed_intersection = set(full_captures) & set(other_captures)
                required_ranks = expected_ranks or (
                    set(full_captures) | set(other_captures)
                )
                missing_ranks = sorted(required_ranks - observed_intersection)
                if missing_ranks:
                    missing_rank_pairs.append(
                        {
                            "case": case["id"],
                            "repetition": repetition,
                            "pair": f"full_{arm}",
                            "decode_step": decode_step,
                            "correlatable": correlatable,
                            "ranks": missing_ranks,
                        }
                    )
                for rank in sorted(set(full_captures) & set(other_captures)):
                    full, full_metadata = full_captures[rank]
                    other, other_metadata = other_captures[rank]
                    comparisons.append(
                        {
                            "case": case["id"],
                            "repetition": repetition,
                            "rank": rank,
                            "pair": f"full_{arm}",
                            "correlatable": correlatable,
                            "left_provenance": full_metadata,
                            "right_provenance": other_metadata,
                            **compare_logit_vectors(full, other, top_k=top_k),
                        }
                    )
    capture_paths = list(capture_dir.glob("*.npz"))
    statuses = [json.loads(path.read_text()) for path in capture_dir.glob("capture-status.rank*.json")]
    step_summaries: dict[int, dict[str, Any]] = {}

    def step_summary(step: int) -> dict[str, Any]:
        return step_summaries.setdefault(
            step,
            {
                "decode_step": step,
                "files": 0,
                "bytes": 0,
                "comparisons": 0,
                "incomplete_reasons": [],
            },
        )

    for path in capture_paths:
        with np.load(path) as capture:
            metadata = json.loads(str(capture["metadata_json"]))
        if metadata.get("request_id") not in diagnostic_request_ids:
            continue
        step = int(metadata.get("decode_step", 0))
        summary = step_summary(step)
        summary["files"] += 1
        summary["bytes"] += path.stat().st_size
    for comparison in comparisons:
        step = int(comparison["left_provenance"].get("decode_step", 0))
        step_summary(step)["comparisons"] += 1
    for missing in missing_rank_pairs:
        step = int(missing["decode_step"])
        step_summary(step)["incomplete_reasons"].append(
            "missing_tensor_parallel_ranks"
        )
    for correlation in correlatable_entries:
        if not correlation["correlatable"]:
            step_summary(int(correlation["decode_step"]))["incomplete_reasons"].append(
                "source_divergence_not_reproduced"
            )
    budget_reasons = sorted(
        {
            str(status.get("reason"))
            for status in statuses
            if status.get("complete") is False and status.get("reason")
        }
    )
    for summary in step_summaries.values():
        summary["incomplete_reasons"] = sorted(
            set(summary["incomplete_reasons"]) | set(budget_reasons)
        )
    return {
        "schema_version": 2,
        "evidence_type": "internal_raw_logits",
        "evidence_identity": report.get("evidence_identity"),
        "comparisons": comparisons,
        "missing_rank_pairs": missing_rank_pairs,
        "linked_source_request_ids": sorted(linked_source_ids),
        "linked_diagnostic_request_ids": sorted(diagnostic_request_ids),
        "correlation": correlatable_entries,
        "capture_budget": {
            "files": sum(item["files"] for item in step_summaries.values()),
            "bytes": sum(item["bytes"] for item in step_summaries.values()),
            "statuses": statuses,
        },
        "steps": [step_summaries[step] for step in sorted(step_summaries)],
        "complete": bool(comparisons)
        and not missing_rank_pairs
        and not any(status.get("complete") is False for status in statuses)
        and all(item["correlatable"] for item in correlatable_entries),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--correctness", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--divergence-plan", type=Path)
    args = parser.parse_args()
    output = compare_report(
        json.loads(args.correctness.read_text()),
        args.captures,
        top_k=args.top_k,
        divergence_plan=(
            json.loads(args.divergence_plan.read_text()) if args.divergence_plan else None
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
