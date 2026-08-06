#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Join Leyline raw first-token logits with a correctness report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


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
    root: Path, request_id: str
) -> dict[int, tuple[np.ndarray, dict[str, Any]]]:
    matches = sorted(root.glob(f"{request_id}.rank*.first-token-logits.npz"))
    if not matches:
        raise FileNotFoundError(f"raw logits missing for request {request_id}")
    captures: dict[int, tuple[np.ndarray, dict[str, Any]]] = {}
    for path in matches:
        with np.load(path) as capture:
            metadata = json.loads(str(capture["metadata_json"]))
            rank = int(metadata["rank"])
            if rank in captures:
                raise ValueError(f"duplicate raw-logit capture for request {request_id} rank {rank}")
            captures[rank] = (capture["logits"], metadata)
    return captures


def compare_report(report: dict[str, Any], capture_dir: Path, top_k: int = 10) -> dict[str, Any]:
    comparisons = []
    missing_rank_pairs = []
    for case in report.get("cases", []):
        runs = case.get("repetitions") or [case]
        for repetition, run in enumerate(runs):
            arms = run.get("arms", {})
            full_id = (arms.get("full") or {}).get("request_id")
            if not full_id:
                continue
            full_captures = _load_captures(capture_dir, full_id)
            for arm in ("honest_edited", "leyline"):
                request_id = (arms.get(arm) or {}).get("request_id")
                if not request_id:
                    continue
                other_captures = _load_captures(capture_dir, request_id)
                missing_ranks = sorted(
                    set(full_captures) ^ set(other_captures)
                )
                if missing_ranks:
                    missing_rank_pairs.append(
                        {
                            "case": case["id"],
                            "repetition": repetition,
                            "pair": f"full_{arm}",
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
                            "left_provenance": full_metadata,
                            "right_provenance": other_metadata,
                            **compare_logit_vectors(full, other, top_k=top_k),
                        }
                    )
    return {
        "schema_version": 2,
        "evidence_type": "internal_raw_logits",
        "comparisons": comparisons,
        "missing_rank_pairs": missing_rank_pairs,
        "complete": bool(comparisons) and not missing_rank_pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--correctness", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    output = compare_report(
        json.loads(args.correctness.read_text()), args.captures, top_k=args.top_k
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
