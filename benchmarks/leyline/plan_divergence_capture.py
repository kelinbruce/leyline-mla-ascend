#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Plan a bounded rerun for full/Leyline first-divergence logits."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.leyline.evidence import file_evidence  # noqa: E402


def build_plan(
    correctness: dict[str, Any],
    *,
    max_step: int = 32,
    case_ids: set[str] | None = None,
    vocab_size: int = 102400,
    ranks: int = 4,
) -> dict[str, Any]:
    if max_step < 1 or max_step > 256:
        raise ValueError("max_step must be between 1 and 256")
    if vocab_size < 2 or ranks < 1:
        raise ValueError("vocab_size and ranks must be positive")
    entries = []
    for case in correctness.get("cases", []):
        if case_ids and case["id"] not in case_ids:
            continue
        runs = case.get("repetitions") or [case]
        for repetition, run in enumerate(runs):
            pair = run.get("pairwise", {}).get("full_leyline", {})
            divergence = pair.get("first_divergence")
            step = pair.get("common_prefix_tokens")
            if not isinstance(step, int) or not divergence or step > max_step:
                continue
            full = (run.get("arms") or {}).get("full") or {}
            leyline = (run.get("arms") or {}).get("leyline") or {}
            honest = (run.get("arms") or {}).get("honest_edited") or {}
            entries.append(
                {
                    "case_id": case["id"],
                    "repetition": repetition,
                    "decode_step": step,
                    "source_prefix_token_ids": (full.get("output_token_ids") or [])[:step],
                    "source_request_ids": {
                        "full": full.get("request_id"),
                        "honest_edited": honest.get("request_id"),
                        "leyline": leyline.get("request_id"),
                    },
                }
            )
    capture_steps = sorted({0, *(entry["decode_step"] for entry in entries)})
    selected_cases = {entry["case_id"] for entry in entries}
    selected_repetitions = sum(
        len(case.get("repetitions") or [case])
        for case in correctness.get("cases", [])
        if case.get("id") in selected_cases
    )
    estimated_files = selected_repetitions * len(capture_steps) * 3 * ranks
    return {
        "schema_version": 1,
        "report_type": "leyline_divergence_capture_plan",
        "source_run_id": (correctness.get("evidence_identity") or {}).get("run_id"),
        "target_run_id": f"divergence-{uuid.uuid4().hex}",
        "max_step": max_step,
        "capture_steps": capture_steps,
        "case_ids": sorted(selected_cases),
        "entries": entries,
        "estimated_capture": {
            "arms": ["full", "honest_edited", "leyline"],
            "ranks": ranks,
            "vocab_size": vocab_size,
            "files": estimated_files,
            "uncompressed_bytes": estimated_files * vocab_size * 4,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--correctness", type=Path, required=True)
    parser.add_argument("--max-step", type=int, default=32)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--vocab-size", type=int, default=102400)
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    correctness = json.loads(args.correctness.read_text())
    report = build_plan(
        correctness,
        max_step=args.max_step,
        case_ids=set(args.case) or None,
        vocab_size=args.vocab_size,
        ranks=args.ranks,
    )
    report["source"] = file_evidence(args.correctness)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
