#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Compare captured Ascend MLA transformation output with the FP32 reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vllm_ascend.distributed.kv_transfer.leyline.reference import (
    rotate_kpe_half_split,
    unit_delta_cos_sin,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--required-deltas", default="0,1,127,128,129,1024")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.capture) as capture:
        source_ckv = capture["source_ckv"]
        actual_ckv = capture["actual_ckv"]
        source_kpe = capture["source_kpe"].astype(np.float32)
        actual_kpe = capture["actual_kpe"].astype(np.float32)
        old_positions = capture["old_positions"].astype(np.int64)
        new_positions = capture["new_positions"].astype(np.int64)
        inv_freq = capture["inv_freq"].astype(np.float32)

    cos, sin = unit_delta_cos_sin(old_positions, new_positions, inv_freq)
    expected_kpe = rotate_kpe_half_split(source_kpe, cos, sin)
    absolute = np.abs(actual_kpe - expected_kpe)
    denominator = np.maximum(np.abs(expected_kpe), np.finfo(np.float32).tiny)
    deltas = set(int(value) for value in (old_positions - new_positions))
    required = {int(value) for value in args.required_deltas.split(",") if value}
    report = {
        "schema_version": 1,
        "capture": str(args.capture.resolve()),
        "rows": int(old_positions.size),
        "observed_deltas": sorted(deltas),
        "required_deltas": sorted(required),
        "missing_deltas": sorted(required - deltas),
        "ckv_bitwise_equal": bool(np.array_equal(source_ckv, actual_ckv)),
        "kpe_max_abs_error": float(absolute.max(initial=0)),
        "kpe_max_rel_error": float((absolute / denominator).max(initial=0)),
        "kpe_allclose": bool(np.allclose(actual_kpe, expected_kpe, atol=args.atol, rtol=args.rtol)),
        "atol": args.atol,
        "rtol": args.rtol,
    }
    report["passed"] = (
        report["ckv_bitwise_equal"] and report["kpe_allclose"] and not report["missing_deltas"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
