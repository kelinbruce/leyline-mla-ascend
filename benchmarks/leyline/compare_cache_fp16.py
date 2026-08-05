#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Qualify a 310P FP16 Leyline capture against analytical and native references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vllm_ascend.distributed.kv_transfer.leyline.reference import (
    rotate_kpe_half_split,
    unit_delta_cos_sin,
)


def _metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    absolute = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    denominator = np.maximum(np.abs(expected.astype(np.float32)), np.finfo(np.float32).tiny)
    relative = absolute / denominator
    return {
        "max_abs_error": float(absolute.max(initial=0)),
        "max_rel_error": float(relative.max(initial=0)),
        "p50_abs_error": float(np.percentile(absolute, 50)),
        "p99_abs_error": float(np.percentile(absolute, 99)),
        "p99_9_abs_error": float(np.percentile(absolute, 99.9)),
    }


def _parse_int_set(value: str) -> set[int]:
    return {int(item) for item in value.split(",") if item}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-atol", type=float)
    parser.add_argument("--reference-rtol", type=float)
    parser.add_argument("--native-atol", type=float)
    parser.add_argument("--native-rtol", type=float)
    parser.add_argument("--required-deltas", default="0,1,127,128,129,1024")
    parser.add_argument("--expected-ranks", default="0,1,2,3")
    parser.add_argument("--expected-layers", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.capture) as capture:
        required_keys = {
            "source_ckv",
            "actual_ckv",
            "source_kpe",
            "actual_kpe",
            "honest_kpe",
            "old_positions",
            "new_positions",
            "inv_freq",
            "rank_ids",
            "layer_ids",
        }
        missing_keys = sorted(required_keys - set(capture.files))
        if missing_keys:
            raise ValueError("capture is missing required arrays: " + ", ".join(missing_keys))
        arrays = {key: capture[key] for key in required_keys}

    source_ckv = arrays["source_ckv"]
    actual_ckv = arrays["actual_ckv"]
    source_kpe = arrays["source_kpe"].astype(np.float32)
    actual_kpe = arrays["actual_kpe"].astype(np.float32)
    honest_kpe = arrays["honest_kpe"].astype(np.float32)
    old_positions = arrays["old_positions"].astype(np.int64)
    new_positions = arrays["new_positions"].astype(np.int64)
    inv_freq = arrays["inv_freq"].astype(np.float32)
    rank_ids = set(int(value) for value in np.unique(arrays["rank_ids"]))
    layer_ids = set(int(value) for value in np.unique(arrays["layer_ids"]))

    cos, sin = unit_delta_cos_sin(old_positions, new_positions, inv_freq)
    analytical_kpe = rotate_kpe_half_split(source_kpe, cos, sin)
    observed_deltas = set(int(value) for value in old_positions - new_positions)
    required_deltas = _parse_int_set(args.required_deltas)
    expected_ranks = _parse_int_set(args.expected_ranks)
    expected_layers = set(range(args.expected_layers)) if args.expected_layers is not None else layer_ids

    thresholds = {
        "reference_atol": args.reference_atol,
        "reference_rtol": args.reference_rtol,
        "native_atol": args.native_atol,
        "native_rtol": args.native_rtol,
    }
    thresholds_provided = all(value is not None for value in thresholds.values())
    reference_allclose = bool(
        thresholds_provided
        and np.allclose(
            actual_kpe,
            analytical_kpe,
            atol=args.reference_atol,
            rtol=args.reference_rtol,
        )
    )
    native_allclose = bool(
        thresholds_provided
        and np.allclose(
            actual_kpe,
            honest_kpe,
            atol=args.native_atol,
            rtol=args.native_rtol,
        )
    )
    report = {
        "schema_version": 1,
        "capture": str(args.capture.resolve()),
        "rows": int(old_positions.size),
        "ckv_bitwise_equal": bool(np.array_equal(source_ckv, actual_ckv)),
        "analytical_reference": _metrics(actual_kpe, analytical_kpe),
        "native_recompute_reference": _metrics(actual_kpe, honest_kpe),
        "observed_deltas": sorted(observed_deltas),
        "missing_deltas": sorted(required_deltas - observed_deltas),
        "observed_ranks": sorted(rank_ids),
        "missing_ranks": sorted(expected_ranks - rank_ids),
        "observed_layers": sorted(layer_ids),
        "missing_layers": sorted(expected_layers - layer_ids),
        "thresholds": thresholds,
        "thresholds_provided": thresholds_provided,
        "reference_allclose": reference_allclose,
        "native_allclose": native_allclose,
    }
    report["passed"] = bool(
        report["ckv_bitwise_equal"]
        and reference_allclose
        and native_allclose
        and not report["missing_deltas"]
        and not report["missing_ranks"]
        and not report["missing_layers"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
