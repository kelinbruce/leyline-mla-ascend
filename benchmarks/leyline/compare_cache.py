#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Compare rank/layer Leyline captures with an independent FP32 reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from vllm_ascend.distributed.kv_transfer.leyline.reference import (
    rotate_kpe_half_split,
    unit_delta_cos_sin,
)


def _capture_paths(inputs: list[Path]) -> tuple[list[Path], list[dict[str, Any]]]:
    paths: list[Path] = []
    manifests: list[dict[str, Any]] = []
    for item in inputs:
        if item.is_dir():
            paths.extend(sorted(item.glob("*.npz")))
            manifests.extend(json.loads(path.read_text()) for path in sorted(item.glob("*.manifest.json")))
        elif item.suffix == ".json":
            manifest = json.loads(item.read_text())
            manifests.append(manifest)
            paths.extend(item.parent / entry["path"] for entry in manifest.get("captures", []))
        else:
            paths.append(item)
    return list(dict.fromkeys(path.resolve() for path in paths)), manifests


def compare_capture(path: Path, *, atol: float, rtol: float) -> dict[str, Any]:
    with np.load(path) as capture:
        source_ckv = capture["source_ckv"]
        actual_ckv = capture["actual_ckv"]
        source_kpe = capture["source_kpe"].astype(np.float32)
        actual_kpe = capture["actual_kpe"].astype(np.float32)
        old_positions = capture["old_positions"].astype(np.int64)
        new_positions = capture["new_positions"].astype(np.int64)
        inv_freq = capture["inv_freq"].astype(np.float32)
        source_slots = capture["source_slots"] if "source_slots" in capture else np.arange(len(source_ckv))
        destination_slots = (
            capture["destination_slots"] if "destination_slots" in capture else np.arange(len(source_ckv))
        )
        metadata = json.loads(str(capture["metadata_json"])) if "metadata_json" in capture else {}
    cos, sin = unit_delta_cos_sin(old_positions, new_positions, inv_freq)
    expected_kpe = rotate_kpe_half_split(source_kpe, cos, sin)
    absolute = np.abs(actual_kpe - expected_kpe)
    denominator = np.maximum(np.abs(expected_kpe), np.finfo(np.float32).tiny)
    ckv_mismatch_mask = np.any(source_ckv != actual_ckv, axis=tuple(range(1, source_ckv.ndim)))
    kpe_failure_mask = np.any(
        ~np.isclose(actual_kpe, expected_kpe, atol=atol, rtol=rtol),
        axis=tuple(range(1, actual_kpe.ndim)),
    )
    deltas = old_positions - new_positions

    def row_evidence(mask: np.ndarray) -> list[dict[str, int]]:
        return [
            {
                "row": int(index),
                "source_slot": int(source_slots[index]),
                "destination_slot": int(destination_slots[index]),
                "old_position": int(old_positions[index]),
                "new_position": int(new_positions[index]),
                "delta": int(deltas[index]),
            }
            for index in np.flatnonzero(mask)
        ]
    return {
        "capture": str(path),
        "layer": metadata.get("layer"),
        "rank": metadata.get("rank"),
        "rows": int(old_positions.size),
        "observed_deltas": sorted({int(value) for value in deltas}),
        "ckv_bitwise_equal": bool(np.array_equal(source_ckv, actual_ckv)),
        "ckv_mismatched_values": int(np.count_nonzero(source_ckv != actual_ckv)),
        "ckv_mismatch_rows": row_evidence(ckv_mismatch_mask),
        "kpe_max_abs_error": float(absolute.max(initial=0)),
        "kpe_mean_abs_error": float(absolute.mean()) if absolute.size else 0.0,
        "kpe_p95_abs_error": float(np.percentile(absolute, 95)) if absolute.size else 0.0,
        "kpe_p99_abs_error": float(np.percentile(absolute, 99)) if absolute.size else 0.0,
        "kpe_max_rel_error": float((absolute / denominator).max(initial=0)),
        "kpe_allclose": bool(np.allclose(actual_kpe, expected_kpe, atol=atol, rtol=rtol)),
        "kpe_failure_rows": row_evidence(kpe_failure_mask),
        "_absolute_errors": absolute,
    }


def compare_captures(
    inputs: list[Path],
    *,
    atol: float,
    rtol: float,
    required_deltas: set[int],
    expected_layers: set[str] | None = None,
    expected_ranks: set[int] | None = None,
) -> dict[str, Any]:
    paths, manifests = _capture_paths(inputs)
    captures = [compare_capture(path, atol=atol, rtol=rtol) for path in paths]
    all_absolute = (
        np.concatenate([capture.pop("_absolute_errors").reshape(-1) for capture in captures])
        if captures
        else np.asarray([], dtype=np.float32)
    )
    observed_deltas = {delta for capture in captures for delta in capture["observed_deltas"]}
    observed_pairs = {(capture["layer"], capture["rank"]) for capture in captures}
    manifest_layers = {
        layer for manifest in manifests for layer in manifest.get("expected_layers", [])
    }
    manifest_ranks = {
        int(rank) for manifest in manifests for rank in manifest.get("expected_ranks", [])
    }
    required_pairs = {
        (layer, rank)
        for layer in (expected_layers or manifest_layers or {capture["layer"] for capture in captures})
        for rank in (expected_ranks or manifest_ranks or {capture["rank"] for capture in captures})
    }
    missing_pairs = sorted(required_pairs - observed_pairs, key=lambda pair: (str(pair[0]), pair[1]))
    report = {
        "schema_version": 2,
        "captures": captures,
        "manifests": manifests,
        "observed_deltas": sorted(observed_deltas),
        "required_deltas": sorted(required_deltas),
        "missing_deltas": sorted(required_deltas - observed_deltas),
        "missing_layer_ranks": [
            {"layer": layer, "rank": rank} for layer, rank in missing_pairs
        ],
        "aggregate": {
            "captures": len(captures),
            "rows": sum(capture["rows"] for capture in captures),
            "ckv_failed_captures": sum(not capture["ckv_bitwise_equal"] for capture in captures),
            "kpe_failed_captures": sum(not capture["kpe_allclose"] for capture in captures),
            "kpe_max_abs_error": float(all_absolute.max(initial=0)),
            "kpe_mean_abs_error": float(all_absolute.mean()) if all_absolute.size else 0.0,
            "kpe_p95_abs_error": float(np.percentile(all_absolute, 95)) if all_absolute.size else 0.0,
            "kpe_p99_abs_error": float(np.percentile(all_absolute, 99)) if all_absolute.size else 0.0,
        },
        "atol": atol,
        "rtol": rtol,
    }
    report["passed"] = bool(
        captures
        and not report["missing_deltas"]
        and not report["missing_layer_ranks"]
        and not report["aggregate"]["ckv_failed_captures"]
        and not report["aggregate"]["kpe_failed_captures"]
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("captures", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--required-deltas", default="0,1,127,128,129,1024")
    parser.add_argument("--expected-layers", default="")
    parser.add_argument("--expected-ranks", default="0,1,2,3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_captures(
        args.captures,
        atol=args.atol,
        rtol=args.rtol,
        required_deltas={int(value) for value in args.required_deltas.split(",") if value},
        expected_layers={value for value in args.expected_layers.split(",") if value} or None,
        expected_ranks={int(value) for value in args.expected_ranks.split(",") if value},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
