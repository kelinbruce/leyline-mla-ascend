# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import vllm_ascend.envs as envs
from benchmarks.leyline.compare_cache import compare_captures
from benchmarks.leyline.compare_logits import compare_logit_vectors
from vllm_ascend.distributed.kv_transfer.leyline.capture import (
    bounded_indices,
    finish_capture,
    prepare_capture,
)
from vllm_ascend.distributed.kv_transfer.leyline.reference import (
    rotate_kpe_half_split,
    unit_delta_cos_sin,
)
from vllm_ascend.worker.leyline_logits_capture import capture_raw_first_token_logits


def _cache() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(1, 8, 1, 512), torch.randn(1, 8, 1, 64)


def test_capture_disabled(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(envs, "VLLM_ASCEND_LEYLINE_CAPTURE_DIR", None)
    ckv, kpe = _cache()
    assert prepare_capture(
        request_id="r",
        session_id="s",
        layer_name="layer",
        ckv_cache=ckv,
        kpe_cache=kpe,
        source_slots=torch.tensor([0]),
        destination_slots=torch.tensor([1]),
        old_positions=torch.tensor([1]),
        new_positions=torch.tensor([0]),
        inv_freq=torch.ones(32),
        native_inv_freq=torch.ones(32),
        block_size=128,
        expected_layers=["layer"],
    ) is None
    assert not list(tmp_path.iterdir())


def test_bounded_selection_prioritizes_required_deltas() -> None:
    old = torch.tensor([0, 2, 129, 131, 1028])
    new = torch.tensor([0, 1, 2, 3, 4])
    selected = bounded_indices(old, new, 3, (0, 127, 1024))
    assert (old - new).index_select(0, selected).tolist() == [0, 127, 1024]


def test_bounded_selection_includes_both_sides_of_block_transition() -> None:
    old = torch.arange(120, 137)
    new = torch.arange(17)
    selected = bounded_indices(old, new, 6, block_size=128)
    selected_old = old.index_select(0, selected).tolist()
    assert 127 in selected_old
    assert 128 in selected_old


def test_capture_clones_aliasing_source_and_writes_manifest(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(envs, "VLLM_ASCEND_LEYLINE_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setattr(envs, "VLLM_ASCEND_LEYLINE_CAPTURE_MAX_ROWS", 8)
    monkeypatch.setattr(envs, "VLLM_ASCEND_LEYLINE_CAPTURE_REQUIRED_DELTAS", ())
    ckv, kpe = _cache()
    source_before = ckv.reshape(-1, 1, 512)[[0, 1]].clone()
    pending = prepare_capture(
        request_id="r",
        session_id="s",
        layer_name="layer.0",
        ckv_cache=ckv,
        kpe_cache=kpe,
        source_slots=torch.tensor([0, 1]),
        destination_slots=torch.tensor([1, 2]),
        old_positions=torch.tensor([1, 2]),
        new_positions=torch.tensor([0, 1]),
        inv_freq=torch.ones(32),
        native_inv_freq=torch.ones(32),
        block_size=128,
        expected_layers=["layer.0"],
    )
    assert pending is not None
    ckv.reshape(-1, 1, 512)[[1, 2]] = source_before
    finish_capture(pending, ckv, kpe)
    with np.load(pending.path) as capture:
        np.testing.assert_array_equal(capture["source_ckv"], source_before.numpy())
    manifest = json.loads(pending.manifest_path.read_text())
    assert manifest["complete_for_rank"]
    assert manifest["observed_layers"] == ["layer.0"]


def _write_comparison_capture(
    path: Path, *, ckv_mismatch: bool, kpe_error: float, rank: int = 0
) -> None:
    source_ckv = np.arange(8, dtype=np.float32).reshape(2, 4)
    actual_ckv = source_ckv.copy()
    if ckv_mismatch:
        actual_ckv[0, 0] += 1
    source_kpe = np.arange(16, dtype=np.float32).reshape(2, 8) / 10
    old_positions = np.asarray([1, 128], dtype=np.int64)
    new_positions = np.asarray([0, 0], dtype=np.int64)
    inv_freq = np.asarray([1.0, 0.5, 0.25, 0.125], dtype=np.float32)
    cos, sin = unit_delta_cos_sin(old_positions, new_positions, inv_freq)
    actual_kpe = rotate_kpe_half_split(source_kpe, cos, sin) + kpe_error
    np.savez_compressed(
        path,
        source_ckv=source_ckv,
        actual_ckv=actual_ckv,
        source_kpe=source_kpe,
        actual_kpe=actual_kpe,
        old_positions=old_positions,
        new_positions=new_positions,
        inv_freq=inv_freq,
        native_inv_freq=inv_freq,
        metadata_json=np.asarray(json.dumps({"layer": "layer.0", "rank": rank})),
    )


def test_compare_reports_missing_rank_ckv_and_kpe_failures(tmp_path: Path) -> None:
    capture = tmp_path / "capture.npz"
    _write_comparison_capture(capture, ckv_mismatch=True, kpe_error=1.0)
    report = compare_captures(
        [capture],
        atol=1e-4,
        rtol=1e-4,
        required_deltas={1, 128},
        expected_layers={"layer.0"},
        expected_ranks={0, 1},
    )
    assert not report["passed"]
    assert report["aggregate"]["ckv_failed_captures"] == 1
    assert report["aggregate"]["kpe_failed_captures"] == 1
    assert report["missing_layer_ranks"] == [{"layer": "layer.0", "rank": 1}]


def test_compare_merges_rank_manifests(tmp_path: Path) -> None:
    manifests = []
    for rank in (0, 1):
        capture = tmp_path / f"rank{rank}.npz"
        _write_comparison_capture(capture, ckv_mismatch=False, kpe_error=0.0, rank=rank)
        manifest = tmp_path / f"rank{rank}.manifest.json"
        manifest.write_text(json.dumps({"schema_version": 2, "captures": [{"path": capture.name}]}))
        manifests.append(manifest)
    report = compare_captures(
        manifests,
        atol=1e-4,
        rtol=1e-4,
        required_deltas={1, 128},
        expected_layers={"layer.0"},
        expected_ranks={0, 1},
    )
    assert report["passed"]
    assert report["aggregate"]["captures"] == 2


def test_compare_rejects_connector_native_frequency_mismatch(tmp_path: Path) -> None:
    capture = tmp_path / "capture.npz"
    _write_comparison_capture(capture, ckv_mismatch=False, kpe_error=0.0)
    with np.load(capture) as original:
        arrays = {key: original[key] for key in original.files}
    arrays["native_inv_freq"] = arrays["inv_freq"] + 0.1
    np.savez_compressed(capture, **arrays)
    report = compare_captures(
        [capture],
        atol=1e-4,
        rtol=1e-4,
        required_deltas={1, 128},
        expected_layers={"layer.0"},
        expected_ranks={0},
    )
    assert not report["passed"]
    assert report["aggregate"]["frequency_failed_captures"] == 1


def test_compare_rejects_missing_layer_and_required_delta(tmp_path: Path) -> None:
    capture = tmp_path / "capture.npz"
    _write_comparison_capture(capture, ckv_mismatch=False, kpe_error=0.0)
    report = compare_captures(
        [capture],
        atol=1e-4,
        rtol=1e-4,
        required_deltas={1, 128, 129},
        expected_layers={"layer.0", "layer.1"},
        expected_ranks={0},
    )
    assert not report["passed"]
    assert report["missing_deltas"] == [129]
    assert report["missing_layer_ranks"] == [{"layer": "layer.1", "rank": 0}]


def test_compare_rejects_rotation_layout_error(tmp_path: Path) -> None:
    capture = tmp_path / "capture.npz"
    _write_comparison_capture(capture, ckv_mismatch=False, kpe_error=0.0)
    with np.load(capture) as original:
        arrays = {key: original[key] for key in original.files}
    arrays["actual_kpe"] = arrays["actual_kpe"][..., ::-1]
    np.savez_compressed(capture, **arrays)
    report = compare_captures(
        [capture],
        atol=1e-4,
        rtol=1e-4,
        required_deltas={1, 128},
        expected_layers={"layer.0"},
        expected_ranks={0},
    )
    assert not report["passed"]
    assert report["aggregate"]["kpe_failed_captures"] == 1


def test_compare_rejects_rotation_sign_error(tmp_path: Path) -> None:
    capture = tmp_path / "capture.npz"
    _write_comparison_capture(capture, ckv_mismatch=False, kpe_error=0.0)
    with np.load(capture) as original:
        arrays = {key: original[key] for key in original.files}
    cos, sin = unit_delta_cos_sin(
        arrays["new_positions"], arrays["old_positions"], arrays["inv_freq"]
    )
    arrays["actual_kpe"] = rotate_kpe_half_split(arrays["source_kpe"], cos, sin)
    np.savez_compressed(capture, **arrays)
    report = compare_captures(
        [capture],
        atol=1e-4,
        rtol=1e-4,
        required_deltas={1, 128},
        expected_layers={"layer.0"},
        expected_ranks={0},
    )
    assert not report["passed"]
    assert report["aggregate"]["kpe_failed_captures"] == 1


def test_compare_rejects_slot_mapping_error(tmp_path: Path) -> None:
    capture = tmp_path / "capture.npz"
    _write_comparison_capture(capture, ckv_mismatch=False, kpe_error=0.0)
    with np.load(capture) as original:
        arrays = {key: original[key] for key in original.files}
    arrays["actual_ckv"] = arrays["actual_ckv"][::-1]
    np.savez_compressed(capture, **arrays)
    report = compare_captures(
        [capture],
        atol=1e-4,
        rtol=1e-4,
        required_deltas={1, 128},
        expected_layers={"layer.0"},
        expected_ranks={0},
    )
    assert not report["passed"]
    assert report["aggregate"]["ckv_failed_captures"] == 1


def test_raw_logits_are_rank_scoped_and_provenanced(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(envs, "VLLM_ASCEND_LEYLINE_RAW_LOGITS_DIR", str(tmp_path))
    with patch("vllm_ascend.worker.leyline_logits_capture._rank", return_value=3):
        capture_raw_first_token_logits(torch.tensor([[1.0, 2.0]]), ["request"], [[]])
    path = tmp_path / "request.rank3.first-token-logits.npz"
    with np.load(path) as capture:
        metadata = json.loads(str(capture["metadata_json"]))
        np.testing.assert_array_equal(capture["logits"], np.asarray([1.0, 2.0]))
    assert metadata["evidence_type"] == "internal_raw_logits"
    assert "before_grammar_and_sampler" in metadata["provenance"]


def test_raw_logit_comparison_reports_distribution_metrics() -> None:
    report = compare_logit_vectors(
        np.asarray([0.0, 2.0, 1.0]),
        np.asarray([0.0, 1.5, 1.25]),
        top_k=2,
    )
    assert report["left_selected_token_id"] == 1
    assert report["right_selected_token_id"] == 1
    assert report["topk_overlap_token_ids"] == [1, 2]
    assert report["max_abs_difference"] == 0.5
