# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import vllm_ascend.envs as envs
from benchmarks.leyline.compare_cache import compare_captures
from benchmarks.leyline.compare_logits import compare_logit_vectors, compare_report
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
    assert metadata["decode_step"] == 0
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


def _configure_step_capture(monkeypatch, tmp_path: Path, *, steps=(1,), cases=("case",)) -> None:
    monkeypatch.setattr(envs, "VLLM_ASCEND_LEYLINE_RAW_LOGITS_DIR", str(tmp_path))
    monkeypatch.setattr(envs, "VLLM_ASCEND_LEYLINE_RAW_LOGITS_STEPS", steps)
    monkeypatch.setattr(envs, "VLLM_ASCEND_LEYLINE_RAW_LOGITS_MAX_STEP", 4)
    monkeypatch.setattr(envs, "VLLM_ASCEND_LEYLINE_RAW_LOGITS_RUN_ID", "run")
    monkeypatch.setattr(envs, "VLLM_ASCEND_LEYLINE_RAW_LOGITS_CASES", cases)
    monkeypatch.setattr(envs, "VLLM_ASCEND_LEYLINE_RAW_LOGITS_MAX_FILES", 16)
    monkeypatch.setattr(envs, "VLLM_ASCEND_LEYLINE_RAW_LOGITS_MAX_BYTES", 1024 * 1024)


def test_raw_logits_capture_selected_decode_step(monkeypatch, tmp_path: Path) -> None:
    _configure_step_capture(monkeypatch, tmp_path)
    request_id = "lv3.run.case.canonical.0.full.nonce"
    with patch("vllm_ascend.worker.leyline_logits_capture._rank", return_value=1):
        capture_raw_first_token_logits(
            torch.tensor([[1.0, 3.0]]), [request_id], [[7]]
        )
    path = tmp_path / f"{request_id}.rank1.step0001-logits.npz"
    with np.load(path) as capture:
        metadata = json.loads(str(capture["metadata_json"]))
    assert metadata["decode_step"] == 1
    assert metadata["run_id"] == "run"


def test_raw_logits_request_filter_and_budget(monkeypatch, tmp_path: Path) -> None:
    _configure_step_capture(monkeypatch, tmp_path, steps=(0,))
    monkeypatch.setattr(envs, "VLLM_ASCEND_LEYLINE_RAW_LOGITS_MAX_FILES", 1)
    with patch("vllm_ascend.worker.leyline_logits_capture._rank", return_value=0):
        capture_raw_first_token_logits(
            torch.tensor([[1.0, 2.0]]),
            ["lv3.run.other.canonical.0.full.nonce"],
            [[]],
        )
        assert not list(tmp_path.glob("*.npz"))
        capture_raw_first_token_logits(
            torch.tensor([[1.0, 2.0]]),
            ["lv3.run.case.canonical.0.full.one"],
            [[]],
        )
        capture_raw_first_token_logits(
            torch.tensor([[1.0, 2.0]]),
            ["lv3.run.case.canonical.0.full.two"],
            [[]],
        )
    status = json.loads((tmp_path / "capture-status.rank0.json").read_text())
    assert status["reason"] == "capture_budget_exhausted"


def _write_logit(path: Path, request_id: str, step: int, values: list[float]) -> None:
    np.savez_compressed(
        path,
        logits=np.asarray(values, dtype=np.float32),
        metadata_json=np.asarray(
            json.dumps({"request_id": request_id, "rank": 0, "decode_step": step})
        ),
    )


def test_divergence_logit_comparison_requires_reproduced_prefix(tmp_path: Path) -> None:
    arms = {
        name: {"request_id": name, "output_token_ids": tokens}
        for name, tokens in {
            "full": [1, 3],
            "honest_edited": [1, 2],
            "leyline": [1, 4],
        }.items()
    }
    run = {
        "arms": arms,
        "pairwise": {
            "full_leyline": {
                "common_prefix_tokens": 1,
                "first_divergence": {"index": 1, "left": 3, "right": 4},
            }
        },
    }
    report = {"cases": [{"id": "case", "repetitions": [run]}]}
    plan = {
        "entries": [
            {
                "case_id": "case",
                "repetition": 0,
                "decode_step": 1,
                "source_prefix_token_ids": [1],
                "source_request_ids": {"full": "source-full", "leyline": "source-l"},
            }
        ]
    }
    for request_id in arms:
        _write_logit(
            tmp_path / f"{request_id}.rank0.step0001-logits.npz",
            request_id,
            1,
            [0.0, 1.0, 2.0],
        )
    comparison = compare_report(report, tmp_path, divergence_plan=plan)
    assert comparison["complete"]
    assert all(item["decode_step"] == 1 for item in comparison["comparisons"])
    plan["entries"][0]["source_prefix_token_ids"] = [9]
    comparison = compare_report(report, tmp_path, divergence_plan=plan)
    assert not comparison["complete"]
    assert not comparison["correlation"][0]["correlatable"]


def test_logit_comparison_requires_every_tensor_parallel_rank(tmp_path: Path) -> None:
    arms = {
        name: {"request_id": name, "output_token_ids": [1]}
        for name in ("full", "honest_edited", "leyline")
    }
    report = {
        "environment": {"runtime_config": {"tensor_parallel_size": 2}},
        "cases": [{"id": "case", "arms": arms}],
    }
    for request_id in arms:
        _write_logit(
            tmp_path / f"{request_id}.rank0.first-token-logits.npz",
            request_id,
            0,
            [0.0, 1.0, 2.0],
        )
    comparison = compare_report(report, tmp_path)
    assert not comparison["complete"]
    assert comparison["missing_rank_pairs"]
    assert all(item["ranks"] == [1] for item in comparison["missing_rank_pairs"])


def test_logit_comparison_labels_legacy_first_token_capture(tmp_path: Path) -> None:
    arms = {
        name: {"request_id": name, "output_token_ids": [1]}
        for name in ("full", "honest_edited", "leyline")
    }
    for request_id in arms:
        np.savez_compressed(
            tmp_path / f"{request_id}.rank0.first-token-logits.npz",
            logits=np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
            metadata_json=np.asarray(
                json.dumps({"request_id": request_id, "rank": 0})
            ),
        )
    comparison = compare_report({"cases": [{"id": "case", "arms": arms}]}, tmp_path)
    assert comparison["complete"]
    assert all(
        item["left_provenance"]["legacy_first_token"]
        and item["right_provenance"]["legacy_first_token"]
        for item in comparison["comparisons"]
    )
