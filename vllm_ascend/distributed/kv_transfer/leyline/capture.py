# SPDX-License-Identifier: Apache-2.0

"""Opt-in, rank-scoped capture of Leyline MLA cache transformations."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

import vllm_ascend.envs as envs


@dataclass(frozen=True)
class PendingCapture:
    path: Path
    manifest_path: Path
    metadata: dict[str, Any]
    source_ckv: torch.Tensor
    source_kpe: torch.Tensor
    destination_slots: torch.Tensor


def capture_enabled() -> bool:
    return bool(envs.VLLM_ASCEND_LEYLINE_CAPTURE_DIR)


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def _ckv_numpy(tensor: torch.Tensor) -> np.ndarray:
    value = tensor.detach()
    if value.dtype == torch.bfloat16:
        value = value.view(torch.uint16)
    return value.cpu().numpy()


def _kpe_numpy(tensor: torch.Tensor) -> np.ndarray:
    # NumPy has no portable bfloat16 representation. FP32 preserves every
    # BF16 value exactly and is also the comparison reference dtype.
    return tensor.detach().float().cpu().numpy()


def bounded_indices(
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
    max_rows: int,
    required_deltas: tuple[int, ...] = (),
) -> torch.Tensor:
    rows = int(old_positions.numel())
    if max_rows <= 0 or rows <= max_rows:
        return torch.arange(rows, device=old_positions.device, dtype=torch.long)
    selected: list[int] = []
    deltas = (old_positions - new_positions).detach().cpu().tolist()
    for required in required_deltas:
        if required in deltas:
            selected.append(deltas.index(required))
    evenly_spaced = torch.linspace(0, rows - 1, steps=max_rows).round().to(torch.long).tolist()
    selected.extend(evenly_spaced)
    selected = list(dict.fromkeys(selected))[:max_rows]
    return torch.tensor(selected, device=old_positions.device, dtype=torch.long)


def prepare_capture(
    *,
    request_id: str,
    session_id: str,
    layer_name: str,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    source_slots: Sequence[int] | torch.Tensor,
    destination_slots: Sequence[int] | torch.Tensor,
    old_positions: Sequence[int] | torch.Tensor,
    new_positions: Sequence[int] | torch.Tensor,
    inv_freq: torch.Tensor,
    block_size: int,
    expected_layers: list[str],
) -> PendingCapture | None:
    capture_dir = envs.VLLM_ASCEND_LEYLINE_CAPTURE_DIR
    if not capture_dir:
        return None
    root = Path(capture_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    rank = _rank()
    device = ckv_cache.device
    source_slots_tensor = torch.as_tensor(source_slots, device=device, dtype=torch.long)
    destination_slots_tensor = torch.as_tensor(
        destination_slots, device=device, dtype=torch.long
    )
    old_positions_tensor = torch.as_tensor(old_positions, device=device, dtype=torch.long)
    new_positions_tensor = torch.as_tensor(new_positions, device=device, dtype=torch.long)
    selected = bounded_indices(
        old_positions_tensor,
        new_positions_tensor,
        envs.VLLM_ASCEND_LEYLINE_CAPTURE_MAX_ROWS,
        envs.VLLM_ASCEND_LEYLINE_CAPTURE_REQUIRED_DELTAS,
    )
    selected_source = source_slots_tensor.index_select(0, selected)
    selected_destination = destination_slots_tensor.index_select(0, selected)
    flat_ckv = ckv_cache.reshape(-1, *ckv_cache.shape[2:])
    flat_kpe = kpe_cache.reshape(-1, *kpe_cache.shape[2:])
    filename = f"{_safe(session_id)}.{_safe(request_id)}.rank{rank}.{_safe(layer_name)}.npz"
    metadata = {
        "schema_version": 2,
        "request_id": request_id,
        "session_id": session_id,
        "layer": layer_name,
        "rank": rank,
        "block_size": block_size,
        "dtype": str(ckv_cache.dtype),
        "synchronization": "torch.npu.synchronize_before_destination_read",
        "expected_layers": expected_layers,
        "expected_ranks": list(range(torch.distributed.get_world_size()))
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else [0],
        "source_slots": selected_source.detach().cpu().numpy(),
        "destination_slots": selected_destination.detach().cpu().numpy(),
        "old_positions": old_positions_tensor.index_select(0, selected).detach().cpu().numpy(),
        "new_positions": new_positions_tensor.index_select(0, selected).detach().cpu().numpy(),
        "inv_freq": inv_freq.detach().cpu().numpy(),
    }
    return PendingCapture(
        path=root / filename,
        manifest_path=root / f"{_safe(session_id)}.rank{rank}.manifest.json",
        metadata=metadata,
        source_ckv=flat_ckv.index_select(0, selected_source).clone(),
        source_kpe=flat_kpe.index_select(0, selected_source).clone(),
        destination_slots=selected_destination,
    )


def finish_capture(
    pending: PendingCapture | None,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
) -> None:
    if pending is None:
        return
    flat_ckv = ckv_cache.reshape(-1, *ckv_cache.shape[2:])
    flat_kpe = kpe_cache.reshape(-1, *kpe_cache.shape[2:])
    actual_ckv = _ckv_numpy(flat_ckv.index_select(0, pending.destination_slots))
    actual_kpe = _kpe_numpy(flat_kpe.index_select(0, pending.destination_slots))
    arrays = {
        key: value for key, value in pending.metadata.items() if isinstance(value, np.ndarray)
    }
    scalar_metadata = {
        key: value for key, value in pending.metadata.items() if not isinstance(value, np.ndarray)
    }
    np.savez_compressed(
        pending.path,
        **arrays,
        source_ckv=_ckv_numpy(pending.source_ckv),
        actual_ckv=actual_ckv,
        source_kpe=_kpe_numpy(pending.source_kpe),
        actual_kpe=actual_kpe,
        metadata_json=np.asarray(json.dumps(scalar_metadata)),
    )
    os.chmod(pending.path, 0o600)
    manifest = {"schema_version": 2, "captures": []}
    if pending.manifest_path.exists():
        manifest = json.loads(pending.manifest_path.read_text())
    manifest["captures"] = [
        item for item in manifest.get("captures", []) if item.get("path") != pending.path.name
    ]
    manifest["captures"].append({**scalar_metadata, "path": pending.path.name})
    expected_layers = set(scalar_metadata["expected_layers"])
    observed_layers = {item["layer"] for item in manifest["captures"]}
    manifest["rank"] = scalar_metadata["rank"]
    manifest["expected_layers"] = sorted(expected_layers)
    manifest["observed_layers"] = sorted(observed_layers)
    manifest["expected_ranks"] = scalar_metadata["expected_ranks"]
    manifest["complete_for_rank"] = expected_layers <= observed_layers
    temporary = pending.manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.chmod(temporary, 0o600)
    temporary.replace(pending.manifest_path)
