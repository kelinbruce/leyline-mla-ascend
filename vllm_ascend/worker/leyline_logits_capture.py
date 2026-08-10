# SPDX-License-Identifier: Apache-2.0

"""Opt-in, bounded raw decode-step logit capture for Leyline diagnostics."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

import vllm_ascend.envs as envs


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _validation_provenance(request_id: str) -> dict[str, Any]:
    request_id = request_id.removeprefix("cmpl-")
    parts = request_id.split(".")
    if len(parts) != 7 or parts[0] != "lv3":
        return {}
    try:
        repetition = int(parts[4])
    except ValueError:
        return {}
    return {
        "run_id": parts[1],
        "case_id": parts[2],
        "variant": parts[3],
        "repetition": repetition,
        "arm": parts[5],
    }


def capture_raw_first_token_logits(
    logits: torch.Tensor | None,
    request_ids: list[str | None],
    output_token_ids: list[list[int] | None],
    *,
    model: str | None = None,
    sampling_provenance: dict[str, Any] | None = None,
) -> None:
    root_value = envs.VLLM_ASCEND_LEYLINE_RAW_LOGITS_DIR
    if not root_value or logits is None:
        return
    root = Path(root_value).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    rank = _rank()
    steps = tuple(getattr(envs, "VLLM_ASCEND_LEYLINE_RAW_LOGITS_STEPS", (0,)))
    maximum_step = int(getattr(envs, "VLLM_ASCEND_LEYLINE_RAW_LOGITS_MAX_STEP", 32))
    if maximum_step < 0 or maximum_step > 256:
        raise ValueError("Leyline raw-logit maximum step must be between 0 and 256")
    if not steps or any(step < 0 or step > maximum_step for step in steps):
        raise ValueError("Leyline raw-logit steps must be finite and within the maximum step")
    run_filter = getattr(envs, "VLLM_ASCEND_LEYLINE_RAW_LOGITS_RUN_ID", None)
    case_filters = tuple(getattr(envs, "VLLM_ASCEND_LEYLINE_RAW_LOGITS_CASES", ()))
    arm_filters = tuple(getattr(envs, "VLLM_ASCEND_LEYLINE_RAW_LOGITS_ARMS", ()))
    if any(step > 0 for step in steps) and not run_filter:
        raise ValueError("multi-step raw-logit capture requires a validation run-id filter")
    maximum_files = int(getattr(envs, "VLLM_ASCEND_LEYLINE_RAW_LOGITS_MAX_FILES", 4096))
    maximum_bytes = int(getattr(envs, "VLLM_ASCEND_LEYLINE_RAW_LOGITS_MAX_BYTES", 8 * 1024**3))
    if maximum_files < 1 or maximum_bytes < 1:
        raise ValueError("Leyline raw-logit capture budgets must be positive")
    rows = min(logits.shape[0], len(request_ids))
    for index in range(rows):
        request_id = request_ids[index]
        generated = output_token_ids[index] if index < len(output_token_ids) else None
        if request_id is None:
            continue
        decode_step = len(generated or [])
        if decode_step not in steps:
            continue
        if run_filter and f".{run_filter}." not in request_id:
            continue
        if case_filters and not any(f".{case}." in request_id for case in case_filters):
            continue
        if arm_filters and not any(f".{arm}." in request_id for arm in arm_filters):
            continue
        safe_request_id = re.sub(r"[^A-Za-z0-9_.-]", "_", request_id)
        suffix = "first-token-logits" if decode_step == 0 else f"step{decode_step:04d}-logits"
        path = root / f"{safe_request_id}.rank{rank}.{suffix}.npz"
        existing = list(root.glob("*.npz"))
        existing_bytes = sum(item.stat().st_size for item in existing)
        estimated_bytes = int(logits[index].numel()) * 4
        if len(existing) >= maximum_files or existing_bytes + estimated_bytes > maximum_bytes:
            status = {
                "schema_version": 1,
                "complete": False,
                "reason": "capture_budget_exhausted",
                "rank": rank,
                "files": len(existing),
                "bytes": existing_bytes,
                "max_files": maximum_files,
                "max_bytes": maximum_bytes,
            }
            status_path = root / f"capture-status.rank{rank}.json"
            status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
            os.chmod(status_path, 0o600)
            continue
        metadata = {
            "schema_version": 2,
            "request_id": request_id,
            "rank": rank,
            "evidence_type": "internal_raw_logits",
            "provenance": "NPUModelRunner.sample_tokens.before_grammar_and_sampler",
            "decode_step": decode_step,
            "run_id": run_filter,
            "case_filter": next(
                (case for case in case_filters if f".{case}." in request_id), None
            ),
            "arm_filter": next(
                (arm for arm in arm_filters if f".{arm}." in request_id), None
            ),
            "dtype": str(logits.dtype),
            "shape": list(logits[index].shape),
            "model": model,
            "sampling_provenance": sampling_provenance or {"stage": "before_sampler"},
            **_validation_provenance(request_id),
        }
        np.savez_compressed(
            path,
            logits=logits[index].detach().float().cpu().numpy(),
            metadata_json=np.asarray(json.dumps(metadata)),
        )
        os.chmod(path, 0o600)
