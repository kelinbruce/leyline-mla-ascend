# SPDX-License-Identifier: Apache-2.0

"""Opt-in raw first-token logit capture for Leyline diagnostics."""

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
    rows = min(logits.shape[0], len(request_ids))
    for index in range(rows):
        request_id = request_ids[index]
        generated = output_token_ids[index] if index < len(output_token_ids) else None
        if request_id is None or generated:
            continue
        safe_request_id = re.sub(r"[^A-Za-z0-9_.-]", "_", request_id)
        path = root / f"{safe_request_id}.rank{rank}.first-token-logits.npz"
        metadata = {
            "schema_version": 2,
            "request_id": request_id,
            "rank": rank,
            "evidence_type": "internal_raw_logits",
            "provenance": "NPUModelRunner.sample_tokens.before_grammar_and_sampler",
            "dtype": str(logits.dtype),
            "shape": list(logits[index].shape),
            "model": model,
            "sampling_provenance": sampling_provenance or {"stage": "before_sampler"},
        }
        np.savez_compressed(
            path,
            logits=logits[index].detach().float().cpu().numpy(),
            metadata_json=np.asarray(json.dumps(metadata)),
        )
        os.chmod(path, 0o600)
