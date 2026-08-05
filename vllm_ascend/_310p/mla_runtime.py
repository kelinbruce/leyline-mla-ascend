# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Fail-closed runtime checks for the experimental 310P MLA path."""

from __future__ import annotations

from typing import Any

import torch

LEYLINE_CONNECTOR_NAME = "LeylineConnector"
LEYLINE_CONNECTOR_MODULE = "vllm_ascend.distributed.kv_transfer.leyline.connector"
LEYLINE_KV_ROLE = "kv_both"
MLA_BLOCK_SIZE_310P = 128


def is_local_leyline_connector(kv_transfer_config: Any | None) -> bool:
    """Return whether config identifies the supported in-process connector."""

    if kv_transfer_config is None:
        return False
    return bool(
        getattr(kv_transfer_config, "kv_connector", None) == LEYLINE_CONNECTOR_NAME
        and getattr(kv_transfer_config, "kv_connector_module_path", None)
        == LEYLINE_CONNECTOR_MODULE
        and getattr(kv_transfer_config, "kv_role", None) == LEYLINE_KV_ROLE
        and getattr(kv_transfer_config, "kv_load_failure_policy", None) == "recompute"
    )


def get_310p_mla_runtime_errors(vllm_config: Any) -> list[str]:
    """Return deterministic reasons why a 310P MLA config is unsupported.

    This validates source-level/runtime invariants only. Passing it does not
    replace the baseline-matched hardware qualification required before the
    experimental backend is advertised as supported.
    """

    model = vllm_config.model_config
    cache = vllm_config.cache_config
    parallel = vllm_config.parallel_config
    errors: list[str] = []

    if getattr(getattr(model, "hf_text_config", None), "model_type", None) != "deepseek_v2":
        errors.append("model type must be deepseek_v2")
    if getattr(model, "dtype", None) != torch.float16:
        errors.append("model dtype must be torch.float16")
    if str(getattr(cache, "cache_dtype", None)) not in {
        "auto",
        "float16",
        "torch.float16",
    }:
        errors.append("KV cache dtype must be auto or float16")
    if getattr(cache, "block_size", None) != MLA_BLOCK_SIZE_310P:
        errors.append(f"KV cache block size must be {MLA_BLOCK_SIZE_310P}")
    if not getattr(cache, "enable_prefix_caching", False):
        errors.append("automatic prefix caching is required")
    if not getattr(model, "enforce_eager", False):
        errors.append("eager execution is required")
    if getattr(model, "quantization", None) is not None:
        errors.append("quantization is not supported")
    if getattr(vllm_config, "speculative_config", None) is not None:
        errors.append("speculative decoding is not supported")
    if getattr(parallel, "tensor_parallel_size", None) != 4:
        errors.append("tensor parallel size must be 4")
    if getattr(parallel, "decode_context_parallel_size", None) != 1:
        errors.append("decode context parallel size must be 1")
    if getattr(parallel, "prefill_context_parallel_size", None) != 1:
        errors.append("prefill context parallel size must be 1")
    if getattr(parallel, "pipeline_parallel_size", None) != 1:
        errors.append("pipeline parallel size must be 1")
    if not getattr(vllm_config.scheduler_config, "enable_chunked_prefill", False):
        errors.append("chunked prefill is required")

    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer_config is not None and not is_local_leyline_connector(kv_transfer_config):
        errors.append("only the local Leyline connector with recompute fallback is supported")
    return errors


def require_310p_mla_runtime(vllm_config: Any) -> None:
    errors = get_310p_mla_runtime_errors(vllm_config)
    if errors:
        raise ValueError("Unsupported 310P MLA runtime: " + "; ".join(errors))
