# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from vllm_ascend._310p.mla_runtime import (
    LEYLINE_CONNECTOR_MODULE,
    get_310p_mla_runtime_errors,
    is_local_leyline_connector,
)


def _kv_transfer(**overrides):
    values = {
        "kv_connector": "LeylineConnector",
        "kv_connector_module_path": LEYLINE_CONNECTOR_MODULE,
        "kv_role": "kv_both",
        "kv_load_failure_policy": "recompute",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _vllm_config(**overrides):
    values = {
        "model_config": SimpleNamespace(
            dtype=torch.float16,
            enforce_eager=True,
            quantization=None,
            hf_text_config=SimpleNamespace(model_type="deepseek_v2"),
        ),
        "cache_config": SimpleNamespace(
            cache_dtype="auto",
            block_size=128,
            enable_prefix_caching=True,
        ),
        "parallel_config": SimpleNamespace(
            tensor_parallel_size=4,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        "scheduler_config": SimpleNamespace(enable_chunked_prefill=True),
        "speculative_config": None,
        "kv_transfer_config": _kv_transfer(),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_local_leyline_connector_requires_exact_safe_identity() -> None:
    assert is_local_leyline_connector(_kv_transfer())
    assert not is_local_leyline_connector(_kv_transfer(kv_role="kv_consumer"))
    assert not is_local_leyline_connector(_kv_transfer(kv_load_failure_policy="fail"))
    assert not is_local_leyline_connector(_kv_transfer(kv_connector_module_path="other.module"))


def test_supported_310p_mla_runtime_has_no_source_level_errors() -> None:
    assert get_310p_mla_runtime_errors(_vllm_config()) == []


def test_unsupported_310p_mla_runtime_reports_every_reason() -> None:
    config = _vllm_config(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            enforce_eager=False,
            quantization="ascend",
            hf_text_config=SimpleNamespace(model_type="qwen3"),
        ),
        cache_config=SimpleNamespace(
            cache_dtype="bfloat16",
            block_size=64,
            enable_prefix_caching=False,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2,
            decode_context_parallel_size=2,
            prefill_context_parallel_size=2,
            pipeline_parallel_size=2,
        ),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=False),
        speculative_config=object(),
        kv_transfer_config=_kv_transfer(kv_connector="OtherConnector"),
    )

    errors = get_310p_mla_runtime_errors(config)

    assert errors == [
        "model type must be deepseek_v2",
        "model dtype must be torch.float16",
        "KV cache dtype must be auto or float16",
        "KV cache block size must be 128",
        "automatic prefix caching is required",
        "eager execution is required",
        "quantization is not supported",
        "speculative decoding is not supported",
        "tensor parallel size must be 4",
        "decode context parallel size must be 1",
        "prefill context parallel size must be 1",
        "pipeline parallel size must be 1",
        "chunked prefill is required",
        "only the local Leyline connector with recompute fallback is supported",
    ]
