# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.distributed.kv_transfer.leyline.connector import (
    LeylineConnector,
    LeylineTransformRequest,
    LeylineWorkerMetadata,
    LeylineWorkerResult,
)
from vllm_ascend.distributed.kv_transfer.leyline.protocol import LeylineFallbackReason


class _Block:
    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.ref_cnt = 1
        self.block_hash = ("hash", block_id)


class _BlockPool:
    def __init__(self, count: int = 32) -> None:
        self.blocks = [_Block(block_id) for block_id in range(count)]

    def touch(self, blocks: list[_Block]) -> None:
        for block in blocks:
            block.ref_cnt += 1

    def free_blocks(self, blocks: list[_Block]) -> None:
        for block in blocks:
            assert block.ref_cnt > 0
            block.ref_cnt -= 1


class _CacheBlocks:
    def __init__(self, block_ids: list[int]) -> None:
        self._block_ids = block_ids

    def get_block_ids(self) -> tuple[list[int], ...]:
        return (self._block_ids,)


def _connector() -> tuple[LeylineConnector, _BlockPool]:
    connector = LeylineConnector.__new__(LeylineConnector)
    connector._block_size = 4
    connector._engine_identity = ("test-engine",)
    connector._runtime_supported = True
    connector._inv_freq = tuple(float(index + 1) for index in range(32))
    connector._block_pool = pool = _BlockPool()
    connector._sessions = {}
    connector._directives = {}
    connector._matches = {}
    connector._metadata_pending = {}
    connector._inflight = {}
    connector._outcomes = {}
    connector._kv_caches = {}
    connector._finished_recving = set()
    connector._invalid_block_ids = set()
    connector._worker_results = {}
    return connector, pool


def _runtime_config(dtype: torch.dtype, cache_dtype: str) -> tuple[SimpleNamespace, SimpleNamespace]:
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=dtype,
            enforce_eager=True,
            quantization=None,
            hf_text_config=SimpleNamespace(
                model_type="deepseek_v2",
                rope_scaling={"type": "yarn"},
            ),
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        cache_config=SimpleNamespace(
            cache_dtype=cache_dtype,
            block_size=128,
            enable_prefix_caching=True,
        ),
        speculative_config=None,
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
    )
    return config, SimpleNamespace(kv_cache_groups=[object()])


def test_runtime_matrix_uses_fp16_only_on_310p() -> None:
    fp16_config, kv_cache_config = _runtime_config(torch.float16, "auto")
    bf16_config, _ = _runtime_config(torch.bfloat16, "auto")

    with patch("vllm_ascend.distributed.kv_transfer.leyline.connector.is_310p", return_value=True):
        assert LeylineConnector._is_supported_runtime(fp16_config, kv_cache_config)
        assert not LeylineConnector._is_supported_runtime(bf16_config, kv_cache_config)

    with patch("vllm_ascend.distributed.kv_transfer.leyline.connector.is_310p", return_value=False):
        assert LeylineConnector._is_supported_runtime(bf16_config, kv_cache_config)
        assert not LeylineConnector._is_supported_runtime(fp16_config, kv_cache_config)


def _request(
    request_id: str,
    tokens: list[int],
    action: str,
    *,
    delete: tuple[int, int] | None = None,
) -> SimpleNamespace:
    directive = {
        "version": 1,
        "action": action,
        "session_id": "session-1",
    }
    if delete is not None:
        directive["delete"] = {"start": delete[0], "end": delete[1]}
    return SimpleNamespace(
        request_id=request_id,
        kv_transfer_params={"leyline": directive},
        prompt_token_ids=tokens,
        prompt_embeds=None,
        mm_features=[],
        num_tokens=len(tokens),
        num_prompt_tokens=len(tokens),
        num_computed_tokens=len(tokens),
        all_token_ids=tokens,
        cache_salt="tenant-a",
        lora_request=None,
        status=SimpleNamespace(name="FINISHED_STOPPED"),
    )


def _record_source(connector: LeylineConnector, pool: _BlockPool) -> list[int]:
    # One partial tail token allows all three edited full blocks to be loaded;
    # vLLM keeps the final prompt token for the sampling forward pass.
    source_tokens = list(range(17))
    request = _request("record", source_tokens, "record")
    connector.on_new_request(request)
    delay, params = connector.request_finished(request, [0, 1, 2, 3])
    assert not delay
    assert params == {
        "leyline": {
            "applied": False,
            "transformed_tokens": 0,
            "local_apc_tokens": 0,
            "normal_prefill_tokens": 0,
            "transform_duration_ms": 0.0,
            "fallback_reason": None,
            "recorded": True,
        }
    }
    # Model the scheduler dropping the finished request's original reference.
    pool.free_blocks(pool.blocks[:4])
    assert [block.ref_cnt for block in pool.blocks[:4]] == [1, 1, 1, 1]
    return source_tokens


def _prepare_amortize(
    connector: LeylineConnector,
    source_tokens: list[int],
) -> SimpleNamespace:
    edited = source_tokens[:4] + source_tokens[8:]
    request = _request("amortize", edited, "amortize", delete=(4, 8))
    connector.on_new_request(request)
    external, asynchronous = connector.get_num_new_matched_tokens(request, 4)
    assert (external, asynchronous) == (8, True)
    connector.update_state_after_alloc(request, _CacheBlocks([10, 11, 12]), external)
    return request


def test_scheduler_success_pins_until_tp_wide_completion() -> None:
    connector, pool = _connector()
    source_tokens = _record_source(connector, pool)
    request = _prepare_amortize(connector, source_tokens)
    assert [block.ref_cnt for block in pool.blocks[:4]] == [2, 2, 2, 2]

    metadata = connector.build_connector_meta(None)
    assert len(metadata.requests) == 1
    assert metadata.requests[0].destination_block_ids == (10, 11, 12)

    result = LeylineWorkerResult(True, 1.25, 8, 27)
    connector.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=LeylineWorkerMetadata({request.request_id: result}),
            invalid_block_ids=set(),
            finished_recving={request.request_id},
        )
    )
    assert connector._outcomes[request.request_id].applied
    assert connector._outcomes[request.request_id].duration_ms == 1.25
    assert "session-1" not in connector._sessions
    assert [block.ref_cnt for block in pool.blocks[:4]] == [0, 0, 0, 0]


def test_missing_source_block_falls_back_without_external_tokens() -> None:
    connector, pool = _connector()
    source_tokens = _record_source(connector, pool)
    pool.blocks[2].ref_cnt = 0
    request = _request(
        "missing",
        source_tokens[:4] + source_tokens[8:],
        "amortize",
        delete=(4, 8),
    )
    connector.on_new_request(request)
    assert connector.get_num_new_matched_tokens(request, 4) == (0, False)
    assert connector._outcomes[request.request_id].fallback_reason is (
        LeylineFallbackReason.SOURCE_BLOCK_MISSING
    )
    assert connector._outcomes[request.request_id].normal_prefill_tokens == 9


def test_unsupported_runtime_uses_honest_prefill() -> None:
    connector, pool = _connector()
    source_tokens = _record_source(connector, pool)
    connector._runtime_supported = False
    request = _request(
        "unsupported",
        source_tokens[:4] + source_tokens[8:],
        "amortize",
        delete=(4, 8),
    )
    connector.on_new_request(request)
    assert connector.get_num_new_matched_tokens(request, 4) == (0, False)
    assert connector._outcomes[request.request_id].fallback_reason is (
        LeylineFallbackReason.UNSUPPORTED_RUNTIME
    )
    assert connector._outcomes[request.request_id].normal_prefill_tokens == 9


def test_tp_aggregation_requires_every_rank_to_succeed() -> None:
    good = LeylineWorkerMetadata({"r": LeylineWorkerResult(True, 1.0, 8, 27)})
    bad = LeylineWorkerMetadata({"r": LeylineWorkerResult(False, 2.0, 8, 11)})
    result = good.aggregate(bad).results["r"]
    assert not result.success
    assert result.duration_ms == 2.0
    assert result.transformed_tokens == 8
    assert result.transformed_layers == 11


def test_failed_transform_rolls_back_and_counts_reprefill() -> None:
    connector, pool = _connector()
    source_tokens = _record_source(connector, pool)
    request = _prepare_amortize(connector, source_tokens)
    plan: LeylineTransformRequest = connector._inflight[request.request_id]
    connector.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=LeylineWorkerMetadata(
                {request.request_id: LeylineWorkerResult(False, 3.0, 8, 0)}
            ),
            invalid_block_ids=set(plan.transformed_destination_blocks),
            finished_recving={request.request_id},
        )
    )
    outcome = connector._outcomes[request.request_id]
    assert not outcome.applied
    assert outcome.fallback_reason is LeylineFallbackReason.TRANSFORM_FAILED
    assert outcome.local_apc_tokens == 4
    assert outcome.transformed_tokens == 8
    assert outcome.normal_prefill_tokens == 9
    assert [block.ref_cnt for block in pool.blocks[:4]] == [0, 0, 0, 0]


def test_invalid_directive_reports_reason_without_leaking_state() -> None:
    connector, _ = _connector()
    request = _request("invalid", list(range(17)), "record")
    request.kv_transfer_params = {"leyline": {"version": 999}}
    connector.on_new_request(request)
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    delay, params = connector.request_finished(request, [])
    assert not delay
    assert params is not None
    assert params["leyline"]["fallback_reason"] == "invalid_directive"
    assert not params["leyline"]["recorded"]
    assert request.request_id not in connector._outcomes


def test_shutdown_releases_session_and_inflight_references() -> None:
    connector, pool = _connector()
    source_tokens = _record_source(connector, pool)
    _prepare_amortize(connector, source_tokens)
    assert [block.ref_cnt for block in pool.blocks[:4]] == [2, 2, 2, 2]
    connector.shutdown()
    assert [block.ref_cnt for block in pool.blocks[:4]] == [0, 0, 0, 0]
    assert not connector._sessions
    assert not connector._inflight
