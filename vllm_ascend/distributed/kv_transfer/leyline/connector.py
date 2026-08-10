# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Local vLLM KV connector for transactional Leyline MLA transformation."""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
    SupportsHMA,
)
from vllm.logger import init_logger
from vllm.v1.outputs import KVConnectorOutput

import vllm_ascend.envs as envs
from vllm_ascend.distributed.kv_transfer.leyline.capture import (
    capture_enabled,
    finish_capture,
    prepare_capture,
)
from vllm_ascend.distributed.kv_transfer.leyline.mapping import (
    DeletionMapping,
    build_slot_mapping,
    find_reusable_target_end,
    validate_deletion,
)
from vllm_ascend.distributed.kv_transfer.leyline.protocol import (
    DeleteSpan,
    LeylineAction,
    LeylineDirective,
    LeylineDirectiveError,
    LeylineFallbackReason,
    parse_leyline_directive,
)
from vllm_ascend.distributed.kv_transfer.leyline.reference import deepseek_yarn_inv_freq
from vllm_ascend.ops.leyline_mla import transform_mla_cache
from vllm_ascend.ops.rotary_embedding import get_native_mla_inv_freq

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass(frozen=True)
class LeylineTransformRequest:
    request_id: str
    session_id: str
    source_block_ids: tuple[int, ...]
    destination_block_ids: tuple[int, ...]
    target_start: int
    target_end: int
    delete_start: int
    delete_end: int
    source_length: int
    block_size: int
    inv_freq: tuple[float, ...]
    fault_rank: int | None = None
    fault_layer: int | None = None
    fault_stage: str | None = None

    @property
    def transformed_tokens(self) -> int:
        return self.target_end - self.target_start

    @property
    def transformed_destination_blocks(self) -> tuple[int, ...]:
        first = self.target_start // self.block_size
        last = self.target_end // self.block_size
        return self.destination_block_ids[first:last]


@dataclass
class LeylineConnectorMetadata(KVConnectorMetadata):
    requests: list[LeylineTransformRequest] = field(default_factory=list)


@dataclass(frozen=True)
class LeylineWorkerResult:
    success: bool
    duration_ms: float
    transformed_tokens: int
    transformed_layers: int
    expected_layers: int = 0
    rank: int = 0
    expected_ranks: int = 1
    successful_ranks: tuple[int, ...] = ()
    missing_layers: tuple[str, ...] = ()
    injection_reached: bool = False
    injected_rank: int | None = None
    injected_layer: int | None = None
    destination_writes: int = 0

    @property
    def observed_successful_ranks(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                set(
                    self.successful_ranks
                    or ((self.rank,) if self.success else ())
                )
            )
        )

    @property
    def missing_ranks(self) -> tuple[int, ...]:
        return tuple(
            sorted(set(range(self.expected_ranks)) - set(self.observed_successful_ranks))
        )

    @property
    def transform_complete(self) -> bool:
        return bool(
            self.success
            and self.expected_layers > 0
            and self.transformed_layers == self.expected_layers
            and not self.missing_layers
            and not self.missing_ranks
        )


@dataclass
class LeylineWorkerMetadata(KVConnectorWorkerMetadata):
    results: dict[str, LeylineWorkerResult] = field(default_factory=dict)

    def aggregate(self, other: KVConnectorWorkerMetadata) -> "LeylineWorkerMetadata":
        if not isinstance(other, LeylineWorkerMetadata):
            raise TypeError(f"cannot aggregate Leyline metadata with {type(other)!r}")
        merged = dict(self.results)
        for request_id, incoming in other.results.items():
            current = merged.get(request_id)
            if current is None:
                merged[request_id] = incoming
            else:
                merged[request_id] = LeylineWorkerResult(
                    success=current.success and incoming.success,
                    duration_ms=max(current.duration_ms, incoming.duration_ms),
                    transformed_tokens=min(current.transformed_tokens, incoming.transformed_tokens),
                    transformed_layers=min(current.transformed_layers, incoming.transformed_layers),
                    expected_layers=max(current.expected_layers, incoming.expected_layers),
                    rank=min(current.rank, incoming.rank),
                    expected_ranks=max(current.expected_ranks, incoming.expected_ranks),
                    successful_ranks=tuple(
                        sorted(
                            set(current.successful_ranks or ((current.rank,) if current.success else ()))
                            | set(incoming.successful_ranks or ((incoming.rank,) if incoming.success else ()))
                        )
                    ),
                    missing_layers=tuple(sorted(set(current.missing_layers) | set(incoming.missing_layers))),
                    injection_reached=current.injection_reached or incoming.injection_reached,
                    injected_rank=(
                        current.injected_rank
                        if current.injection_reached
                        else incoming.injected_rank
                    ),
                    injected_layer=(
                        current.injected_layer
                        if current.injection_reached
                        else incoming.injected_layer
                    ),
                    destination_writes=max(current.destination_writes, incoming.destination_writes),
                )
        return LeylineWorkerMetadata(merged)


@dataclass(frozen=True)
class _SourceIdentity:
    engine: tuple[str, ...]
    cache_salt: str | None
    lora: tuple[str, str, str] | None


@dataclass(frozen=True)
class _SourceSession:
    session_id: str
    token_ids: tuple[int, ...]
    block_ids: tuple[int, ...]
    identity: _SourceIdentity


@dataclass(frozen=True)
class _Match:
    session: _SourceSession
    deletion: DeletionMapping
    local_computed_tokens: int
    reusable_end: int

    @property
    def external_tokens(self) -> int:
        return self.reusable_end - self.local_computed_tokens


@dataclass
class _Outcome:
    applied: bool = False
    transformed_tokens: int = 0
    local_apc_tokens: int = 0
    normal_prefill_tokens: int = 0
    duration_ms: float = 0.0
    fallback_reason: LeylineFallbackReason | None = None
    expected_layers: int = 0
    transformed_layers: int = 0
    expected_ranks: int = 0
    successful_ranks: int = 0
    missing_layers: tuple[str, ...] = ()
    missing_ranks: tuple[int, ...] = ()
    transform_complete: bool = False
    injection_reached: bool = False
    injected_rank: int | None = None
    injected_layer: int | None = None
    destination_writes: int = 0
    invalidated_destination_blocks: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "transformed_tokens": self.transformed_tokens,
            "local_apc_tokens": self.local_apc_tokens,
            "normal_prefill_tokens": self.normal_prefill_tokens,
            "transform_duration_ms": self.duration_ms,
            "fallback_reason": self.fallback_reason.value if self.fallback_reason else None,
            "expected_layers": self.expected_layers,
            "transformed_layers": self.transformed_layers,
            "expected_ranks": self.expected_ranks,
            "successful_ranks": self.successful_ranks,
            "missing_layers": list(self.missing_layers),
            "missing_ranks": list(self.missing_ranks),
            "transform_complete": self.transform_complete,
            "injection_reached": self.injection_reached,
            "injected_rank": self.injected_rank,
            "injected_layer": self.injected_layer,
            "destination_writes": self.destination_writes,
            "invalidated_destination_blocks": self.invalidated_destination_blocks,
        }


class LeylineConnector(KVConnectorBase_V1, SupportsHMA):
    """One-shot local cache transform using the vLLM async-load lifecycle."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        self._block_size = vllm_config.cache_config.block_size
        self._engine_identity = self._make_engine_identity(vllm_config)
        self._runtime_supported = self._is_supported_runtime(vllm_config, kv_cache_config)
        self._inv_freq = self._make_inv_freq(vllm_config)
        self._expected_mla_layers = int(vllm_config.model_config.hf_text_config.num_hidden_layers)
        self._expected_tp_ranks = int(vllm_config.parallel_config.tensor_parallel_size)

        self._block_pool: BlockPool | None = None
        self._sessions: dict[str, _SourceSession] = {}
        self._directives: dict[str, LeylineDirective] = {}
        self._matches: dict[str, _Match] = {}
        self._metadata_pending: dict[str, LeylineTransformRequest] = {}
        self._inflight: dict[str, LeylineTransformRequest] = {}
        self._outcomes: dict[str, _Outcome] = {}

        self._kv_caches: dict[str, Any] = {}
        self._finished_recving: set[str] = set()
        self._invalid_block_ids: set[int] = set()
        self._worker_results: dict[str, LeylineWorkerResult] = {}

    @staticmethod
    def _rope_parameters(text_config: Any) -> dict[str, Any]:
        return dict(
            getattr(text_config, "rope_parameters", None)
            or getattr(text_config, "rope_scaling", None)
            or {}
        )

    @staticmethod
    def _make_engine_identity(vllm_config: "VllmConfig") -> tuple[str, ...]:
        model = vllm_config.model_config
        text = model.hf_text_config
        return (
            str(getattr(model, "model", None)),
            str(getattr(model, "revision", None)),
            str(getattr(model, "tokenizer", None)),
            str(getattr(model, "tokenizer_revision", None)),
            repr(LeylineConnector._rope_parameters(text)),
            str(getattr(text, "rope_theta", None)),
            str(getattr(text, "qk_rope_head_dim", None)),
            str(vllm_config.cache_config.cache_dtype),
            str(vllm_config.cache_config.block_size),
        )

    @staticmethod
    def _is_supported_runtime(vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig") -> bool:
        model = vllm_config.model_config
        parallel = vllm_config.parallel_config
        cache = vllm_config.cache_config
        scheduler = vllm_config.scheduler_config
        rope_parameters = LeylineConnector._rope_parameters(model.hf_text_config)
        return bool(
            getattr(model.hf_text_config, "model_type", None) == "deepseek_v2"
            and rope_parameters.get("rope_type", rope_parameters.get("type"))
            in {"yarn", "deepseek_yarn"}
            and str(getattr(model, "dtype", None)) == "torch.bfloat16"
            and getattr(model, "enforce_eager", False)
            and getattr(model, "quantization", None) is None
            and str(cache.cache_dtype) in {"auto", "bfloat16", "torch.bfloat16"}
            and parallel.tensor_parallel_size == 4
            and parallel.pipeline_parallel_size == 1
            and parallel.data_parallel_size == 1
            and parallel.decode_context_parallel_size == 1
            and parallel.prefill_context_parallel_size == 1
            and cache.block_size == 128
            and cache.enable_prefix_caching
            and vllm_config.speculative_config is None
            and len(kv_cache_config.kv_cache_groups) == 1
            and scheduler.enable_chunked_prefill
        )

    @staticmethod
    def _make_inv_freq(vllm_config: "VllmConfig") -> tuple[float, ...]:
        text = vllm_config.model_config.hf_text_config
        rope_scaling = LeylineConnector._rope_parameters(text)
        values = deepseek_yarn_inv_freq(
            rotary_dim=int(text.qk_rope_head_dim),
            base=float(getattr(text, "rope_theta", 10000.0)),
            scaling_factor=float(rope_scaling.get("factor", 1.0)),
            original_max_position_embeddings=int(
                rope_scaling.get(
                    "original_max_position_embeddings",
                    getattr(text, "max_position_embeddings", 4096),
                )
            ),
            beta_fast=float(rope_scaling.get("beta_fast", 32)),
            beta_slow=float(rope_scaling.get("beta_slow", 1)),
            extrapolation_factor=float(rope_scaling.get("extrapolation_factor", 1)),
        )
        return tuple(float(value) for value in values)

    def _request_identity(self, request: "Request") -> _SourceIdentity:
        lora = request.lora_request
        lora_identity = None
        if lora is not None:
            lora_identity = (
                str(getattr(lora, "lora_name", None)),
                str(getattr(lora, "lora_int_id", None)),
                str(getattr(lora, "lora_path", None)),
            )
        return _SourceIdentity(self._engine_identity, request.cache_salt, lora_identity)

    def _set_fallback(self, request_id: str, reason: LeylineFallbackReason) -> None:
        outcome = self._outcomes.setdefault(request_id, _Outcome())
        outcome.fallback_reason = reason
        logger.info("Leyline fallback request=%s reason=%s", request_id, reason.value)

    def on_new_request(self, request: "Request") -> None:
        try:
            directive = parse_leyline_directive(request.kv_transfer_params)
        except LeylineDirectiveError as exc:
            self._set_fallback(request.request_id, exc.reason)
            return
        if directive is not None:
            if (
                directive.fault_injection is not None
                and envs.VLLM_ASCEND_LEYLINE_FAULT_INJECTION != "validation-only"
            ):
                self._set_fallback(
                    request.request_id, LeylineFallbackReason.INVALID_DIRECTIVE
                )
                return
            self._directives[request.request_id] = directive
            self._outcomes.setdefault(request.request_id, _Outcome())

    def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        self._block_pool = gpu_block_pool

    def _source_blocks_are_pinned(self, session: _SourceSession) -> bool:
        if self._block_pool is None:
            return False
        return all(
            0 <= block_id < len(self._block_pool.blocks)
            and self._block_pool.blocks[block_id].ref_cnt > 0
            and self._block_pool.blocks[block_id].block_hash is not None
            for block_id in session.block_ids
        )

    def _compute_match(self, request: "Request", local_computed_tokens: int) -> _Match | None:
        directive = self._directives.get(request.request_id)
        if directive is None or directive.action is not LeylineAction.AMORTIZE:
            return None
        if not self._runtime_supported or request.prompt_token_ids is None or request.prompt_embeds is not None:
            self._set_fallback(request.request_id, LeylineFallbackReason.UNSUPPORTED_RUNTIME)
            return None
        if request.mm_features:
            self._set_fallback(request.request_id, LeylineFallbackReason.UNSUPPORTED_RUNTIME)
            return None

        session = self._sessions.get(directive.session_id)
        if session is None:
            self._set_fallback(request.request_id, LeylineFallbackReason.MISSING_SOURCE)
            return None
        if session.identity != self._request_identity(request):
            self._set_fallback(request.request_id, LeylineFallbackReason.INCOMPATIBLE_IDENTITY)
            return None
        if not self._source_blocks_are_pinned(session):
            self._set_fallback(request.request_id, LeylineFallbackReason.SOURCE_BLOCK_MISSING)
            return None

        assert directive.delete is not None
        try:
            deletion = validate_deletion(session.token_ids, request.prompt_token_ids, directive.delete)
        except LeylineDirectiveError as exc:
            self._set_fallback(request.request_id, exc.reason)
            return None

        max_target_tokens = min(max(request.num_tokens - 1, 0), deletion.edited_source_length)
        reusable_end = find_reusable_target_end(
            deletion,
            local_computed_tokens,
            max_target_tokens,
            range(len(session.block_ids)),
            self._block_size,
        )
        if reusable_end <= local_computed_tokens:
            self._set_fallback(request.request_id, LeylineFallbackReason.NO_REUSABLE_BLOCKS)
            return None
        return _Match(session, deletion, local_computed_tokens, reusable_end)

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int | None, bool]:
        match = self._compute_match(request, num_computed_tokens)
        if match is None:
            outcome = self._outcomes.get(request.request_id)
            if outcome is not None:
                outcome.local_apc_tokens = num_computed_tokens
                outcome.normal_prefill_tokens = max(
                    request.num_prompt_tokens - num_computed_tokens,
                    0,
                )
            return 0, False
        self._matches[request.request_id] = match
        return match.external_tokens, True

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        if num_external_tokens <= 0:
            return
        match = self._matches.get(request.request_id)
        if match is None or match.external_tokens != num_external_tokens:
            raise RuntimeError("Leyline allocation does not match the scheduler query")
        block_ids = blocks.get_block_ids()
        if block_ids is None or len(block_ids) != 1:
            raise RuntimeError("Leyline v0 requires exactly one KV cache group")

        directive = self._directives[request.request_id]
        assert directive.delete is not None
        plan = LeylineTransformRequest(
            request_id=request.request_id,
            session_id=directive.session_id,
            source_block_ids=match.session.block_ids,
            destination_block_ids=tuple(block_ids[0]),
            target_start=match.local_computed_tokens,
            target_end=match.reusable_end,
            delete_start=directive.delete.start,
            delete_end=directive.delete.end,
            source_length=match.deletion.source_length,
            block_size=self._block_size,
            inv_freq=self._inv_freq,
            fault_rank=(directive.fault_injection.rank if directive.fault_injection else None),
            fault_layer=(directive.fault_injection.layer if directive.fault_injection else None),
            fault_stage=(directive.fault_injection.stage if directive.fault_injection else None),
        )
        # The session owns one reference, and every accepted transaction owns
        # another. This keeps the source resident until all concurrent TP-wide
        # transforms that already matched the session have completed.
        if self._block_pool is None:
            raise RuntimeError("Leyline GPU block pool is not bound")
        self._block_pool.touch(
            [self._block_pool.blocks[block_id] for block_id in plan.source_block_ids]
        )
        self._metadata_pending[request.request_id] = plan
        self._inflight[request.request_id] = plan
        self._outcomes[request.request_id] = _Outcome(
            transformed_tokens=plan.transformed_tokens,
            local_apc_tokens=match.local_computed_tokens,
            normal_prefill_tokens=max(
                request.num_prompt_tokens - match.local_computed_tokens - plan.transformed_tokens,
                0,
            ),
        )

    def build_connector_meta(self, scheduler_output: Any) -> KVConnectorMetadata:
        del scheduler_output
        metadata = LeylineConnectorMetadata(list(self._metadata_pending.values()))
        self._metadata_pending.clear()
        return metadata

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        worker_meta = connector_output.kv_connector_worker_meta
        worker_results = worker_meta.results if isinstance(worker_meta, LeylineWorkerMetadata) else {}
        invalid = connector_output.invalid_block_ids
        for request_id in connector_output.finished_recving or ():
            plan = self._inflight.pop(request_id, None)
            if plan is None:
                continue
            result = worker_results.get(request_id)
            failed = bool(set(plan.transformed_destination_blocks) & invalid) or result is None or not result.success
            outcome = self._outcomes.setdefault(request_id, _Outcome())
            if result is not None:
                outcome.duration_ms = result.duration_ms
                outcome.expected_layers = result.expected_layers
                outcome.transformed_layers = result.transformed_layers
                outcome.expected_ranks = result.expected_ranks
                outcome.successful_ranks = len(result.observed_successful_ranks)
                outcome.missing_layers = result.missing_layers
                outcome.missing_ranks = result.missing_ranks
                outcome.transform_complete = result.transform_complete
                outcome.injection_reached = result.injection_reached
                outcome.injected_rank = result.injected_rank
                outcome.injected_layer = result.injected_layer
                outcome.destination_writes = result.destination_writes
            failed = failed or result is None or not result.transform_complete
            if failed:
                outcome.applied = False
                outcome.fallback_reason = LeylineFallbackReason.TRANSFORM_FAILED
                outcome.normal_prefill_tokens += outcome.transformed_tokens
                invalid.update(plan.transformed_destination_blocks)
                outcome.invalidated_destination_blocks = len(
                    plan.transformed_destination_blocks
                )
            else:
                outcome.applied = True
                outcome.fallback_reason = None
            self._release_plan_pin(plan)
            self._release_session(plan.session_id)
            self._matches.pop(request_id, None)

    def _release_plan_pin(self, plan: LeylineTransformRequest) -> None:
        if self._block_pool is None:
            return
        self._block_pool.free_blocks(
            [self._block_pool.blocks[block_id] for block_id in plan.source_block_ids]
        )

    def _release_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is None or self._block_pool is None:
            return
        self._block_pool.free_blocks([self._block_pool.blocks[block_id] for block_id in session.block_ids])

    def _record_source(self, request: "Request", block_ids: list[int], session_id: str) -> bool:
        if self._block_pool is None or request.prompt_token_ids is None:
            return False
        # Generated tokens may already be present in ``all_token_ids`` when the
        # finish hook runs. A record directive names the input prompt, so retain
        # only its full cache blocks and token sequence.
        full_blocks = min(len(block_ids), len(request.prompt_token_ids) // self._block_size)
        if full_blocks == 0:
            return False
        source_block_ids = tuple(block_ids[:full_blocks])
        blocks = [self._block_pool.blocks[block_id] for block_id in source_block_ids]
        if any(block.block_hash is None for block in blocks):
            return False

        self._release_session(session_id)
        self._block_pool.touch(blocks)
        self._sessions[session_id] = _SourceSession(
            session_id=session_id,
            token_ids=tuple(request.prompt_token_ids),
            block_ids=source_block_ids,
            identity=self._request_identity(request),
        )
        return True

    def _finish_request(self, request: "Request", block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        directive = self._directives.pop(request.request_id, None)
        if directive is None:
            # Invalid opt-in directives deliberately have no parsed directive,
            # but must still return their typed fallback and release bookkeeping.
            outcome = self._outcomes.pop(request.request_id, None)
            self._matches.pop(request.request_id, None)
            if outcome is None:
                return False, None
            return False, {"leyline": {**outcome.as_dict(), "recorded": False}}

        normal_finish = request.status.name in {
            "FINISHED_STOPPED",
            "FINISHED_LENGTH_CAPPED",
            "FINISHED_REPETITION",
        }
        recorded = bool(
            directive.action is LeylineAction.RECORD
            and normal_finish
            and self._record_source(request, block_ids, directive.session_id)
        )
        outcome = self._outcomes.pop(request.request_id, _Outcome())
        self._matches.pop(request.request_id, None)
        self._metadata_pending.pop(request.request_id, None)
        plan = self._inflight.pop(request.request_id, None)
        if plan is not None:
            self._release_plan_pin(plan)
            self._release_session(plan.session_id)
        elif directive.action is LeylineAction.AMORTIZE:
            self._release_session(directive.session_id)
        cleanup = {
            "sessions": int(directive.session_id in self._sessions),
            "inflight": int(request.request_id in self._inflight),
            "pending": int(request.request_id in self._metadata_pending),
            "matches": int(request.request_id in self._matches),
            "transaction_owned_references": int(request.request_id in self._inflight),
        }
        return False, {
            "leyline": {**outcome.as_dict(), "recorded": recorded, "cleanup": cleanup}
        }

    def request_finished(self, request: "Request", block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        return self._finish_request(request, block_ids)

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        if len(block_ids) != 1:
            self._set_fallback(request.request_id, LeylineFallbackReason.UNSUPPORTED_RUNTIME)
            directive = self._directives.pop(request.request_id, None)
            self._matches.pop(request.request_id, None)
            self._metadata_pending.pop(request.request_id, None)
            plan = self._inflight.pop(request.request_id, None)
            if plan is not None:
                self._release_plan_pin(plan)
                self._release_session(plan.session_id)
            elif directive is not None and directive.action is LeylineAction.AMORTIZE:
                self._release_session(directive.session_id)
            outcome = self._outcomes.pop(request.request_id, _Outcome())
            return False, {"leyline": {**outcome.as_dict(), "recorded": False}}
        return self._finish_request(request, block_ids[0])

    # Worker-side lifecycle -------------------------------------------------

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        self._kv_caches = dict(kv_caches)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        del forward_context, kwargs
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, LeylineConnectorMetadata):
            raise TypeError(f"unexpected Leyline connector metadata: {type(metadata)!r}")

        for request in metadata.requests:
            started = time.perf_counter()
            transformed_layers = 0
            missing_layers: tuple[str, ...] = ()
            success = True
            injection_reached = False
            injected_rank: int | None = None
            injected_layer: int | None = None
            destination_writes = 0
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_available() and torch.distributed.is_initialized()
                else 0
            )
            try:
                deletion = DeletionMapping(
                    source_length=request.source_length,
                    edited_source_length=request.source_length - (request.delete_end - request.delete_start),
                    delete=DeleteSpan(request.delete_start, request.delete_end),
                )
                slots = build_slot_mapping(
                    deletion,
                    request.target_start,
                    request.target_end,
                    dict(enumerate(request.source_block_ids)),
                    request.destination_block_ids,
                    request.block_size,
                )
                compatible_layers = [
                    layer_name
                    for layer_name, cache in self._kv_caches.items()
                    if isinstance(cache, (tuple, list))
                    and len(cache) >= 2
                    and cache[0].ndim == 4
                    and cache[1].ndim == 4
                    and cache[0].shape[-1] == 512
                    and cache[1].shape[-1] == 64
                ]
                expected_layer_count = self._expected_mla_layers
                missing_layers = tuple(
                    sorted(
                        layer_name
                        for layer_name, cache in self._kv_caches.items()
                        if layer_name not in compatible_layers
                    )
                )
                if len(compatible_layers) != expected_layer_count:
                    missing_layers = (
                        *missing_layers,
                        f"expected={expected_layer_count}:compatible={len(compatible_layers)}",
                    )
                    raise RuntimeError(
                        "incomplete MLA cache registration: "
                        f"expected {expected_layer_count}, found {len(compatible_layers)}"
                    )
                for layer_name, cache in self._kv_caches.items():
                    if not isinstance(cache, (tuple, list)) or len(cache) < 2:
                        continue
                    ckv_cache, kpe_cache = cache[0], cache[1]
                    if ckv_cache.ndim != 4 or kpe_cache.ndim != 4:
                        continue
                    if ckv_cache.shape[-1] != 512 or kpe_cache.shape[-1] != 64:
                        continue
                    inv_freq = torch.tensor(request.inv_freq, dtype=torch.float32, device=ckv_cache.device)
                    pending_capture = prepare_capture(
                        request_id=request.request_id,
                        session_id=request.session_id,
                        layer_name=layer_name,
                        ckv_cache=ckv_cache,
                        kpe_cache=kpe_cache,
                        source_slots=slots.source_slots,
                        destination_slots=slots.destination_slots,
                        old_positions=slots.old_positions,
                        new_positions=slots.new_positions,
                        inv_freq=inv_freq,
                        native_inv_freq=get_native_mla_inv_freq(),
                        block_size=request.block_size,
                        expected_layers=compatible_layers,
                    )
                    transform_mla_cache(
                        ckv_cache,
                        kpe_cache,
                        slots.source_slots,
                        slots.destination_slots,
                        slots.old_positions,
                        slots.new_positions,
                        inv_freq,
                    )
                    destination_writes += 1
                    if pending_capture is not None:
                        torch.npu.synchronize()
                        finish_capture(pending_capture, ckv_cache, kpe_cache)
                    transformed_layers += 1
                    completed_layer_index = transformed_layers - 1
                    if (
                        request.fault_stage == "after_layer_write"
                        and request.fault_rank == rank
                        and request.fault_layer == completed_layer_index
                    ):
                        injection_reached = True
                        injected_rank = rank
                        injected_layer = completed_layer_index
                        raise RuntimeError(
                            "validation-only Leyline fault injection after layer write"
                        )
                    logger.debug("Leyline transformed layer=%s request=%s", layer_name, request.request_id)
                if transformed_layers != expected_layer_count:
                    raise RuntimeError(
                        f"incomplete MLA transformation: expected {expected_layer_count}, "
                        f"transformed {transformed_layers}"
                    )
                if not capture_enabled():
                    torch.npu.synchronize()
            except Exception:
                logger.exception("Leyline cache transformation failed request=%s", request.request_id)
                success = False
                self._invalid_block_ids.update(request.transformed_destination_blocks)

            expected_ranks = self._expected_tp_ranks
            self._worker_results[request.request_id] = LeylineWorkerResult(
                success=success,
                duration_ms=(time.perf_counter() - started) * 1000,
                transformed_tokens=request.transformed_tokens,
                transformed_layers=transformed_layers,
                expected_layers=self._expected_mla_layers,
                rank=rank,
                expected_ranks=expected_ranks,
                successful_ranks=(rank,) if success else (),
                missing_layers=missing_layers,
                injection_reached=injection_reached,
                injected_rank=injected_rank,
                injected_layer=injected_layer,
                destination_writes=destination_writes,
            )
            self._finished_recving.add(request.request_id)

    def wait_for_layer_load(self, layer_name: str) -> None:
        del layer_name

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        del layer_name, kv_layer, attn_metadata, kwargs

    def wait_for_save(self) -> None:
        return

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        del finished_req_ids
        finished = self._finished_recving
        self._finished_recving = set()
        return None, finished or None

    def get_block_ids_with_load_errors(self) -> set[int]:
        invalid = self._invalid_block_ids
        self._invalid_block_ids = set()
        return invalid

    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata | None:
        if not self._worker_results:
            return None
        metadata = LeylineWorkerMetadata(self._worker_results)
        self._worker_results = {}
        return metadata

    def shutdown(self) -> None:
        for plan in list(self._inflight.values()):
            self._release_plan_pin(plan)
        for session_id in list(self._sessions):
            self._release_session(session_id)
        self._metadata_pending.clear()
        self._inflight.clear()
        self._directives.clear()
        self._matches.clear()
        self._outcomes.clear()
        self._finished_recving.clear()
        self._invalid_block_ids.clear()
        self._worker_results.clear()
