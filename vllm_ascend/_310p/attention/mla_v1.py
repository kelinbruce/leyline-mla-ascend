# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Correctness-first Ascend 310P MLA backend for Leyline validation.

This backend deliberately uses ordinary Torch tensor operations instead of
the fused MLA operators that are unavailable on 310P.  It is restricted to a
single request and the pinned DeepSeek-V2-Lite validation configuration; it is
not intended to be a performance or production backend.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm_ascend._310p.attention.mla_validation import (
    deepseek_interleaved_rope,
    dense_mla_attention,
    gather_paged_rows,
    latent_decode_attention,
    write_logical_mla_cache,
)
from vllm_ascend._310p.mla_runtime import require_310p_mla_runtime
from vllm_ascend.attention.mla_v1 import (
    AscendMLABackend,
    AscendMLAImpl,
    AscendMLAMetadata,
    AscendMLAMetadataBuilder,
)


class AscendMLABackend310(AscendMLABackend):
    """310P MLA backend with a logical ND cKV/Kpe cache contract."""

    @staticmethod
    def get_builder_cls() -> type[AscendMLAMetadataBuilder]:
        return AscendMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[AscendMLAImpl310]:
        return AscendMLAImpl310

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [128]


class AscendMLAImpl310(AscendMLAImpl):
    """Minimal unfused MLA implementation for 310P semantic validation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        require_310p_mla_runtime(self.vllm_config)
        if self.num_kv_heads != 1:
            raise ValueError(
                "Ascend 310P validation MLA requires one latent KV head per rank"
            )

    def rope_single(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        return deepseek_interleaved_rope(x, cos, sin)

    def _write_cache(
        self,
        kv_no_split: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(kv_cache) < 2:
            raise ValueError("Ascend 310P validation MLA requires cKV and Kpe caches")
        if self.kv_a_layernorm is None:
            raise ValueError("Ascend 310P validation MLA requires KV RMSNorm")
        return write_logical_mla_cache(
            kv_no_split,
            self.kv_a_layernorm.weight,
            cos,
            sin,
            slots,
            kv_cache[0],
            kv_cache[1],
            epsilon=self.kv_a_layernorm.variance_epsilon,
        )

    def exec_kv_prefill(
        self,
        kv_no_split: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._write_cache(kv_no_split, cos, sin, kv_cache, slots)

    def exec_kv_decode(
        self,
        kv_no_split: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._write_cache(kv_no_split, cos, sin, kv_cache, slots)
        # Decode consumes the complete logical caches, not only the new row.
        return kv_cache[1], kv_cache[0]

    @staticmethod
    def _one_length(values: Any, name: str) -> int:
        if isinstance(values, torch.Tensor):
            values = values.reshape(-1)
            if values.numel() != 1:
                raise ValueError(f"Ascend 310P validation MLA requires one {name}")
            return int(values[0].item())
        if len(values) != 1:
            raise ValueError(f"Ascend 310P validation MLA requires one {name}")
        return int(values[0])

    def _forward_prefill(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        value: torch.Tensor,
        kv_c_and_k_pe_cache: tuple[torch.Tensor, ...],
        attn_metadata: AscendMLAMetadata,
    ) -> torch.Tensor:
        prefill_meta = attn_metadata.prefill
        if prefill_meta is None:
            raise ValueError("prefill metadata is required")
        query_length = q_nope.shape[0]
        total_length = self._one_length(prefill_meta.context_lens, "prefill request")
        context_length = total_length - query_length
        if context_length < 0:
            raise ValueError("prefill sequence is shorter than its query tail")

        if context_length:
            cached_ckv = gather_paged_rows(
                kv_c_and_k_pe_cache[0],
                prefill_meta.block_table,
                context_length,
            )
            cached_kpe = gather_paged_rows(
                kv_c_and_k_pe_cache[1],
                prefill_meta.block_table,
                context_length,
            )
            cached_k_nope, cached_value = (
                self.kv_b_proj(cached_ckv)[0]
                .view(
                    context_length,
                    self.num_heads,
                    self.qk_nope_head_dim + self.v_head_dim,
                )
                .split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            )
            cached_kpe = cached_kpe.expand(
                context_length,
                self.num_heads,
                self.qk_rope_head_dim,
            )
            k_nope = torch.cat((cached_k_nope, k_nope), dim=0)
            k_pe = torch.cat((cached_kpe, k_pe), dim=0)
            value = torch.cat((cached_value, value), dim=0)

        attention = dense_mla_attention(
            q_nope,
            q_pe,
            k_nope,
            k_pe,
            value,
            scale=self.scale,
            context_length=context_length,
        )
        return attention.reshape(query_length, self.num_heads * self.v_head_dim)

    def _v_up_proj(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.num_heads, self.kv_lora_rank)
        projected = torch.bmm(x.transpose(0, 1), self.W_UV)
        return projected.transpose(0, 1).reshape(-1, self.num_heads * self.v_head_dim)

    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        block_size: int,
        attn_metadata: AscendMLAMetadata,
        dequant_scale_q_nope: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del block_size
        if dequant_scale_q_nope is not None:
            raise ValueError("quantized MLA decode is not supported on Ascend 310P")
        decode_meta = attn_metadata.decode
        if decode_meta is None:
            raise ValueError("decode metadata is required")
        if q_nope.shape[0] != 1:
            raise ValueError("Ascend 310P validation MLA supports one decode token")
        context_length = self._one_length(decode_meta.seq_lens_list, "decode request")
        ckv_rows = gather_paged_rows(k_nope, decode_meta.block_table, context_length)
        kpe_rows = gather_paged_rows(k_pe, decode_meta.block_table, context_length)
        latent_output = latent_decode_attention(
            q_nope,
            q_pe,
            ckv_rows,
            kpe_rows,
            scale=self.scale,
        )
        return self._v_up_proj(latent_output)
