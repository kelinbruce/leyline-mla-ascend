# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Correctness-first tensor operations for the Ascend 310P MLA validator.

The helpers intentionally use ordinary Torch operations.  They are not a
performance backend; keeping them independent from vLLM objects makes the MLA
math testable on CPU and executable on 310P without unsupported fused kernels.
"""

from __future__ import annotations

import math

import torch


def rms_norm_fp32(inputs: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Apply RMSNorm with FP32 variance accumulation and input-dtype output."""

    if inputs.shape[-1] != weight.numel():
        raise ValueError("RMSNorm weight does not match the input dimension")
    variance = inputs.to(torch.float32).square().mean(dim=-1, keepdim=True)
    normalized = inputs.to(torch.float32) * torch.rsqrt(variance + epsilon)
    return (normalized * weight.to(device=inputs.device, dtype=torch.float32)).to(inputs.dtype)


def _rope_rows(values: torch.Tensor, num_tokens: int, rotary_dim: int, name: str) -> torch.Tensor:
    if values.shape[-1] != rotary_dim:
        raise ValueError(f"{name} does not match the rotary dimension")
    rows = values.reshape(-1, rotary_dim)
    if rows.shape[0] == 1:
        return rows.expand(num_tokens, rotary_dim)
    if rows.shape[0] != num_tokens:
        raise ValueError(f"{name} must provide one row per token")
    return rows


def deepseek_interleaved_rope(
    inputs: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Reproduce ``npu_interleave_rope`` using transparent tensor math.

    DeepSeek's input arrives pair-interleaved.  Ascend first groups even and
    odd coordinates into two halves and then applies the standard half-split
    rotary transform.
    """

    rotary_dim = inputs.shape[-1]
    if rotary_dim % 2:
        raise ValueError("rotary dimension must be even")
    num_tokens = inputs.shape[0]
    cos_rows = _rope_rows(cos, num_tokens, rotary_dim, "cos")
    sin_rows = _rope_rows(sin, num_tokens, rotary_dim, "sin")

    interleaved = torch.cat((inputs[..., 0::2], inputs[..., 1::2]), dim=-1).to(torch.float32)
    half = rotary_dim // 2
    rotated_half = torch.cat((-interleaved[..., half:], interleaved[..., :half]), dim=-1)
    broadcast_shape = (num_tokens,) + (1,) * (inputs.ndim - 2) + (rotary_dim,)
    result = interleaved * cos_rows.view(broadcast_shape).to(torch.float32)
    result = result + rotated_half * sin_rows.view(broadcast_shape).to(torch.float32)
    return result.to(inputs.dtype)


def write_logical_mla_cache(
    kv_no_split: torch.Tensor,
    gamma: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    slots: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    *,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize/rotate projected KV rows and write logical paged caches.

    Returns ``(kpe_rows, ckv_rows)`` for immediate attention use.  Negative
    slots are padding and intentionally do not update cache storage.
    """

    if ckv_cache.ndim != 4 or kpe_cache.ndim != 4:
        raise ValueError("MLA caches must use [blocks, block, heads, dim]")
    if ckv_cache.shape[:3] != kpe_cache.shape[:3]:
        raise ValueError("cKV and Kpe page layouts must match")
    if ckv_cache.shape[2] != 1:
        raise ValueError("validation MLA requires one latent KV head")

    num_tokens = kv_no_split.shape[0]
    ckv_dim = ckv_cache.shape[-1]
    kpe_dim = kpe_cache.shape[-1]
    rows = kv_no_split.reshape(num_tokens, -1)
    if rows.shape[-1] != ckv_dim + kpe_dim:
        raise ValueError("projected KV dimension does not match the logical caches")

    ckv_rows = rms_norm_fp32(rows[:, :ckv_dim], gamma, epsilon).view(num_tokens, 1, ckv_dim)
    raw_kpe = rows[:, ckv_dim:].view(num_tokens, 1, kpe_dim)
    kpe_rows = deepseek_interleaved_rope(raw_kpe, cos, sin)

    slot_rows = slots.reshape(-1).to(device=ckv_cache.device, dtype=torch.long)
    if slot_rows.numel() != num_tokens:
        raise ValueError("slot mapping must contain one entry per token")
    valid = slot_rows >= 0
    if bool(valid.any().item()):
        flat_ckv = ckv_cache.reshape(-1, 1, ckv_dim)
        flat_kpe = kpe_cache.reshape(-1, 1, kpe_dim)
        valid_slots = slot_rows[valid]
        if bool((valid_slots >= flat_ckv.shape[0]).any().item()):
            raise IndexError("slot mapping is outside the logical MLA cache")
        flat_ckv.index_copy_(0, valid_slots, ckv_rows[valid])
        flat_kpe.index_copy_(0, valid_slots, kpe_rows[valid])
    return kpe_rows, ckv_rows


def gather_paged_rows(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    context_length: int,
) -> torch.Tensor:
    """Gather the first request's logical cache rows in token order."""

    if cache.ndim != 4:
        raise ValueError("paged cache must use [blocks, block, heads, dim]")
    if context_length < 0:
        raise ValueError("context length cannot be negative")
    if context_length == 0:
        return cache.new_empty((0, cache.shape[2], cache.shape[3]))
    if block_table.ndim not in (1, 2):
        raise ValueError("block table must be one- or two-dimensional")
    if block_table.ndim == 2 and block_table.shape[0] != 1:
        raise ValueError("validation MLA supports one request at a time")

    block_size = cache.shape[1]
    required_blocks = math.ceil(context_length / block_size)
    blocks = block_table.reshape(-1)[:required_blocks].to(device=cache.device, dtype=torch.long)
    if blocks.numel() != required_blocks:
        raise ValueError("block table does not cover the requested context")
    if bool(((blocks < 0) | (blocks >= cache.shape[0])).any().item()):
        raise IndexError("block table references an invalid cache block")
    rows = cache.index_select(0, blocks).reshape(-1, cache.shape[2], cache.shape[3])
    return rows[:context_length]


def offset_causal_mask(
    query_length: int,
    key_length: int,
    context_length: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return a mask where query ``i`` can see cached context plus ``0..i``."""

    if min(query_length, key_length, context_length) < 0:
        raise ValueError("attention lengths cannot be negative")
    if key_length != context_length + query_length:
        raise ValueError("key length must equal cached context plus query length")
    query_positions = torch.arange(query_length, device=device).unsqueeze(1) + context_length
    key_positions = torch.arange(key_length, device=device).unsqueeze(0)
    return key_positions <= query_positions


def dense_mla_attention(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float,
    context_length: int,
) -> torch.Tensor:
    """Compute one-request causal MLA attention with FP32 softmax."""

    query_length, num_heads = q_nope.shape[:2]
    key_length = k_nope.shape[0]
    if q_pe.shape[:2] != (query_length, num_heads):
        raise ValueError("q_nope and q_pe layouts must match")
    if k_nope.shape[:2] != (key_length, num_heads) or k_pe.shape[:2] != (key_length, num_heads):
        raise ValueError("key layouts must match the query head count")
    if value.shape[:2] != (key_length, num_heads):
        raise ValueError("value layout must match the key layout")
    if q_nope.shape[-1] != k_nope.shape[-1] or q_pe.shape[-1] != k_pe.shape[-1]:
        raise ValueError("query and key dimensions must match")

    q_nope_heads = q_nope.transpose(0, 1)
    q_pe_heads = q_pe.transpose(0, 1)
    k_nope_heads = k_nope.transpose(0, 1).transpose(1, 2)
    k_pe_heads = k_pe.transpose(0, 1).transpose(1, 2)
    scores = torch.bmm(q_nope_heads, k_nope_heads).to(torch.float32)
    scores = scores + torch.bmm(q_pe_heads, k_pe_heads).to(torch.float32)
    scores.mul_(scale)

    mask = offset_causal_mask(
        query_length,
        key_length,
        context_length,
        device=q_nope.device,
    )
    scores.masked_fill_(~mask.unsqueeze(0), torch.finfo(torch.float32).min)
    probabilities = torch.softmax(scores, dim=-1).to(value.dtype)
    output = torch.bmm(probabilities, value.transpose(0, 1))
    return output.transpose(0, 1).contiguous()


def latent_decode_attention(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_rows: torch.Tensor,
    kpe_rows: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Compute absorbed MLA decode attention directly against latent cache."""

    query_length, num_heads = q_nope.shape[:2]
    if ckv_rows.shape[1] != 1 or kpe_rows.shape[1] != 1:
        raise ValueError("validation MLA requires one latent KV head")
    expanded_ckv = ckv_rows.expand(-1, num_heads, -1)
    expanded_kpe = kpe_rows.expand(-1, num_heads, -1)
    context_length = ckv_rows.shape[0] - query_length
    if context_length < 0:
        raise ValueError("decode cache is shorter than the query")
    return dense_mla_attention(
        q_nope,
        q_pe,
        expanded_ckv,
        expanded_kpe,
        expanded_ckv,
        scale=scale,
        context_length=context_length,
    )
