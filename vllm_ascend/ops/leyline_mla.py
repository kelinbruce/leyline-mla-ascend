# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Torch implementation of the Leyline MLA page transformation."""

from collections.abc import Sequence

import torch


def transform_mla_cache(
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    source_slots: Sequence[int] | torch.Tensor,
    destination_slots: Sequence[int] | torch.Tensor,
    old_positions: Sequence[int] | torch.Tensor,
    new_positions: Sequence[int] | torch.Tensor,
    inv_freq: torch.Tensor,
) -> None:
    """Copy cKV and delta-rotate Kpe into destination physical slots.

    Source rows are cloned before any destination write. This is required when
    block allocation happens to make source and destination ranges overlap.
    """

    if ckv_cache.ndim != 4 or kpe_cache.ndim != 4:
        raise ValueError("MLA caches must have [blocks, block, heads, dim] layout")
    if ckv_cache.shape[:3] != kpe_cache.shape[:3]:
        raise ValueError("cKV and Kpe cache page layouts must match")
    if kpe_cache.shape[-1] != inv_freq.numel() * 2:
        raise ValueError("inv_freq does not match the Kpe dimension")

    device = ckv_cache.device
    source = torch.as_tensor(source_slots, dtype=torch.long, device=device)
    destination = torch.as_tensor(destination_slots, dtype=torch.long, device=device)
    old = torch.as_tensor(old_positions, dtype=torch.float32, device=device)
    new = torch.as_tensor(new_positions, dtype=torch.float32, device=device)
    if not (source.numel() == destination.numel() == old.numel() == new.numel()):
        raise ValueError("slot and position arrays must have the same length")
    if source.numel() == 0:
        return

    ckv_flat = ckv_cache.reshape(-1, *ckv_cache.shape[2:])
    kpe_flat = kpe_cache.reshape(-1, *kpe_cache.shape[2:])
    if source.min().item() < 0 or source.max().item() >= ckv_flat.shape[0]:
        raise IndexError("source slot is outside the MLA cache")
    if destination.min().item() < 0 or destination.max().item() >= ckv_flat.shape[0]:
        raise IndexError("destination slot is outside the MLA cache")

    source_ckv = ckv_flat.index_select(0, source).clone()
    source_kpe = kpe_flat.index_select(0, source).to(torch.float32).clone()
    frequencies = inv_freq.to(device=device, dtype=torch.float32).unsqueeze(0)
    # Preserve the FP32 rounding used when the native absolute-position RoPE
    # tables were built, then take the phase difference.
    angles = new.unsqueeze(1) * frequencies - old.unsqueeze(1) * frequencies
    cos = angles.cos().unsqueeze(1)
    sin = angles.sin().unsqueeze(1)
    half = source_kpe.shape[-1] // 2
    first = source_kpe[..., :half]
    second = source_kpe[..., half:]
    rotated = torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)

    ckv_flat.index_copy_(0, destination, source_ckv)
    kpe_flat.index_copy_(0, destination, rotated.to(kpe_cache.dtype))
