# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Independent NumPy reference for DeepSeek YaRN cache rotation."""

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


def deepseek_yarn_inv_freq(
    rotary_dim: int,
    base: float,
    scaling_factor: float,
    original_max_position_embeddings: int,
    beta_fast: float = 32,
    beta_slow: float = 1,
    extrapolation_factor: float = 1,
) -> npt.NDArray[np.float32]:
    if rotary_dim <= 0 or rotary_dim % 2:
        raise ValueError("rotary_dim must be a positive even integer")
    if (
        base <= 0
        or scaling_factor <= 0
        or original_max_position_embeddings <= 0
        or extrapolation_factor < 0
    ):
        raise ValueError("YaRN parameters must be positive")

    dims = np.arange(0, rotary_dim, 2, dtype=np.float32)
    freq_extra = 1.0 / np.power(base, dims / rotary_dim)
    freq_inter = freq_extra / scaling_factor

    def correction_dim(num_rotations: float) -> float:
        return rotary_dim * math.log(original_max_position_embeddings / (num_rotations * 2 * math.pi)) / (
            2 * math.log(base)
        )

    low = max(0, math.floor(correction_dim(beta_fast)))
    high = min(rotary_dim // 2 - 1, math.ceil(correction_dim(beta_slow)))
    if low == high:
        high = low + 0.001
    ramp = np.clip((np.arange(rotary_dim // 2, dtype=np.float32) - low) / (high - low), 0, 1)
    inv_freq_mask = (1.0 - ramp) * extrapolation_factor
    return (freq_inter * (1.0 - inv_freq_mask) + freq_extra * inv_freq_mask).astype(np.float32)


def unit_delta_cos_sin(
    old_positions: Sequence[int],
    new_positions: Sequence[int],
    inv_freq: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    if len(old_positions) != len(new_positions):
        raise ValueError("old and new position arrays must have the same length")
    frequencies = np.asarray(inv_freq, dtype=np.float32)
    # Match DeepSeek's native cache construction: each absolute phase is
    # rounded in FP32 before subtraction.  Computing (new - old) * inv_freq
    # instead produces measurably different long-context phases.
    old_angles = np.outer(np.asarray(old_positions, dtype=np.float32), frequencies)
    new_angles = np.outer(np.asarray(new_positions, dtype=np.float32), frequencies)
    angles = new_angles - old_angles
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def rotate_kpe_half_split(
    kpe: npt.NDArray[np.generic],
    cos: npt.NDArray[np.float32],
    sin: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Rotate Ascend's transformed, NeoX-style cached Kpe layout."""

    values = np.asarray(kpe, dtype=np.float32)
    if values.shape[-1] % 2:
        raise ValueError("Kpe dimension must be even")
    half = values.shape[-1] // 2
    if cos.shape != sin.shape or cos.shape[-1] != half:
        raise ValueError("cos/sin shape does not match Kpe")
    broadcast_shape = (cos.shape[0],) + (1,) * (values.ndim - 2) + (half,)
    cos_b = cos.reshape(broadcast_shape)
    sin_b = sin.reshape(broadcast_shape)
    first = values[..., :half]
    second = values[..., half:]
    return np.concatenate((first * cos_b - second * sin_b, second * cos_b + first * sin_b), axis=-1)
