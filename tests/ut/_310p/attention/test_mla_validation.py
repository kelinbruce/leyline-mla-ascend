# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_ascend._310p.attention.mla_validation import (
    deepseek_interleaved_rope,
    dense_mla_attention,
    gather_paged_rows,
    latent_decode_attention,
    offset_causal_mask,
    rms_norm_fp32,
    write_logical_mla_cache,
)


def test_rms_norm_uses_fp32_reference() -> None:
    inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float16)
    weight = torch.tensor([1.0, 0.5, 2.0, 1.5], dtype=torch.float16)
    actual = rms_norm_fp32(inputs, weight, 1e-6)
    expected = inputs.float() * torch.rsqrt(inputs.float().square().mean(-1, keepdim=True) + 1e-6)
    expected = (expected * weight.float()).half()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_deepseek_interleaved_rope_groups_pairs_before_rotation() -> None:
    inputs = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    cos = torch.zeros(1, 1, 1, 4)
    sin = torch.ones(1, 1, 1, 4)
    actual = deepseek_interleaved_rope(inputs, cos, sin)
    expected = torch.tensor([[[-2.0, -4.0, 1.0, 3.0]]])
    torch.testing.assert_close(actual, expected)


def test_cache_write_maps_rows_and_ignores_padding_slots() -> None:
    ckv_cache = torch.zeros(2, 2, 1, 4)
    kpe_cache = torch.zeros(2, 2, 1, 4)
    projected = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0],
            [2.0, 2.0, 2.0, 2.0, 5.0, 6.0, 7.0, 8.0],
        ]
    )
    kpe_rows, ckv_rows = write_logical_mla_cache(
        projected,
        torch.ones(4),
        torch.ones(2, 1, 1, 4),
        torch.zeros(2, 1, 1, 4),
        torch.tensor([2, -1]),
        ckv_cache,
        kpe_cache,
        epsilon=1e-6,
    )

    torch.testing.assert_close(ckv_cache.reshape(-1, 1, 4)[2], ckv_rows[0])
    torch.testing.assert_close(kpe_cache.reshape(-1, 1, 4)[2], kpe_rows[0])
    torch.testing.assert_close(kpe_rows[0], torch.tensor([[1.0, 3.0, 2.0, 4.0]]))
    assert torch.count_nonzero(ckv_cache.reshape(-1, 1, 4)[3]) == 0


def test_gather_paged_rows_preserves_logical_token_order() -> None:
    cache = torch.arange(4 * 2, dtype=torch.float32).view(4, 2, 1, 1)
    actual = gather_paged_rows(cache, torch.tensor([[2, 0, 3]]), 5)
    expected = torch.tensor([4.0, 5.0, 0.0, 1.0, 6.0]).view(5, 1, 1)
    torch.testing.assert_close(actual, expected)


def test_offset_causal_mask_includes_cached_prefix() -> None:
    actual = offset_causal_mask(2, 5, 3, device=torch.device("cpu"))
    expected = torch.tensor(
        [
            [True, True, True, True, False],
            [True, True, True, True, True],
        ]
    )
    assert torch.equal(actual, expected)


def test_dense_prefill_matches_manual_reference() -> None:
    q_nope = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    q_pe = torch.zeros(2, 1, 2)
    k_nope = q_nope.clone()
    k_pe = torch.zeros_like(q_pe)
    value = torch.tensor([[[2.0, 4.0]], [[6.0, 8.0]]])
    actual = dense_mla_attention(
        q_nope,
        q_pe,
        k_nope,
        k_pe,
        value,
        scale=1.0,
        context_length=0,
    )

    second_weights = torch.softmax(torch.tensor([0.0, 1.0]), dim=0)
    expected_second = second_weights[0] * value[0] + second_weights[1] * value[1]
    torch.testing.assert_close(actual[0], value[0])
    torch.testing.assert_close(actual[1], expected_second)


def test_dense_cached_tail_sees_prefix_and_prior_tail_only() -> None:
    q_nope = torch.ones(2, 1, 1)
    q_pe = torch.zeros(2, 1, 1)
    k_nope = torch.ones(4, 1, 1)
    k_pe = torch.zeros(4, 1, 1)
    value = torch.tensor([1.0, 3.0, 5.0, 9.0]).view(4, 1, 1)
    actual = dense_mla_attention(
        q_nope,
        q_pe,
        k_nope,
        k_pe,
        value,
        scale=1.0,
        context_length=2,
    )
    torch.testing.assert_close(actual[:, 0, 0], torch.tensor([3.0, 4.5]))


def test_latent_decode_matches_expanded_dense_attention() -> None:
    q_nope = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    q_pe = torch.zeros(1, 2, 2)
    ckv = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]])
    kpe = torch.zeros(3, 1, 2)
    actual = latent_decode_attention(q_nope, q_pe, ckv, kpe, scale=1.0)
    expected = dense_mla_attention(
        q_nope,
        q_pe,
        ckv.expand(-1, 2, -1),
        kpe.expand(-1, 2, -1),
        ckv.expand(-1, 2, -1),
        scale=1.0,
        context_length=2,
    )
    torch.testing.assert_close(actual, expected)
