# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import vllm_ascend._310p.attention.mla_v1 as mla_310
from vllm_ascend._310p.attention.mla_v1 import (
    AscendMLABackend310,
    AscendMLAImpl310,
)
from vllm_ascend.attention.mla_v1 import AscendMLAImpl, AscendMLAMetadataBuilder


def test_mla_backend_310_exposes_logical_nd_cache() -> None:
    assert AscendMLABackend310.get_kv_cache_shape(8, 128, 1, 576) == (8, 128, 1, 576)
    assert AscendMLABackend310.get_supported_kernel_block_sizes() == [128]
    assert AscendMLABackend310.get_builder_cls() is AscendMLAMetadataBuilder
    assert AscendMLABackend310.get_impl_cls() is AscendMLAImpl310


def _parent_init(instance, *args, **kwargs) -> None:
    del args
    instance.vllm_config = object()
    instance.num_kv_heads = kwargs.pop("num_kv_heads", 1)


def test_mla_impl_310_constructs_inside_validation_envelope() -> None:
    with (
        patch.object(AscendMLAImpl, "__init__", autospec=True, side_effect=_parent_init),
        patch.object(mla_310, "require_310p_mla_runtime") as require_runtime,
    ):
        impl = AscendMLAImpl310(num_kv_heads=1)

    require_runtime.assert_called_once_with(impl.vllm_config)


def test_mla_impl_310_rejects_multiple_latent_kv_heads() -> None:
    with (
        patch.object(AscendMLAImpl, "__init__", autospec=True, side_effect=_parent_init),
        patch.object(mla_310, "require_310p_mla_runtime"),
        pytest.raises(ValueError, match="one latent KV head"),
    ):
        AscendMLAImpl310(num_kv_heads=2)


def test_mla_impl_310_routes_cache_writes_to_validation_helper() -> None:
    impl = AscendMLAImpl310.__new__(AscendMLAImpl310)
    impl.kv_a_layernorm = SimpleNamespace(
        weight=torch.ones(4),
        variance_epsilon=1e-6,
    )
    projected = torch.zeros(1, 8)
    cos = torch.ones(1, 1, 1, 4)
    sin = torch.zeros_like(cos)
    slots = torch.tensor([0])
    caches = (torch.zeros(1, 1, 1, 4), torch.zeros(1, 1, 1, 4))
    expected = (torch.full((1, 1, 4), 2.0), torch.full((1, 1, 4), 3.0))

    with patch.object(mla_310, "write_logical_mla_cache", return_value=expected) as write_cache:
        actual = impl.exec_kv_prefill(projected, cos, sin, caches, slots)

    assert actual is expected
    write_cache.assert_called_once_with(
        projected,
        impl.kv_a_layernorm.weight,
        cos,
        sin,
        slots,
        caches[0],
        caches[1],
        epsilon=impl.kv_a_layernorm.variance_epsilon,
    )


def test_mla_impl_310_decode_returns_complete_logical_caches() -> None:
    impl = AscendMLAImpl310.__new__(AscendMLAImpl310)
    kpe_rows = torch.zeros(1, 1, 4)
    ckv_rows = torch.zeros(1, 1, 4)
    ckv_cache = torch.zeros(1, 1, 1, 4)
    kpe_cache = torch.zeros(1, 1, 1, 4)
    impl._write_cache = lambda *args: (kpe_rows, ckv_rows)

    actual = impl.exec_kv_decode(
        torch.zeros(1, 8),
        torch.ones(1, 1, 1, 4),
        torch.zeros(1, 1, 1, 4),
        (ckv_cache, kpe_cache),
        torch.tensor([0]),
    )

    assert actual[0] is kpe_cache
    assert actual[1] is ckv_cache


def test_mla_impl_310_rejects_batched_lengths() -> None:
    with pytest.raises(ValueError, match="one decode request"):
        AscendMLAImpl310._one_length([3, 4], "decode request")
