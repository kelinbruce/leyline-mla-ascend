# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_ascend._310p.attention.mla_v1 import (
    AscendMLABackend310,
    AscendMLAImpl310,
)
from vllm_ascend.attention.mla_v1 import AscendMLAMetadataBuilder


def test_mla_backend_310_exposes_logical_nd_cache() -> None:
    assert AscendMLABackend310.get_kv_cache_shape(8, 128, 1, 576) == (8, 128, 1, 576)
    assert AscendMLABackend310.get_supported_kernel_block_sizes() == [128]
    assert AscendMLABackend310.get_builder_cls() is AscendMLAMetadataBuilder
    assert AscendMLABackend310.get_impl_cls() is AscendMLAImpl310


def test_mla_impl_310_fails_before_hardware_qualification() -> None:
    with pytest.raises(RuntimeError, match="not been hardware-qualified"):
        AscendMLAImpl310()
