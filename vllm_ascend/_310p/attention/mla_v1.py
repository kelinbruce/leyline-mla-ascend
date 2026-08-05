# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Ascend 310P MLA backend boundary.

The cache contract and backend selection are intentionally landed before the
device implementation. This prevents MLA from silently using the incompatible
310P GQA backend while the FP16 operator path is being qualified on hardware.
"""

from typing import Any

from vllm_ascend.attention.mla_v1 import (
    AscendMLABackend,
    AscendMLAImpl,
    AscendMLAMetadataBuilder,
)


class AscendMLABackend310(AscendMLABackend):
    """310P MLA backend with a logical ND cKV/Kpe cache contract."""

    @staticmethod
    def get_builder_cls() -> type[AscendMLAMetadataBuilder]:
        return AscendMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["AscendMLAImpl310"]:
        return AscendMLAImpl310

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [128]


class AscendMLAImpl310(AscendMLAImpl):
    """Fail-closed placeholder for the hardware-qualified FP16 MLA kernels."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError(
            "Ascend 310P MLA cache layout is available, but the FP16 attention "
            "operator path has not been hardware-qualified. Run "
            "benchmarks/leyline/probe_310p_mla_ops.py on the pinned 310P "
            "environment before enabling model execution."
        )
