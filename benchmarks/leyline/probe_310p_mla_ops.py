#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Probe 310P primitives required by the correctness-first FP16 MLA path.

This script is intentionally independent of model startup so unsupported
operators can be reported before the experimental backend is instantiated.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch_npu

from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

TARGET_BASELINE = "05e095a202bdcfef4da61168eae34bfd3b99da13"


def _version(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    text = (result.stdout or result.stderr).strip()
    return text or None


def _git(*args: str) -> str | None:
    return _version(["git", *args])


def _read_cann_version() -> dict[str, str] | None:
    candidates = []
    if ascend_home := os.getenv("ASCEND_HOME_PATH"):
        candidates.extend(
            [
                Path(ascend_home) / "version.cfg",
                Path(ascend_home) / "compiler" / "version.info",
            ]
        )
    candidates.extend(
        [
            Path("/usr/local/Ascend/ascend-toolkit/latest/version.cfg"),
            Path("/usr/local/Ascend/ascend-toolkit/latest/compiler/version.info"),
        ]
    )
    for candidate in candidates:
        try:
            return {"path": str(candidate), "contents": candidate.read_text().strip()}
        except OSError:
            continue
    return None


def _run_probe(name: str, operation: Callable[[], None]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        operation()
        torch.npu.synchronize()
    except Exception as exc:  # device capability probe must preserve the exact error
        return {
            "name": name,
            "passed": False,
            "duration_ms": (time.perf_counter() - started) * 1000,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return {
        "name": name,
        "passed": True,
        "duration_ms": (time.perf_counter() - started) * 1000,
    }


def _basic_fp16() -> None:
    x = torch.randn(32, 64, dtype=torch.float16, device="npu")
    y = torch.randn(64, 32, dtype=torch.float16, device="npu")
    result = x @ y
    if result.shape != (32, 32):
        raise AssertionError(f"unexpected matmul shape: {result.shape}")


def _fp32_trigonometry() -> None:
    angles = torch.linspace(-1024.0, 1024.0, 4096, dtype=torch.float32, device="npu")
    result = angles.cos().square() + angles.sin().square()
    torch.testing.assert_close(result.cpu(), torch.ones_like(result).cpu(), atol=2e-5, rtol=2e-5)


def _cache_gather_scatter() -> None:
    ckv = torch.randn(4, 128, 1, 512, dtype=torch.float16, device="npu")
    source = torch.tensor([0, 127, 128, 383], dtype=torch.long, device="npu")
    destination = torch.tensor([384, 385, 386, 387], dtype=torch.long, device="npu")
    flat = ckv.reshape(-1, 1, 512)
    expected = flat.index_select(0, source).clone()
    flat.index_copy_(0, destination, expected)
    torch.testing.assert_close(flat.index_select(0, destination).cpu(), expected.cpu(), atol=0, rtol=0)


def _leyline_rotation() -> None:
    from vllm_ascend.ops.leyline_mla import transform_mla_cache

    ckv = torch.randn(4, 128, 1, 512, dtype=torch.float16, device="npu")
    kpe = torch.randn(4, 128, 1, 64, dtype=torch.float16, device="npu")
    inv_freq = torch.linspace(1.0, 0.001, 32, dtype=torch.float32, device="npu")
    transform_mla_cache(
        ckv,
        kpe,
        [0, 127, 128, 383],
        [384, 385, 386, 387],
        [0, 127, 128, 1024],
        [0, 126, 1, 0],
        inv_freq,
    )


def _operator_presence() -> None:
    required = [
        "npu_kv_rmsnorm_rope_cache",
        "npu_interleave_rope",
        "npu_fused_infer_attention_score",
        "npu_fused_infer_attention_score_v2",
    ]
    missing = [name for name in required if not hasattr(torch_npu, name)]
    if missing:
        raise RuntimeError("missing torch_npu operators: " + ", ".join(missing))


def _mla_cache_write() -> None:
    num_tokens = 2
    ckv = torch.zeros(2, 128, 1, 512, dtype=torch.float16, device="npu")
    kpe = torch.zeros(2, 128, 1, 64, dtype=torch.float16, device="npu")
    kv = torch.randn(num_tokens, 1, 1, 576, dtype=torch.float16, device="npu")
    gamma = torch.ones(512, dtype=torch.float16, device="npu")
    cos = torch.ones(num_tokens, 1, 1, 64, dtype=torch.float16, device="npu")
    sin = torch.zeros_like(cos)
    slots = torch.tensor([0, 129], dtype=torch.int64, device="npu")
    torch_npu.npu_kv_rmsnorm_rope_cache(
        kv,
        gamma,
        cos,
        sin,
        slots,
        kpe,
        ckv,
        epsilon=1e-6,
        cache_mode="PA",
        is_output_kv=True,
    )
    if not torch.count_nonzero(ckv).cpu().item():
        raise AssertionError("npu_kv_rmsnorm_rope_cache did not update cKV")


def _mla_prefill_attention() -> None:
    num_tokens = 8
    num_heads = 4
    q_nope = torch.randn(num_tokens, num_heads, 128, dtype=torch.float16, device="npu")
    k_nope = torch.randn_like(q_nope)
    value = torch.randn_like(q_nope)
    q_pe = torch.randn(num_tokens, num_heads, 64, dtype=torch.float16, device="npu")
    k_pe = torch.randn_like(q_pe)
    output, _ = torch_npu.npu_fused_infer_attention_score(
        q_nope,
        k_nope,
        value,
        query_rope=q_pe,
        key_rope=k_pe,
        num_heads=num_heads,
        num_key_value_heads=num_heads,
        input_layout="TND",
        atten_mask=None,
        sparse_mode=0,
        scale=128**-0.5,
        softmax_lse_flag=True,
        actual_seq_lengths=[num_tokens],
        actual_seq_lengths_kv=[num_tokens],
    )
    if output.shape != value.shape:
        raise AssertionError(f"unexpected prefill output shape: {output.shape}")


def _mla_paged_decode_attention() -> None:
    num_heads = 4
    q_nope = torch.randn(1, num_heads, 1, 512, dtype=torch.float16, device="npu")
    q_pe = torch.randn(1, num_heads, 1, 64, dtype=torch.float16, device="npu")
    ckv = torch.randn(2, 1, 128, 512, dtype=torch.float16, device="npu")
    kpe = torch.randn(2, 1, 128, 64, dtype=torch.float16, device="npu")
    output, _ = torch_npu.npu_fused_infer_attention_score_v2(
        q_nope,
        ckv,
        ckv,
        query_rope=q_pe,
        key_rope=kpe,
        num_query_heads=num_heads,
        num_key_value_heads=1,
        input_layout="BNSD_NBSD",
        atten_mask=None,
        sparse_mode=0,
        softmax_scale=192**-0.5,
        block_table=torch.tensor([[0]], dtype=torch.int32, device="npu"),
        block_size=128,
        actual_seq_qlen=None,
        actual_seq_kvlen=[8],
    )
    if output.numel() == 0:
        raise AssertionError("paged decode returned an empty tensor")


def _mla_chunked_cache_load() -> None:
    ckv = torch.randn(2, 128, 1, 512, dtype=torch.float16, device="npu")
    kpe = torch.randn(2, 128, 1, 64, dtype=torch.float16, device="npu")
    key = torch.empty(8, 1, 512, dtype=torch.float16, device="npu")
    value = torch.empty(8, 1, 64, dtype=torch.float16, device="npu")
    torch_npu.atb.npu_paged_cache_load(
        ckv,
        kpe,
        torch.tensor([[0]], dtype=torch.int32, device="npu"),
        torch.tensor([8], dtype=torch.int32, device="npu"),
        seq_starts=torch.tensor([0], dtype=torch.int32, device="npu"),
        key=key,
        value=value,
    )
    if key.numel() == 0 or value.numel() == 0:
        raise AssertionError("paged cache load returned an empty tensor")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_report(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.npu.is_available():
        raise RuntimeError("an available Ascend NPU is required")
    if get_ascend_device_type() != AscendDeviceType._310P:
        raise RuntimeError("this probe must run on an Ascend 310P device")
    if args.tensor_parallel_size != 4:
        raise ValueError("the initial 310P Leyline qualification requires TP4")
    device_name = torch.npu.get_device_name(0)
    probes = [
        _run_probe("required_operator_presence", _operator_presence),
        _run_probe("basic_fp16_matmul", _basic_fp16),
        _run_probe("fp32_cos_sin", _fp32_trigonometry),
        _run_probe("fp16_cache_gather_scatter", _cache_gather_scatter),
        _run_probe("fp16_leyline_rotation", _leyline_rotation),
        _run_probe("fp16_mla_cache_write", _mla_cache_write),
        _run_probe("fp16_mla_prefill_attention", _mla_prefill_attention),
        _run_probe("fp16_mla_paged_decode", _mla_paged_decode_attention),
        _run_probe("fp16_mla_chunked_cache_load", _mla_chunked_cache_load),
    ]
    return {
        "schema_version": 1,
        "status": "passed" if all(probe["passed"] for probe in probes) else "failed",
        "vllm_ascend_baseline": TARGET_BASELINE,
        "checkout_head": _git("rev-parse", "HEAD"),
        "checkout_status": _git("status", "--porcelain"),
        "vllm_checkout_head": _version(
            ["git", "-C", str(args.vllm_repo), "rev-parse", "HEAD"]
        ),
        "device": device_name,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_npu": getattr(torch_npu, "__version__", None),
        "vllm": _package_version("vllm"),
        "vllm_ascend": _package_version("vllm-ascend"),
        "soc_version": os.getenv("SOC_VERSION"),
        "ascend_home_path": os.getenv("ASCEND_HOME_PATH"),
        "cann": _read_cann_version(),
        "npu_smi": _version(["npu-smi", "info"]),
        "model": {
            "name": args.model,
            "revision": args.model_revision,
            "tokenizer_revision": args.tokenizer_revision,
            "dtype": "float16",
        },
        "topology": {
            "rank": args.rank,
            "tensor_parallel_size": args.tensor_parallel_size,
            "decode_context_parallel_size": 1,
            "prefill_context_parallel_size": 1,
        },
        "cache": {
            "block_size": 128,
            "dtype": "float16",
            "logical_ckv_shape": ["num_blocks", 128, 1, 512],
            "logical_kpe_shape": ["num_blocks", 128, 1, 64],
        },
        "probes": probes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--vllm-repo", type=Path, default=Path("/vllm-workspace/vllm"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = collect_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
