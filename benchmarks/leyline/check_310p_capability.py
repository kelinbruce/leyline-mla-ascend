#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Report whether this checkout can safely launch DeepSeek MLA Leyline on 310P."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

TARGET_BASELINE = "05e095a202bdcfef4da61168eae34bfd3b99da13"
EXPECTED_VLLM_TAG = "v0.23.0"
_MLA_GUARD = "MLAAttention is not supported for 310P."
_KV_TRANSFER_GUARD = "KV cache transfer is not supported for 310P."
_MLA_BACKEND = "vllm_ascend._310p.attention.mla_v1.AscendMLABackend310"
_UNQUALIFIED_IMPL_MARKER = "not been hardware-qualified"


def _git(repo: Path, *args: str) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"returncode": None, "error": str(exc)}
    return {
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def inspect_310p_capability(repo: Path, runtime_config: dict[str, Any] | None = None) -> dict[str, Any]:
    runner = repo / "vllm_ascend" / "_310p" / "model_runner_310p.py"
    platform_source = repo / "vllm_ascend" / "platform.py"
    backend_source = repo / "vllm_ascend" / "_310p" / "attention" / "mla_v1.py"
    qualification_path = repo / "benchmarks" / "leyline" / "310p_qualification.json"
    try:
        runner_text = runner.read_text()
    except OSError as exc:
        runner_text = ""
        source_error = str(exc)
    else:
        source_error = None
    try:
        platform_text = platform_source.read_text()
        backend_text = backend_source.read_text()
    except OSError as exc:
        platform_text = ""
        backend_text = ""
        backend_source_error = str(exc)
    else:
        backend_source_error = None

    head = _git(repo, "rev-parse", "HEAD")
    baseline = _git(repo, "merge-base", "--is-ancestor", TARGET_BASELINE, "HEAD")
    mla_blocked = _MLA_GUARD in runner_text
    kv_transfer_blocked = _KV_TRANSFER_GUARD in runner_text
    mla_backend_registered = _MLA_BACKEND in platform_text
    mla_operator_path_implemented = bool(
        backend_text and _UNQUALIFIED_IMPL_MARKER not in backend_text
    )
    blockers = []
    if source_error:
        blockers.append("310p_runner_source_unavailable")
    if mla_blocked:
        blockers.append("310p_runner_rejects_mla")
    if kv_transfer_blocked:
        blockers.append("310p_runner_rejects_kv_transfer")
    if not mla_backend_registered:
        blockers.append("310p_mla_backend_not_registered")
    if not mla_operator_path_implemented:
        blockers.append("310p_mla_operator_path_unimplemented")

    try:
        qualification = json.loads(qualification_path.read_text())
    except (OSError, json.JSONDecodeError):
        qualification = None
    hardware_qualified = bool(
        qualification
        and qualification.get("status") == "passed"
        and qualification.get("vllm_ascend_baseline") == TARGET_BASELINE
        and qualification.get("checkout_head") == head.get("stdout")
        and qualification.get("qualified_ranks") == [0, 1, 2, 3]
        and all(qualification.get("gates", {}).values())
    )
    if not hardware_qualified:
        blockers.append("310p_hardware_qualification_missing")

    runtime_checks: dict[str, Any] = {}
    if runtime_config is None:
        blockers.append("runtime_configuration_not_provided")
    else:
        dtype = runtime_config.get("dtype")
        runtime_checks["fp16_configured"] = dtype in {"float16", "half"}
        runtime_checks["mla_model_expected"] = "deepseek" in str(runtime_config.get("model", "")).lower()
        runtime_checks["leyline_connector_configured"] = runtime_config.get("kv_connector") == "LeylineConnector"
        if not runtime_checks["fp16_configured"]:
            blockers.append("310p_requires_fp16_validation_configuration")
        if not runtime_checks["mla_model_expected"]:
            blockers.append("deepseek_mla_model_not_configured")
        if not runtime_checks["leyline_connector_configured"]:
            blockers.append("leyline_connector_not_configured")

    return {
        "schema_version": 1,
        "target": {
            "soc": "Ascend 310P",
            "vllm_ascend_baseline": TARGET_BASELINE,
            "vllm_tag": EXPECTED_VLLM_TAG,
        },
        "checkout": {
            "path": str(repo.resolve()),
            "head": head,
            "contains_target_baseline": baseline.get("returncode") == 0,
        },
        "source_checks": {
            "runner": str(runner),
            "platform": str(platform_source),
            "mla_backend": str(backend_source),
            "mla_guard_present": mla_blocked,
            "kv_transfer_guard_present": kv_transfer_blocked,
            "error": source_error,
            "mla_backend_registered": mla_backend_registered,
            "mla_operator_path_implemented": mla_operator_path_implemented,
            "mla_backend_error": backend_source_error,
        },
        "hardware_qualification": {
            "path": str(qualification_path),
            "present_and_passed": hardware_qualified,
        },
        "runtime_checks": runtime_checks,
        "safe_to_launch_deepseek_mla_leyline": not blockers,
        "blockers": blockers,
        "required_work": [
            "implement and validate a 310P MLA attention/cache backend",
            "integrate the KV Connector lifecycle with the 310P model runner",
            "validate Leyline cKV copy and Kpe rotation in FP16 on 310P hardware",
            "run rollback, semantic, and performance gates before enabling the runtime",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-supported", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_config = json.loads(args.runtime_config.read_text()) if args.runtime_config else None
    report = inspect_310p_capability(args.repo, runtime_config)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    if args.require_supported and not report["safe_to_launch_deepseek_mla_leyline"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
