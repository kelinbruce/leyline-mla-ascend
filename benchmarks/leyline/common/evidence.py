# SPDX-License-Identifier: Apache-2.0

"""Shared immutable evidence helpers for Leyline qualification reports."""

from __future__ import annotations

import hashlib
import re
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


EVIDENCE_SCHEMA_VERSION = 1
FINAL_REPORT_SCHEMA_VERSION = 3
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_component(value: object) -> str:
    cleaned = _SAFE_COMPONENT.sub("-", str(value)).strip("-.")
    return cleaned[:96] or "unknown"


def ensure_run_id(config: dict[str, Any]) -> str:
    configured = config.get("run_id")
    if configured is not None:
        if not isinstance(configured, str) or not configured.strip():
            raise ValueError("run_id must be a non-empty string")
        normalized = safe_component(configured).replace(".", "-")
        config["run_id"] = normalized
        return normalized
    generated = f"leyline-{uuid.uuid4().hex}"
    config["run_id"] = generated
    return generated


def validation_request_id(
    run_id: str,
    case_id: str,
    arm: str,
    repetition: int,
    *,
    variant: str = "canonical",
) -> str:
    nonce = uuid.uuid4().hex
    return "lv3.{run}.{case}.{variant}.{rep}.{arm}.{nonce}".format(
        run=safe_component(run_id).replace(".", "-"),
        case=safe_component(case_id).replace(".", "-"),
        variant=safe_component(variant).replace(".", "-"),
        rep=int(repetition),
        arm=safe_component(arm).replace(".", "-"),
        nonce=nonce,
    )


def parse_validation_request_id(request_id: str) -> dict[str, Any] | None:
    request_id = request_id.removeprefix("cmpl-")
    parts = request_id.split(".")
    if len(parts) != 7 or parts[0] != "lv3":
        return None
    try:
        repetition = int(parts[4])
    except ValueError:
        return None
    return {
        "run_id": parts[1],
        "case_id": parts[2],
        "variant": parts[3],
        "repetition": repetition,
        "arm": parts[5],
        "nonce": parts[6],
    }


def request_id_matches(observed: str, expected: str) -> bool:
    return observed == expected or observed == f"cmpl-{expected}" or observed.startswith(
        f"cmpl-{expected}-"
    )


def evidence_key(
    *,
    case_id: str,
    repetition: int,
    arm: str,
    request_id: str,
    rank: int | None = None,
    decode_step: int | None = None,
    variant: str = "canonical",
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "variant": variant,
        "repetition": int(repetition),
        "arm": arm,
        "request_id": request_id,
        "rank": rank,
        "decode_step": decode_step,
    }


def file_evidence(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def stable_value(value: Any, *, excluded_keys: set[str] | None = None) -> Any:
    excluded = excluded_keys or set()
    if isinstance(value, dict):
        return {
            key: stable_value(item, excluded_keys=excluded)
            for key, item in sorted(value.items())
            if key not in excluded
        }
    if isinstance(value, list):
        return [stable_value(item, excluded_keys=excluded) for item in value]
    return value


def report_identity(report: dict[str, Any]) -> dict[str, Any]:
    config = report.get("config") or {}
    environment = report.get("environment") or {}
    evaluation = report.get("evaluation_contract") or {}

    # Repository dirtiness changes while a validation writes artifacts and is
    # therefore not an execution identity. Preserve only the resolved source
    # root and immutable commit result.
    repositories = {}
    for name, manifest in (environment.get("repositories") or {}).items():
        commit = (manifest or {}).get("commit")
        if isinstance(commit, dict):
            commit = {
                key: commit.get(key)
                for key in ("returncode", "stdout", "stderr")
                if key in commit
            }
        repositories[name] = {
            "path": (manifest or {}).get("path"),
            "commit": commit,
        }
    return {
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "run_id": ((report.get("evidence_identity") or {}).get("run_id") or config.get("run_id")),
        "corpus_id": report.get("corpus_id"),
        "workload_version": report.get("workload_version"),
        "evaluation_contract": {
            key: evaluation.get(key)
            for key in ("mode", "prompt_format", "score_evidence")
            if key in evaluation
        }
        or None,
        "checkpoint_identity": report.get("checkpoint_identity") or environment.get("model"),
        "imported_modules": environment.get("imported_modules"),
        "repositories": repositories or None,
        "runtime": environment.get("runtime_config") or environment.get("runtime"),
        "topology": environment.get("topology"),
        "block_size": config.get("block_size"),
    }


def checkpoint_identity(report: dict[str, Any]) -> dict[str, Any]:
    identity = report_identity(report)
    return {
        key: identity.get(key)
        for key in (
            "corpus_id",
            "workload_version",
            "evaluation_contract",
            "checkpoint_identity",
            "imported_modules",
            "repositories",
            "runtime",
            "topology",
            "block_size",
        )
    }


def identity_conflicts(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    allow_missing: bool = False,
    allow_cache_mode_difference: bool = False,
) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    left_identity = checkpoint_identity(left)
    right_identity = checkpoint_identity(right)
    for key in left_identity:
        left_value = stable_value(left_identity[key], excluded_keys={"hashing_seconds"})
        right_value = stable_value(right_identity[key], excluded_keys={"hashing_seconds"})
        if allow_cache_mode_difference and key == "runtime":
            ignored = {
                "enable_prefix_caching",
                "kv_connector",
                "kv_role",
                "kv_connector_module_path",
                "kv_load_failure_policy",
            }
            if isinstance(left_value, dict):
                left_value = {k: v for k, v in left_value.items() if k not in ignored}
            if isinstance(right_value, dict):
                right_value = {k: v for k, v in right_value.items() if k not in ignored}
        if allow_missing and (left_value is None or right_value is None):
            continue
        if left_value != right_value:
            conflicts.append({"field": key, "left": left_value, "right": right_value})
    return conflicts


def environment_blockers(environment: dict[str, Any] | None) -> list[str]:
    if not environment:
        return ["environment_manifest_missing"]
    blockers: list[str] = []
    repositories = environment.get("repositories") or {}
    modules = environment.get("imported_modules") or {}
    for module_name, repository_name in (
        ("vllm", "vllm"),
        ("vllm_ascend", "vllm_ascend"),
    ):
        module_path = (modules.get(module_name) or {}).get("module_file")
        repository_path = (repositories.get(repository_name) or {}).get("path")
        if not module_path or not repository_path:
            blockers.append(f"{module_name}_import_provenance_missing")
            continue
        try:
            Path(module_path).resolve().relative_to(Path(repository_path).resolve())
        except ValueError:
            blockers.append(f"{module_name}_import_outside_recorded_repository")
    if not ((environment.get("cann_installation") or {}).get("version_files")):
        blockers.append("cann_version_unresolved")
    return blockers


def iter_runs(report: dict[str, Any]) -> Iterable[tuple[dict[str, Any], int, dict[str, Any]]]:
    for case in report.get("cases", []):
        runs = case.get("repetitions") or [case]
        for repetition, run in enumerate(runs):
            yield case, repetition, run


def request_index(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for case, repetition, run in iter_runs(report):
        for arm, result in (run.get("arms") or {}).items():
            request_id = (result or {}).get("request_id")
            if request_id:
                index[request_id] = evidence_key(
                    case_id=case["id"], repetition=repetition, arm=arm, request_id=request_id
                )
            record = (result or {}).get("record") or {}
            record_id = record.get("request_id")
            if record_id:
                index[record_id] = evidence_key(
                    case_id=case["id"],
                    repetition=repetition,
                    arm=f"{arm}_record",
                    request_id=record_id,
                )
        for variant_index, result in enumerate(run.get("counterfactuals") or []):
            request_id = (result or {}).get("request_id")
            if request_id:
                index[request_id] = evidence_key(
                    case_id=case["id"],
                    repetition=repetition,
                    arm="full",
                    variant=f"counterfactual-{variant_index}",
                    request_id=request_id,
                )
    smoke_result = (report.get("smoke_gate") or {}).get("result") or {}
    smoke_case = str((report.get("smoke_gate") or {}).get("case_id") or "smoke")
    for arm, result in (smoke_result.get("arms") or {}).items():
        request_id = (result or {}).get("request_id")
        if request_id:
            index[request_id] = evidence_key(
                case_id=smoke_case,
                repetition=0,
                arm=f"smoke-{arm}",
                request_id=request_id,
            )
        record_id = ((result or {}).get("record") or {}).get("request_id")
        if record_id:
            index[record_id] = evidence_key(
                case_id=smoke_case,
                repetition=0,
                arm=f"smoke-{arm}-record",
                request_id=record_id,
            )
    return index


def auxiliary_request_ids(document: dict[str, Any]) -> set[str]:
    found: set[str] = set()

    def visit(value: Any, key: str | None = None) -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, child_key)
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif key == "request_id" and isinstance(value, str):
            found.add(value)

    visit(document)
    return found


def align_cache_off(
    source: dict[str, Any], cache_off: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    source_cases = {case["id"]: case for case in source.get("cases", [])}
    off_cases = {case["id"]: case for case in cache_off.get("cases", [])}
    aligned: dict[str, Any] = {}
    if set(source_cases) != set(off_cases):
        errors.append("cache_off_case_set_mismatch")
    for case_id in sorted(set(source_cases) & set(off_cases)):
        source_runs = source_cases[case_id].get("repetitions") or [source_cases[case_id]]
        off_runs = off_cases[case_id].get("repetitions") or [off_cases[case_id]]
        if len(source_runs) != len(off_runs):
            errors.append(f"cache_off_repetition_count_mismatch:{case_id}")
        runs = []
        for repetition, off_run in enumerate(off_runs):
            result = (off_run.get("arms") or {}).get("cache_off")
            if result is None:
                errors.append(f"cache_off_arm_missing:{case_id}:{repetition}")
                continue
            runs.append({"repetition": repetition, "result": deepcopy(result)})
        counterfactual_counts = [len(run.get("counterfactuals") or []) for run in off_runs]
        expected_counts = [len(run.get("counterfactuals") or []) for run in source_runs]
        if counterfactual_counts != expected_counts:
            errors.append(f"cache_off_counterfactual_mismatch:{case_id}")
        for repetition, (source_run, off_run) in enumerate(zip(source_runs, off_runs)):
            source_variants = [
                result.get("variant")
                for result in source_run.get("counterfactuals") or []
            ]
            off_variants = [
                result.get("variant")
                for result in off_run.get("counterfactuals") or []
            ]
            if source_variants != off_variants:
                errors.append(
                    f"cache_off_counterfactual_variant_mismatch:{case_id}:{repetition}"
                )
        aligned[case_id] = {
            "runs": runs,
            "counterfactuals": [deepcopy(run.get("counterfactuals") or []) for run in off_runs],
            "stable": bool(runs)
            and len(
                {
                    bool((item["result"] or {}).get("matches_completion_target"))
                    for item in runs
                }
            )
            == 1,
        }
        if source_cases[case_id].get("gates", {}).get("admitted") is True:
            if not runs or not all(
                item["result"].get("matches_completion_target") is True
                or item["result"].get("matches_oracle") is True
                for item in runs
            ):
                errors.append(f"cache_off_target_failed:{case_id}")
            if not all(
                result.get("matches_completion_target") is True
                or result.get("matches_oracle") is True
                for variants in aligned[case_id]["counterfactuals"]
                for result in variants
            ):
                errors.append(f"cache_off_counterfactual_target_failed:{case_id}")
    return aligned, errors


def source_baseline_errors(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for case in report.get("cases", []):
        if case.get("gates", {}).get("admitted") is not True:
            continue
        for repetition, run in enumerate(case.get("repetitions") or [case]):
            for arm in ("full", "honest_edited", "patched_disabled", "vanilla_apc"):
                result = (run.get("arms") or {}).get(arm)
                if result is None:
                    errors.append(f"baseline_arm_missing:{case['id']}:{repetition}:{arm}")
                elif not (
                    result.get("matches_completion_target") is True
                    or result.get("matches_oracle") is True
                ):
                    errors.append(f"baseline_target_failed:{case['id']}:{repetition}:{arm}")
    return errors


def retained_context_summary(report: dict[str, Any]) -> dict[str, Any]:
    entries = []
    informative_families: set[str] = set()
    for case in report.get("cases", []):
        if case.get("claim_type") != "retained_context_diagnostic":
            continue
        runs = case.get("repetitions") or [case]
        full = [bool(run.get("gates", {}).get("full_matches")) for run in runs]
        honest = [bool(run.get("gates", {}).get("honest_edited_matches")) for run in runs]
        different = [
            not bool(run.get("pairwise", {}).get("full_honest_edited", {}).get("first_token_agreement"))
            for run in runs
        ]
        informative = bool(runs) and all(full) and not any(honest) and all(different)
        if informative:
            informative_families.add(str(case.get("diagnostic_family") or case.get("family")))
        entries.append(
            {
                "case_id": case["id"],
                "family": case.get("diagnostic_family") or case.get("family"),
                "classification": "informative" if informative else "diagnostic_uninformative",
                "informative": informative,
                "full_target_stable": len(set(full)) == 1 and all(full),
                "honest_non_target_stable": len(set(honest)) == 1 and not any(honest),
                "first_token_distinct_stable": len(set(different)) == 1 and all(different),
                "leyline_target": [
                    run.get("gates", {}).get("leyline_matches") for run in runs
                ],
                "first_token_ids": [
                    {
                        arm: ((run.get("arms", {}).get(arm) or {}).get("output_token_ids") or [None])[0]
                        for arm in ("full", "honest_edited", "leyline")
                    }
                    for run in runs
                ],
                "leyline_full": [
                    run.get("pairwise", {}).get("full_leyline") for run in runs
                ],
                "leyline_honest": [
                    run.get("pairwise", {}).get("honest_edited_leyline") for run in runs
                ],
            }
        )
    informative_count = sum(item["informative"] for item in entries)
    return {
        "required_informative_cases": 4,
        "required_informative_families": 3,
        "informative_cases": informative_count,
        "informative_families": sorted(informative_families),
        "passed": informative_count >= 4 and len(informative_families) >= 3,
        "insufficient_coverage": informative_count < 4 or len(informative_families) < 3,
        "cases": entries,
    }


def classify_case(
    case: dict[str, Any],
    *,
    environment_state: str,
    numerical_state: str,
    rollback_state: str,
    baseline_state: str,
) -> str:
    if environment_state != "passed":
        return "invalid_provenance" if environment_state == "failed" else "missing_environment_evidence"
    if not case.get("gates", {}).get("leyline_execution_valid"):
        return "connector_execution_failure"
    if numerical_state != "passed":
        return "numerical_failure" if numerical_state == "failed" else "missing_numerical_evidence"
    if rollback_state != "passed":
        return "rollback_failure" if rollback_state == "failed" else "missing_rollback_evidence"
    if baseline_state != "passed":
        return "baseline_failure" if baseline_state == "failed" else "missing_baseline_evidence"
    if case.get("claim_type") == "retained_context_diagnostic":
        runs = case.get("repetitions") or [case]
        informative = bool(runs) and all(
            run.get("gates", {}).get("full_matches") is True
            and run.get("gates", {}).get("honest_edited_matches") is False
            and run.get("pairwise", {})
            .get("full_honest_edited", {})
            .get("first_token_agreement")
            is False
            for run in runs
        )
        if not informative:
            return "diagnostic_uninformative"
        return (
            "retained_context_evidence_observed"
            if all(run.get("gates", {}).get("leyline_matches") is True for run in runs)
            else "retained_context_evidence_not_observed"
        )
    if case.get("claim_type") == "negative_control":
        return "negative_control_observed"
    if not case.get("gates", {}).get("admitted"):
        return "invalid_workload_baseline"
    if not case.get("gates", {}).get("leyline_accepted"):
        return "leyline_target_limitation"
    pairwise = case.get("pairwise", {}).get("full_leyline", {})
    output = case.get("arms", {}).get("leyline", {}).get("output_token_ids") or []
    if pairwise.get("common_prefix_tokens", 0) < len(output):
        return "accepted_target_with_autoregressive_divergence"
    return "accepted_target_and_generation"


def gate_state(document: dict[str, Any] | None, *, key: str = "passed") -> str:
    if document is None:
        return "missing"
    return "passed" if document.get(key) is True else "failed"


def cache_comparison_errors(document: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if document.get("schema_version") != 2:
        errors.append("cache_comparison_schema_unsupported")
    captures = document.get("captures") or []
    if not captures:
        errors.append("cache_comparison_captures_missing")
    if document.get("missing_deltas"):
        errors.append("cache_comparison_deltas_incomplete")
    if document.get("missing_layer_ranks"):
        errors.append("cache_comparison_layer_ranks_incomplete")
    aggregate = document.get("aggregate") or {}
    for field in (
        "ckv_failed_captures",
        "kpe_failed_captures",
        "frequency_failed_captures",
    ):
        if field not in aggregate:
            errors.append(f"cache_comparison_aggregate_missing:{field}")
        elif int(aggregate[field]) != 0:
            errors.append(f"cache_comparison_aggregate_failed:{field}")
    if document.get("passed") is not True:
        errors.append("cache_comparison_report_failed")
    return errors


def rollback_report_errors(document: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if document.get("schema_version") != 1:
        errors.append("rollback_schema_unsupported")
    if document.get("report_type") != "leyline_rollback_validation":
        errors.append("rollback_report_type_invalid")
    conditions = document.get("conditions") or {}
    if not conditions or not all(value is True for value in conditions.values()):
        errors.append("rollback_conditions_incomplete")
    request_ids = document.get("request_ids") or {}
    evidence_identity = document.get("evidence_identity") or {}
    run_id = evidence_identity.get("run_id")
    if evidence_identity.get("evidence_schema_version") != EVIDENCE_SCHEMA_VERSION:
        errors.append("rollback_run_identity_invalid")
    for arm in ("honest", "record", "amortize"):
        parsed = parse_validation_request_id(str(request_ids.get(arm) or ""))
        if parsed is None:
            errors.append(f"rollback_request_invalid:{arm}")
        elif parsed["run_id"] != run_id:
            errors.append(f"rollback_request_run_mismatch:{arm}")
    sources = document.get("sources") or []
    if not sources:
        errors.append("rollback_sources_missing")
    for source in sources:
        if not source.get("path") or not source.get("sha256") or source.get("size") is None:
            errors.append("rollback_source_manifest_incomplete")
            break
    for field in ("injection", "record_metadata", "amortize_metadata"):
        if not document.get(field):
            errors.append(f"rollback_evidence_missing:{field}")
    if document.get("passed") is not True:
        errors.append("rollback_report_failed")
    return errors


def first_determining_gate(gates: dict[str, str]) -> str | None:
    for name in ("environment", "execution", "numerical", "rollback", "baselines", "leyline_target"):
        if gates.get(name) != "passed":
            return name
    return None
