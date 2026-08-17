# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.leyline.common.evidence import (
    align_cache_off,
    file_evidence,
    parse_validation_request_id,
    request_id_matches,
    validation_request_id,
)
from benchmarks.leyline.scripts.finalize_validation import finalize_documents
from benchmarks.leyline.scripts.plan_divergence_capture import build_plan


def _environment() -> dict:
    return {
        "model": {"model": "/model", "hashes": {"weights": "abc"}},
        "imported_modules": {
            "vllm": {"module_file": "/src/vllm/vllm/__init__.py"},
            "vllm_ascend": {"module_file": "/src/ascend/vllm_ascend/__init__.py"},
        },
        "repositories": {
            "vllm": {"path": "/src/vllm", "commit": "v"},
            "vllm_ascend": {"path": "/src/ascend", "commit": "a"},
        },
        "runtime": {"tensor_parallel_size": 4, "block_size": 128},
        "topology": {"returncode": 0, "stdout": "TP4", "stderr": ""},
        "cann_installation": {"version_files": [{"path": "/cann/version.info"}]},
    }


def _arm(request_id: str, *, target: bool = True) -> dict:
    return {
        "request_id": request_id,
        "output_token_ids": [1, 2],
        "matches_completion_target": target,
    }


def _source(repetitions: int = 3) -> dict:
    runs = []
    for repetition in range(repetitions):
        full = _arm(f"full-{repetition}")
        honest = _arm(f"honest-{repetition}")
        patched = _arm(f"patched-{repetition}")
        vanilla = _arm(f"vanilla-{repetition}")
        leyline = _arm(f"leyline-{repetition}")
        leyline["record"] = {"request_id": f"record-{repetition}"}
        runs.append(
            {
                "id": "case",
                "arms": {
                    "full": full,
                    "honest_edited": honest,
                    "patched_disabled": patched,
                    "vanilla_apc": vanilla,
                    "leyline": leyline,
                },
                "counterfactuals": [],
                "gates": {
                    "admitted": True,
                    "leyline_accepted": True,
                    "leyline_execution_valid": True,
                    "full_matches": True,
                    "honest_edited_matches": True,
                },
                "pairwise": {
                    "full_leyline": {
                        "common_prefix_tokens": 2,
                        "first_divergence": None,
                        "first_token_agreement": True,
                    }
                },
            }
        )
    case = deepcopy(runs[0])
    case.update(
        {
            "family": "admissible",
            "claim_type": "admissible_target",
            "category": "admissible",
            "repetitions": runs,
            "stability": {"execution_stable": True},
        }
    )
    environment = _environment()
    return {
        "schema_version": 2,
        "corpus_id": "corpus",
        "workload_version": 2,
        "evaluation_contract": {"mode": "completion_target", "prompt_format": "raw"},
        "config": {"run_id": "source", "block_size": 128},
        "evidence_identity": {"evidence_schema_version": 1, "run_id": "source"},
        "checkpoint_identity": environment["model"],
        "environment": environment,
        "cases": [case],
    }


def _cache_off(source: dict, repetitions: int = 3) -> dict:
    report = deepcopy(source)
    report["config"]["run_id"] = "cache-off"
    report["evidence_identity"]["run_id"] = "cache-off"
    case = report["cases"][0]
    runs = []
    for repetition in range(repetitions):
        runs.append(
            {
                "id": "case",
                "arms": {"cache_off": _arm(f"cache-off-{repetition}")},
                "counterfactuals": [],
                "gates": {},
                "pairwise": {},
            }
        )
    case["repetitions"] = runs
    case["arms"] = runs[0]["arms"]
    return report


def _rollback(source: dict, *, passed: bool = True) -> dict:
    environment = source["environment"]
    run_id = "rollback"
    return {
        "schema_version": 1,
        "report_type": "leyline_rollback_validation",
        "passed": passed,
        "evidence_identity": {
            "evidence_schema_version": 1,
            "run_id": run_id,
            "checkpoint_identity": source["checkpoint_identity"],
            "imported_modules": environment["imported_modules"],
            "repositories": environment["repositories"],
            "runtime": environment["runtime"],
            "topology": environment["topology"],
        },
        "request_ids": {
            arm: validation_request_id(run_id, "case", f"rollback-{arm}", 0)
            for arm in ("honest", "record", "amortize")
        },
        "conditions": {"rollback_complete": passed},
        "sources": [
            {"name": "environment", "path": "/evidence/env.json", "size": 1, "sha256": "a" * 64}
        ],
        "injection": {"rank": 1, "layer": 2, "stage": "after_layer_write"},
        "record_metadata": {"recorded": True},
        "amortize_metadata": {"applied": False, "fallback_reason": "transform_failed"},
    }


def _cache_comparison(source: dict, *, passed: bool = True) -> dict:
    failures = 0 if passed else 1
    return {
        "schema_version": 2,
        "passed": passed,
        "captures": [
            {"request_id": "full-0", "layer": "layer.0", "rank": rank}
            for rank in range(4)
        ],
        "missing_deltas": [],
        "missing_layer_ranks": [],
        "aggregate": {
            "ckv_failed_captures": failures,
            "kpe_failed_captures": 0,
            "frequency_failed_captures": 0,
        },
    }


def test_finalizer_complete_pass_and_preserves_source() -> None:
    source = _source()
    before = deepcopy(source)
    report = finalize_documents(
        source,
        environment=source["environment"],
        cache_comparison=_cache_comparison(source),
        cache_off=_cache_off(source),
        rollback=_rollback(source),
    )
    assert report["qualification"]["passed"]
    assert report["qualification"]["determining_gate"] is None
    assert len(report["cache_off_evidence"]["case"]["runs"]) == 3
    assert source == before


@pytest.mark.parametrize(
    ("override", "gate", "classification"),
    [
        (("cache_comparison", None), "numerical", "missing_numerical_evidence"),
        (("cache_comparison", False), "numerical", "numerical_failure"),
        (("rollback", None), "rollback", "missing_rollback_evidence"),
        (("rollback", False), "rollback", "rollback_failure"),
        (("cache_off", None), "baselines", "missing_baseline_evidence"),
    ],
)
def test_finalizer_ordered_determining_gates(override, gate, classification) -> None:
    source = _source()
    arguments = {
        "environment": source["environment"],
        "cache_comparison": _cache_comparison(source),
        "cache_off": _cache_off(source),
        "rollback": _rollback(source),
    }
    key, value = override
    if value is False:
        arguments[key] = (
            _rollback(source, passed=False)
            if key == "rollback"
            else _cache_comparison(source, passed=False)
        )
    else:
        arguments[key] = value
    report = finalize_documents(source, **arguments)
    assert report["qualification"]["determining_gate"] == gate
    assert report["qualification"]["case_classifications"]["case"] == classification


def test_finalizer_rejects_mixed_identity_and_unknown_request() -> None:
    source = _source()
    cache_off = _cache_off(source)
    cache_off["checkpoint_identity"]["hashes"]["weights"] = "different"
    report = finalize_documents(
        source,
        environment=source["environment"],
        cache_comparison=_cache_comparison(source),
        cache_off=cache_off,
        rollback=_rollback(source),
        first_token_logits={"comparisons": [{"left_provenance": {"request_id": "unknown"}}]},
    )
    assert report["qualification"]["determining_gate"] == "environment"
    assert report["qualification"]["provenance_errors"]


def test_finalizer_environment_execution_target_and_suffix_classifications() -> None:
    source = _source()
    valid = {
        "environment": source["environment"],
        "cache_comparison": _cache_comparison(source),
        "cache_off": _cache_off(source),
        "rollback": _rollback(source),
    }

    invalid_environment = deepcopy(source)
    invalid_environment["environment"]["imported_modules"]["vllm"]["module_file"] = (
        "/other/vllm/__init__.py"
    )
    report = finalize_documents(
        invalid_environment,
        **{
            **valid,
            "environment": invalid_environment["environment"],
            "cache_off": _cache_off(invalid_environment),
            "rollback": _rollback(invalid_environment),
        },
    )
    assert report["qualification"]["determining_gate"] == "environment"
    assert report["qualification"]["case_classifications"]["case"] == "invalid_provenance"

    execution_failure = deepcopy(source)
    execution_failure["cases"][0]["gates"]["leyline_execution_valid"] = False
    report = finalize_documents(execution_failure, **valid)
    assert report["qualification"]["determining_gate"] == "execution"
    assert report["qualification"]["case_classifications"]["case"] == (
        "connector_execution_failure"
    )

    target_failure = deepcopy(source)
    target_failure["cases"][0]["gates"]["leyline_accepted"] = False
    report = finalize_documents(target_failure, **valid)
    assert report["qualification"]["determining_gate"] == "leyline_target"
    assert report["qualification"]["case_classifications"]["case"] == (
        "leyline_target_limitation"
    )

    suffix = deepcopy(source)
    suffix["cases"][0]["pairwise"]["full_leyline"]["common_prefix_tokens"] = 1
    report = finalize_documents(suffix, **valid)
    assert report["qualification"]["passed"]
    assert report["qualification"]["case_classifications"]["case"] == (
        "accepted_target_with_autoregressive_divergence"
    )


@pytest.mark.parametrize("field", ["missing_deltas", "missing_layer_ranks"])
def test_finalizer_rejects_incomplete_cache_comparison(field: str) -> None:
    source = _source()
    comparison = _cache_comparison(source)
    comparison[field] = [129] if field == "missing_deltas" else [{"layer": "layer.1", "rank": 2}]
    report = finalize_documents(
        source,
        environment=source["environment"],
        cache_comparison=comparison,
        cache_off=_cache_off(source),
        rollback=_rollback(source),
    )
    assert report["qualification"]["determining_gate"] == "numerical"


def test_cache_off_join_retains_repetitions_and_counterfactuals() -> None:
    source = _source()
    cache_off = _cache_off(source)
    for repetition, (source_run, off_run) in enumerate(
        zip(
            source["cases"][0]["repetitions"],
            cache_off["cases"][0]["repetitions"],
        )
    ):
        source_run["counterfactuals"] = [
            _arm(f"source-counterfactual-{repetition}-{variant}")
            for variant in range(2)
        ]
        off_run["counterfactuals"] = [
            _arm(f"cache-off-counterfactual-{repetition}-{variant}")
            for variant in range(2)
        ]
    aligned, errors = align_cache_off(source, cache_off)
    assert not errors
    assert len(aligned["case"]["runs"]) == 3
    assert [len(variants) for variants in aligned["case"]["counterfactuals"]] == [2, 2, 2]


def test_finalizer_reports_measured_baseline_failure() -> None:
    source = _source()
    source["cases"][0]["repetitions"][1]["arms"]["patched_disabled"][
        "matches_completion_target"
    ] = False
    report = finalize_documents(
        source,
        environment=source["environment"],
        cache_comparison=_cache_comparison(source),
        cache_off=_cache_off(source),
        rollback=_rollback(source),
    )
    assert report["qualification"]["determining_gate"] == "baselines"
    assert report["qualification"]["case_classifications"]["case"] == "baseline_failure"


def test_cache_off_missing_repetition_is_reported() -> None:
    source = _source()
    aligned, errors = align_cache_off(source, _cache_off(source, repetitions=2))
    assert len(aligned["case"]["runs"]) == 2
    assert "cache_off_repetition_count_mismatch:case" in errors


def test_unsupported_source_schema_fails() -> None:
    source = _source()
    source["schema_version"] = 1
    with pytest.raises(ValueError, match="schema-v2/v3"):
        finalize_documents(source)


def test_artifact_hash_and_structured_request_id(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps({"ok": True}))
    evidence = file_evidence(path)
    assert evidence["size"] == path.stat().st_size
    assert len(evidence["sha256"]) == 64
    request_id = validation_request_id("run", "case", "full", 2)
    assert request_id.startswith("lv3.run.case.canonical.2.full.")
    assert parse_validation_request_id(f"cmpl-{request_id}")["repetition"] == 2
    assert request_id_matches(f"cmpl-{request_id}-0", request_id)


def test_divergence_plan_is_bounded() -> None:
    source = _source(repetitions=1)
    run = source["cases"][0]["repetitions"][0]
    run["arms"]["full"]["output_token_ids"] = [1, 2, 3]
    run["pairwise"]["full_leyline"] = {
        "common_prefix_tokens": 2,
        "first_divergence": {"index": 2, "left": 3, "right": 4},
    }
    plan = build_plan(source, max_step=4)
    assert plan["capture_steps"] == [0, 2]
    assert plan["entries"][0]["source_prefix_token_ids"] == [1, 2]
