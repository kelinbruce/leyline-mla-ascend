# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch

from benchmarks.leyline.check_310p_capability import inspect_310p_capability
from benchmarks.leyline.qualify_310p import build_qualification
from benchmarks.leyline.run_310p_service_validation import (
    build_server_command,
    merged_report_passed,
    runner_config,
)
from benchmarks.leyline.run_validation import (
    REFERENCE_PREFIX,
    STRUCTURED_JSON,
    build_prompt_plan,
    evaluate_case_results,
    leyline_execution_evidence,
    request_completion,
)


class _CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        encoded = [ord(character) for character in text]
        return [1, *encoded] if add_special_tokens else encoded

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize
        assert add_generation_prompt
        return f"<bos><user>{messages[0]['content']}</user><assistant>"


def _case() -> dict:
    return {
        "id": "case",
        "category": "admissible",
        "prefix": "prefix:",
        "removed": "remove-me",
        "surviving": "survive",
        "query": "?",
        "oracle": {"answer": "yes"},
    }


def _result(
    token_ids: list[int] | None,
    structured: dict | None = None,
    *,
    leyline: dict | None = None,
) -> dict:
    return {
        "output_token_ids": token_ids,
        "structured": structured,
        "kv_transfer_params": {"leyline": leyline} if leyline is not None else None,
    }


def _applied_leyline() -> dict:
    return {
        "applied": True,
        "transformed_tokens": 128,
        "fallback_reason": None,
    }


def test_raw_prompt_plan_is_exactly_one_token_deletion() -> None:
    tokenizer = _CharacterTokenizer()
    plan = build_prompt_plan(tokenizer, _case())

    assert plan.full[: plan.delete_start] + plan.full[plan.delete_end :] == plan.edited
    assert plan.delete_end - plan.delete_start == len("remove-me")
    assert plan.full[0] == 1


def test_chat_template_preserves_generation_prompt_and_exact_deletion() -> None:
    tokenizer = _CharacterTokenizer()
    plan = build_prompt_plan(tokenizer, _case(), prompt_format="chat_template")
    decoded_full = "".join(chr(token) for token in plan.full)
    decoded_edited = "".join(chr(token) for token in plan.edited)

    assert plan.full[: plan.delete_start] + plan.full[plan.delete_end :] == plan.edited
    assert decoded_full == "<bos><user>prefix:remove-mesurvive?</user><assistant>"
    assert decoded_edited == "<bos><user>prefix:survive?</user><assistant>"


def test_completion_request_preserves_ids_and_returns_generated_ids() -> None:
    captured = {}

    def _urlopen(request, timeout):
        assert timeout == 600
        captured.update(json.loads(request.data))
        return io.BytesIO(
            json.dumps(
                {
                    "choices": [{"text": "ok", "token_ids": [7, 8]}],
                    "usage": {"completion_tokens": 2},
                }
            ).encode()
        )

    with patch("benchmarks.leyline.run_validation.urlopen", _urlopen):
        result = request_completion(
            "http://localhost:8000",
            "model",
            [1, 2, 3],
            api_key=None,
            max_tokens=2,
        )

    assert captured["prompt"] == [1, 2, 3]
    assert captured["add_special_tokens"] is False
    assert captured["return_token_ids"] is True
    assert result["output_token_ids"] == [7, 8]


def test_reference_prefix_uses_full_output_as_mechanism_reference() -> None:
    arms = {
        "full": _result([10, 11]),
        "honest_edited": _result([10, 99]),
        "leyline": _result([10, 12], leyline=_applied_leyline()),
    }
    counterfactuals = [_result([10, 13])]

    evaluation = evaluate_case_results(
        _case(), arms, counterfactuals, mode=REFERENCE_PREFIX, reference_tokens=1
    )

    assert evaluation["gates"] == {
        "full_matches": True,
        "honest_edited_matches": True,
        "counterfactuals_match": True,
        "admitted": True,
        "semantic_admitted": False,
        "reference_admitted": True,
        "leyline_matches": True,
        "leyline_execution_valid": True,
        "leyline_accepted": True,
    }
    assert not evaluation["semantic_oracle_validated"]
    assert arms["leyline"]["matches_reference"]


def test_structured_json_requires_the_declared_oracle() -> None:
    oracle = _case()["oracle"]
    arms = {
        "full": _result([10], oracle),
        "honest_edited": _result([10], oracle),
        "leyline": _result([10], {"answer": "no"}, leyline=_applied_leyline()),
    }

    evaluation = evaluate_case_results(
        _case(), arms, [], mode=STRUCTURED_JSON, reference_tokens=1
    )

    assert evaluation["gates"]["admitted"]
    assert evaluation["gates"]["semantic_admitted"]
    assert not evaluation["gates"]["reference_admitted"]
    assert not evaluation["gates"]["leyline_matches"]
    assert evaluation["gates"]["leyline_execution_valid"]
    assert not evaluation["gates"]["leyline_accepted"]
    assert evaluation["semantic_oracle_validated"]


def test_leyline_execution_requires_transform_and_generated_ids() -> None:
    assert leyline_execution_evidence(
        _result([10], leyline=_applied_leyline())
    )["valid"]

    for result in (
        _result(None, leyline=_applied_leyline()),
        _result([10], leyline={**_applied_leyline(), "applied": False}),
        _result([10], leyline={**_applied_leyline(), "transformed_tokens": 0}),
        _result([10], leyline={**_applied_leyline(), "fallback_reason": "transform_failed"}),
    ):
        assert not leyline_execution_evidence(result)["valid"]


def test_leyline_fallback_cannot_pass_merged_arm_gate() -> None:
    arms = {
        "full": _result([10]),
        "honest_edited": _result([10]),
        "leyline": _result(
            [10],
            leyline={
                "applied": False,
                "transformed_tokens": 0,
                "fallback_reason": "transform_failed",
            },
        ),
    }

    evaluation = evaluate_case_results(
        _case(), arms, [], mode=REFERENCE_PREFIX, reference_tokens=1
    )

    assert evaluation["gates"]["leyline_matches"]
    assert not evaluation["gates"]["leyline_execution_valid"]
    assert not evaluation["gates"]["leyline_accepted"]


def test_310p_preflight_reports_upstream_guards(tmp_path: Path) -> None:
    runner = tmp_path / "vllm_ascend" / "_310p" / "model_runner_310p.py"
    runner.parent.mkdir(parents=True)
    runner.write_text(
        'raise ValueError("KV cache transfer is not supported for 310P.")\n'
        'raise ValueError("MLAAttention is not supported for 310P.")\n'
    )

    report = inspect_310p_capability(
        tmp_path,
        {
            "dtype": "float16",
            "model": "deepseek-ai/DeepSeek-V2-Lite",
            "kv_connector": "LeylineConnector",
        },
    )

    assert not report["safe_to_launch_deepseek_mla_leyline"]
    assert report["blockers"] == [
        "310p_runner_rejects_mla",
        "310p_runner_rejects_kv_transfer",
        "310p_mla_backend_not_registered",
        "310p_mla_operator_path_unimplemented",
        "310p_hardware_qualification_missing",
    ]


def _probe(rank: int) -> dict:
    return {
        "status": "passed",
        "probe_profile": "unfused_validation_mla_v1",
        "vllm_ascend_baseline": "05e095a202bdcfef4da61168eae34bfd3b99da13",
        "checkout_head": "implementation-head",
        "checkout_status": "",
        "vllm_checkout_head": "vllm-head",
        "model": {
            "name": "deepseek-ai/DeepSeek-V2-Lite",
            "revision": "model-commit",
            "tokenizer_revision": "tokenizer-commit",
            "dtype": "float16",
        },
        "topology": {
            "rank": rank,
            "tensor_parallel_size": 4,
            "decode_context_parallel_size": 1,
            "prefill_context_parallel_size": 1,
        },
        "cache": {"block_size": 128, "dtype": "float16"},
        "torch": "test",
        "torch_npu": "test",
        "cann": {"contents": "test"},
    }


def test_310p_qualification_requires_all_gates_and_tp_ranks() -> None:
    e2e = {
        "cases": [
            {
                "category": "admissible",
                "gates": {
                    "admitted": True,
                    "leyline_matches": True,
                    "leyline_execution_valid": True,
                    "leyline_accepted": True,
                },
            }
        ]
    }
    report = build_qualification(
        [_probe(rank) for rank in range(4)],
        {"passed": True},
        e2e,
        {"passed": True},
    )

    assert report["status"] == "passed"
    assert report["qualified_ranks"] == [0, 1, 2, 3]
    assert all(report["gates"].values())

    incomplete = build_qualification(
        [_probe(rank) for rank in range(3)],
        {"passed": True},
        e2e,
        {"passed": True},
    )
    assert incomplete["status"] == "failed"
    assert "operator_probes_incomplete_tp4" in incomplete["blockers"]

    legacy = [_probe(rank) for rank in range(4)]
    for probe in legacy:
        probe.pop("probe_profile")
    legacy_report = build_qualification(
        legacy,
        {"passed": True},
        e2e,
        {"passed": True},
    )
    assert legacy_report["status"] == "failed"
    assert "unfused_validation_probe_profile_missing" in legacy_report["blockers"]


def _service_args(**overrides):
    values = {
        "vllm_executable": "vllm",
        "model": "/models/DeepSeek-V2-Lite",
        "served_model_name": "deepseek-v2-lite",
        "port": 8000,
        "max_model_len": 1024,
        "max_tokens": 4,
        "model_revision": "local",
        "tokenizer": "/models/DeepSeek-V2-Lite",
        "tokenizer_revision": "local",
    }
    values.update(overrides)
    return type("Args", (), values)()


def test_310p_service_command_has_restricted_runtime_and_optional_connector() -> None:
    args = _service_args()
    base = build_server_command(args, connector=False)
    connected = build_server_command(args, connector=True)

    assert base[:3] == ["vllm", "serve", "/models/DeepSeek-V2-Lite"]
    assert base[base.index("--tensor-parallel-size") + 1] == "4"
    assert base[base.index("--dtype") + 1] == "float16"
    assert base[base.index("--max-num-seqs") + 1] == "1"
    assert "--enforce-eager" in base
    assert "--revision" not in base
    assert "--kv-transfer-config" not in base
    assert "--kv-transfer-config" in connected


def test_310p_service_configs_split_cold_and_leyline_arms() -> None:
    args = _service_args()
    assert runner_config(args, connector=False)["arms"] == {
        "cache_off": "http://127.0.0.1:8000"
    }
    assert set(runner_config(args, connector=True)["arms"]) == {
        "full",
        "honest_edited",
        "patched_disabled",
        "vanilla_apc",
        "leyline",
    }


def test_310p_merged_report_requires_cold_output_and_leyline_acceptance() -> None:
    case = {
        "category": "admissible",
        "gates": {"admitted": True, "leyline_accepted": True},
        "arms": {"cache_off": {"output_token_ids": [10]}},
    }
    assert merged_report_passed({"cases": [case]})
    case["gates"]["leyline_accepted"] = False
    assert not merged_report_passed({"cases": [case]})
