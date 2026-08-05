# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch

from benchmarks.leyline.check_310p_capability import inspect_310p_capability
from benchmarks.leyline.qualify_310p import build_qualification
from benchmarks.leyline.run_validation import (
    REFERENCE_PREFIX,
    STRUCTURED_JSON,
    build_prompt_plan,
    evaluate_case_results,
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


def _result(token_ids: list[int], structured: dict | None = None) -> dict:
    return {"output_token_ids": token_ids, "structured": structured}


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
        "leyline": _result([10, 12]),
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
    }
    assert not evaluation["semantic_oracle_validated"]
    assert arms["leyline"]["matches_reference"]


def test_structured_json_requires_the_declared_oracle() -> None:
    oracle = _case()["oracle"]
    arms = {
        "full": _result([10], oracle),
        "honest_edited": _result([10], oracle),
        "leyline": _result([10], {"answer": "no"}),
    }

    evaluation = evaluate_case_results(
        _case(), arms, [], mode=STRUCTURED_JSON, reference_tokens=1
    )

    assert evaluation["gates"]["admitted"]
    assert evaluation["gates"]["semantic_admitted"]
    assert not evaluation["gates"]["reference_admitted"]
    assert not evaluation["gates"]["leyline_matches"]
    assert evaluation["semantic_oracle_validated"]


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
                "gates": {"admitted": True, "leyline_matches": True},
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
