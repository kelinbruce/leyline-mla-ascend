# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.leyline.collect_environment import artifact_manifest
from benchmarks.leyline.merge_reports import merge_documents
from benchmarks.leyline.run_validation import (
    COMPLETION_TARGET,
    REFERENCE_PREFIX,
    STRUCTURED_JSON,
    build_prompt_plan,
    evaluate_case_results,
    evaluation_config,
    performance_gate,
    request_completion,
    run_preflight,
    validate_workload_corpus,
)


class CharacterTokenizer:
    chat_template = "test"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        values = [ord(character) for character in text]
        return [1, *values] if add_special_tokens else values

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize
        assert add_generation_prompt
        return f"<user>{messages[0]['content']}</user><assistant>"


class PieceTokenizer(CharacterTokenizer):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        pieces = text.replace(" neutral", "§")
        values = [ord(character) for character in pieces]
        return [1, *values] if add_special_tokens else values


def case(category: str = "admissible") -> dict:
    return {
        "id": "case",
        "category": category,
        "prefix": "prefix:",
        "removed": "remove-me",
        "surviving": "survive",
        "query": "?",
        "oracle": {"answer": "yes"},
    }


def result(token_ids: list[int] | None, structured: dict | None = None, *, applied: bool = False) -> dict:
    metadata = None
    if applied:
        metadata = {
            "leyline": {
                "applied": True,
                "transformed_tokens": 128,
                "fallback_reason": None,
                "transform_complete": True,
            }
        }
    value = {"output_token_ids": token_ids, "structured": structured, "kv_transfer_params": metadata}
    if applied:
        value["record"] = {"kv_transfer_params": {"leyline": {"recorded": True}}}
    return value


class ValidationHarnessTest(unittest.TestCase):
    def test_raw_and_chat_plans_are_exact_deletions(self) -> None:
        tokenizer = CharacterTokenizer()
        for prompt_format in ("raw", "chat_template"):
            plan = build_prompt_plan(tokenizer, case(), prompt_format=prompt_format)
            self.assertEqual(plan.full[: plan.delete_start] + plan.full[plan.delete_end :], plan.edited)

    def test_chat_requires_template(self) -> None:
        class NoChatTokenizer:
            chat_template = None

            def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
                return [ord(character) for character in text]

        with self.assertRaises(ValueError):
            build_prompt_plan(NoChatTokenizer(), case(), prompt_format="chat_template")

    def test_boundary_replacement_is_rejected(self) -> None:
        class BoundaryTokenizer(CharacterTokenizer):
            def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
                if "remove-me" in text:
                    return [1, 2, 3]
                return [1, 9]

        with self.assertRaises(ValueError):
            build_prompt_plan(BoundaryTokenizer(), case())

    def test_mode_prompt_compatibility(self) -> None:
        self.assertEqual(
            evaluation_config({"evaluation": {"mode": REFERENCE_PREFIX}, "prompt_format": "raw"})[0],
            REFERENCE_PREFIX,
        )
        with self.assertRaises(ValueError):
            evaluation_config({"evaluation": {"mode": STRUCTURED_JSON}, "prompt_format": "raw"})
        self.assertEqual(
            evaluation_config(
                {"evaluation": {"mode": COMPLETION_TARGET}, "prompt_format": "raw"}
            )[0],
            COMPLETION_TARGET,
        )

    def test_completion_target_is_stable_and_non_whitespace(self) -> None:
        tokenizer = CharacterTokenizer()
        target_case = {**case(), "expected_completion": " Y"}
        plan = build_prompt_plan(tokenizer, target_case)
        self.assertEqual(plan.target_token_ids, (32, 89))
        with self.assertRaisesRegex(ValueError, "non-whitespace"):
            build_prompt_plan(tokenizer, {**case(), "expected_completion": "\n"})

    def test_tokenizer_aware_filler_hits_exact_deleted_length(self) -> None:
        tokenizer = PieceTokenizer()
        filler_case = {
            **case(),
            "prefix": "a=0 -> N\n",
            "removed": " neutral",
            "filler_unit": " neutral",
            "surviving": "\na=1 -> Y\n",
            "query": "a=1 ->",
            "expected_completion": " Y",
            "position_coverage": {"delete_tokens": 5},
        }
        plan = build_prompt_plan(tokenizer, filler_case)
        self.assertEqual(plan.delete_end - plan.delete_start, 5)
        self.assertTrue(plan.target_token_ids)

    def test_completion_preserves_ids_and_scores(self) -> None:
        captured: dict = {}

        def urlopen(request, timeout):
            self.assertEqual(timeout, 600)
            captured.update(json.loads(request.data))
            return io.BytesIO(
                json.dumps(
                    {
                        "choices": [
                            {
                                "text": "ok",
                                "token_ids": [7],
                                "logprobs": {
                                    "token_logprobs": [-0.1],
                                    "top_logprobs": [{"token_id:7": -0.1, "token_id:8": -0.6}],
                                },
                            }
                        ]
                    }
                ).encode()
            )

        with patch("benchmarks.leyline.run_validation.urlopen", urlopen):
            output = request_completion(
                "http://localhost:8000", "model", [1, 2], api_key=None, max_tokens=1, top_logprobs=2
            )
        self.assertFalse(captured["add_special_tokens"])
        self.assertTrue(captured["return_token_ids"])
        self.assertEqual(output["output_token_ids"], [7])
        self.assertEqual(output["first_token_scores"]["top1_top2_margin"], 0.5)
        self.assertEqual(output["first_token_scores"]["evidence_type"], "api_logprobs")

    def test_reference_diagnostics_and_strict_leyline_acceptance(self) -> None:
        arms = {
            "full": result([10, 11]),
            "honest_edited": result([10, 99]),
            "leyline": result([10, 12], applied=True),
        }
        evaluation = evaluate_case_results(
            case(), arms, [], mode=REFERENCE_PREFIX, reference_tokens=1
        )
        self.assertFalse(evaluation["gates"]["reference_admitted"])
        self.assertTrue(evaluation["gates"]["reference_diagnostic_matches"])
        self.assertFalse(evaluation["gates"]["leyline_accepted"])
        self.assertEqual(evaluation["pairwise"]["full_leyline"]["common_prefix_tokens"], 1)

        fallback = result([10])
        arms["leyline"] = fallback
        evaluation = evaluate_case_results(
            case(), arms, [], mode=REFERENCE_PREFIX, reference_tokens=1
        )
        self.assertFalse(evaluation["gates"]["leyline_accepted"])

    def test_completion_target_requires_full_honest_and_counterfactual(self) -> None:
        target = (89,)
        arms = {
            "full": result([89]),
            "honest_edited": result([89]),
            "leyline": result([89], applied=True),
        }
        evaluation = evaluate_case_results(
            case(),
            arms,
            [result([89])],
            mode=COMPLETION_TARGET,
            reference_tokens=1,
            target_token_ids=target,
        )
        self.assertTrue(evaluation["gates"]["completion_admitted"])
        self.assertTrue(evaluation["gates"]["leyline_accepted"])
        arms["honest_edited"] = result([78])
        failed = evaluate_case_results(
            case(),
            arms,
            [],
            mode=COMPLETION_TARGET,
            reference_tokens=1,
            target_token_ids=target,
        )
        self.assertFalse(failed["gates"]["admitted"])

    def test_categories_and_structured_preflight_gate(self) -> None:
        oracle = case()["oracle"]
        arms = {"full": result([1], oracle), "honest_edited": result([1], oracle)}
        for category, expected_admitted in (
            ("admissible", True),
            ("counterfactual_admissible", True),
            ("mechanism_diagnostic", False),
            ("negative_control", False),
        ):
            evaluation = evaluate_case_results(
                case(category),
                arms,
                [result([1], oracle)],
                mode=STRUCTURED_JSON,
                reference_tokens=1,
                preflight_passed=True,
            )
            self.assertEqual(evaluation["gates"]["admitted"], expected_admitted)
        failed = evaluate_case_results(
            case(), arms, [], mode=STRUCTURED_JSON, reference_tokens=1, preflight_passed=False
        )
        self.assertFalse(failed["gates"]["admitted"])
        self.assertTrue(failed["gates"]["diagnostic_only"])

    def test_preflight_skip_pass_parse_and_endpoint_failure(self) -> None:
        tokenizer = CharacterTokenizer()
        reference_config = {
            "model": "m",
            "prompt_format": "raw",
            "evaluation": {"mode": REFERENCE_PREFIX},
            "arms": {},
        }
        self.assertTrue(run_preflight(tokenizer, reference_config)["passed"])
        completion_config = {
            "model": "m",
            "prompt_format": "raw",
            "evaluation": {"mode": COMPLETION_TARGET},
            "arms": {},
        }
        self.assertEqual(
            run_preflight(tokenizer, completion_config)["skipped_reason"],
            "completion_target_mode",
        )
        structured_config = {
            "model": "m",
            "prompt_format": "chat_template",
            "evaluation": {"mode": STRUCTURED_JSON},
            "arms": {"full": "http://localhost"},
        }
        with patch(
            "benchmarks.leyline.run_validation.request_completion",
            return_value={"structured": {"ok": True}},
        ):
            self.assertTrue(run_preflight(tokenizer, structured_config)["passed"])
        with patch(
            "benchmarks.leyline.run_validation.request_completion",
            return_value={"structured": None},
        ):
            self.assertFalse(run_preflight(tokenizer, structured_config)["passed"])
        with patch(
            "benchmarks.leyline.run_validation.request_completion", side_effect=RuntimeError("offline")
        ):
            self.assertIn("offline", run_preflight(tokenizer, structured_config)["error"])
        mismatched = {
            **structured_config,
            "preflight": {"endpoint": "http://different-endpoint"},
        }
        self.assertIn("configured full endpoint", run_preflight(tokenizer, mismatched)["error"])

    def test_checkpoint_manifest_hashes_local_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text('{"model_type":"deepseek_v2"}')
            (root / "tokenizer_config.json").write_text('{"chat_template":"template"}')
            (root / "model.safetensors").write_bytes(b"weights")
            manifest = artifact_manifest(str(root), "local")
        self.assertEqual(len(manifest["artifacts"]), 3)
        self.assertTrue(manifest["chat_template_present"])
        self.assertTrue(all(len(item["sha256"]) == 64 for item in manifest["artifacts"]))

    def test_versioned_corpora_meet_family_contracts(self) -> None:
        root = Path(__file__).resolve().parents[5] / "benchmarks" / "leyline"
        base = json.loads((root / "workloads.base.json").read_text())
        chat = json.loads((root / "workloads.chat.json").read_text())
        validate_workload_corpus(
            base,
            {"prompt_format": "raw", "evaluation": {"mode": COMPLETION_TARGET}},
        )
        validate_workload_corpus(
            chat,
            {"prompt_format": "chat_template", "evaluation": {"mode": STRUCTURED_JSON}},
        )
        tokenizer = CharacterTokenizer()
        for chat_case in chat["cases"]:
            plan = build_prompt_plan(tokenizer, chat_case, prompt_format="chat_template")
            self.assertEqual(
                plan.full[: plan.delete_start] + plan.full[plan.delete_end :],
                plan.edited,
            )

    def test_legacy_workloads_require_explicit_diagnostic_mode(self) -> None:
        legacy = {"version": 1, "cases": [case()]}
        config = {
            "prompt_format": "raw",
            "evaluation": {"mode": REFERENCE_PREFIX},
            "legacy_diagnostic": True,
        }
        validate_workload_corpus(legacy, config)
        with self.assertRaisesRegex(ValueError, "legacy_diagnostic"):
            validate_workload_corpus(legacy, {**config, "legacy_diagnostic": False})

    def test_schema_v2_merge_and_performance_gate(self) -> None:
        base_case = case()
        arms = {
            "full": result([10]),
            "honest_edited": result([10]),
            "leyline": result([10], applied=True),
        }
        evaluation = evaluate_case_results(
            base_case,
            arms,
            [],
            mode=COMPLETION_TARGET,
            reference_tokens=1,
            target_token_ids=(10,),
        )
        document = {
            "schema_version": 2,
            "evaluation_contract": {
                "mode": COMPLETION_TARGET,
                "prompt_format": "raw",
                "score_evidence": "api_logprobs",
                "raw_logits_captured": False,
            },
            "preflight": {"passed": True},
            "checkpoint_identity": {"name": "checkpoint", "revision": "pinned"},
            "cases": [
                {
                    **base_case,
                    "prompt_tokens": {"full": 3, "edited": 2, "deleted": 1},
                    "target_token_ids": [10],
                    "evaluation": {"target_token_ids": [10]},
                    "arms": arms,
                    "counterfactuals": [],
                    "gates": evaluation["gates"],
                }
            ],
        }
        merged = merge_documents([document], ["one.json"])
        self.assertEqual(merged["schema_version"], 2)
        with self.assertRaises(ValueError):
            merge_documents([{"schema_version": 1}], ["legacy.json"])
        gate = performance_gate(
            merged["cases"],
            {"performance_prerequisites": {"numerical_passed": True, "rollback_passed": True}},
            {"passed": True},
        )
        self.assertTrue(gate["passed"])


if __name__ == "__main__":
    unittest.main()
