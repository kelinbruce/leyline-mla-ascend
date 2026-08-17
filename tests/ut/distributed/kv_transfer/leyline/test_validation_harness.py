# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from benchmarks.leyline.common.evidence import retained_context_summary
from benchmarks.leyline.scripts.collect_environment import artifact_manifest
from benchmarks.leyline.scripts.merge_reports import merge_documents
from benchmarks.leyline.scripts.run_rollback_validation import run_rollback
from benchmarks.leyline.scripts.run_validation import (
    COMPLETION_TARGET,
    PromptPlan,
    REFERENCE_PREFIX,
    STRUCTURED_JSON,
    _stability_summary,
    build_prompt_plan,
    evaluate_case_results,
    evaluate_transform_smoke,
    evaluation_config,
    plan_transform_feasibility,
    performance_gate,
    request_completion,
    run_preflight,
    validate_workload_corpus,
    validate_workload_feasibility,
    workload_feasibility_report,
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
                "expected_layers": 27,
                "transformed_layers": 27,
                "expected_ranks": 4,
                "successful_ranks": 4,
            }
        }
    value = {"output_token_ids": token_ids, "structured": structured, "kv_transfer_params": metadata}
    if applied:
        value["record"] = {"kv_transfer_params": {"leyline": {"recorded": True}}}
    return value


class ValidationHarnessTest(unittest.TestCase):
    def test_transform_feasibility_boundary_conditions(self) -> None:
        short_source = PromptPlan(list(range(100)), list(range(90)), 10, 20)
        self.assertEqual(
            plan_transform_feasibility(short_source, block_size=128).reason,
            "missing_source_blocks",
        )

        short_edited = PromptPlan(list(range(256)), list(range(100)), 0, 156)
        self.assertEqual(
            plan_transform_feasibility(short_edited, block_size=128).reason,
            "no_reusable_target_blocks",
        )

        partial_source = PromptPlan(list(range(200)), list(range(150)), 0, 50)
        partial = plan_transform_feasibility(partial_source, block_size=128)
        self.assertEqual(partial.reason, "mapped_source_block_not_resident")
        self.assertEqual(partial.blocking_source_block, 1)

        pre_delete_only = PromptPlan(list(range(400)), list(range(300)), 256, 356)
        pre_delete = plan_transform_feasibility(pre_delete_only, block_size=128)
        self.assertEqual(pre_delete.predicted_transformed_tokens, 256)
        self.assertEqual(pre_delete.predicted_shifted_tokens, 0)

    def test_transform_feasibility_long_deletion_and_local_apc(self) -> None:
        long_delete = PromptPlan(list(range(1200)), list(range(176)), 32, 1056)
        long_result = plan_transform_feasibility(long_delete, block_size=128)
        self.assertEqual(long_result.predicted_transformed_tokens, 128)
        self.assertIsNone(long_result.reason)

        local_apc = PromptPlan(list(range(400)), list(range(350)), 250, 300)
        local_result = plan_transform_feasibility(
            local_apc,
            block_size=128,
            local_computed_tokens=128,
        )
        self.assertEqual(local_result.reusable_target_start, 128)
        self.assertEqual(local_result.predicted_transformed_tokens, 128)

    def test_surviving_filler_expands_to_transformable_plan(self) -> None:
        tokenizer = CharacterTokenizer()
        filler_case = {
            **case(),
            "expected_completion": " Y",
            "surviving_filler_unit": "neutral\n",
            "surviving_filler_max_repeat": 64,
            "minimum_transform_tokens": 128,
        }
        plan = build_prompt_plan(tokenizer, filler_case, block_size=128)
        feasibility = plan_transform_feasibility(plan, block_size=128)
        self.assertGreater(plan.surviving_filler_repeat, 0)
        self.assertGreaterEqual(feasibility.predicted_transformed_tokens, 128)
        self.assertEqual(
            plan.full[: plan.delete_start] + plan.full[plan.delete_end :],
            plan.edited,
        )

    def test_required_and_diagnostic_feasibility_admission(self) -> None:
        tokenizer = CharacterTokenizer()
        required = {
            **case(),
            "execution_expectation": "required",
            "minimum_transform_tokens": 128,
        }
        report = workload_feasibility_report(
            [required],
            tokenizer,
            {"prompt_format": "raw", "evaluation": {"mode": REFERENCE_PREFIX}},
        )
        self.assertFalse(report["passed"])
        with self.assertRaisesRegex(ValueError, "missing_source_blocks"):
            validate_workload_feasibility(
                [required],
                tokenizer,
                {"prompt_format": "raw", "evaluation": {"mode": REFERENCE_PREFIX}},
            )
        diagnostic = {
            **required,
            "execution_expectation": "diagnostic",
            "minimum_transform_tokens": 0,
        }
        self.assertTrue(
            workload_feasibility_report(
                [diagnostic],
                tokenizer,
                {"prompt_format": "raw", "evaluation": {"mode": REFERENCE_PREFIX}},
            )["passed"]
        )

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

        with patch("benchmarks.leyline.scripts.run_validation.urlopen", urlopen):
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
            "benchmarks.leyline.scripts.run_validation.request_completion",
            return_value={"structured": {"ok": True}},
        ):
            self.assertTrue(run_preflight(tokenizer, structured_config)["passed"])
        with patch(
            "benchmarks.leyline.scripts.run_validation.request_completion",
            return_value={"structured": None},
        ):
            self.assertFalse(run_preflight(tokenizer, structured_config)["passed"])
        with patch(
            "benchmarks.leyline.scripts.run_validation.request_completion",
            side_effect=RuntimeError("offline"),
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
        root = Path(__file__).resolve().parents[5] / "benchmarks" / "leyline" / "workloads"
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

    def test_retained_context_family_layout_and_empirical_admission(self) -> None:
        root = Path(__file__).resolve().parents[5] / "benchmarks" / "leyline" / "workloads"
        base = json.loads((root / "workloads.base.json").read_text())
        schema = json.loads((root / "workload.schema.json").read_text())
        case_schema = schema["$defs"]["case"]
        self.assertIn(
            "diagnostic_family",
            case_schema["allOf"][0]["then"]["required"],
        )
        allowed_fields = set(case_schema["properties"])
        required_fields = set(case_schema["required"])
        for item in base["cases"]:
            self.assertFalse(set(item) - allowed_fields)
            self.assertFalse(required_fields - set(item))
        diagnostics = [
            item
            for item in base["cases"]
            if item.get("claim_type") == "retained_context_diagnostic"
        ]
        self.assertGreaterEqual(len(diagnostics), 4)
        self.assertGreaterEqual(
            len({item["diagnostic_family"] for item in diagnostics}), 3
        )
        tokenizer = CharacterTokenizer()
        for item in diagnostics:
            plan = build_prompt_plan(tokenizer, item, block_size=128)
            self.assertEqual(
                plan.full[: plan.delete_start] + plan.full[plan.delete_end :],
                plan.edited,
            )
            self.assertGreaterEqual(
                plan_transform_feasibility(plan, block_size=128).predicted_transformed_tokens,
                128,
            )
            query_ids = tokenizer.encode(item["query"], add_special_tokens=False)
            self.assertEqual(plan.edited[-len(query_ids) :], query_ids)

        report_cases = []
        for item in diagnostics:
            runs = [
                {
                    "gates": {
                        "full_matches": True,
                        "honest_edited_matches": False,
                        "leyline_matches": True,
                    },
                    "arms": {
                        "full": {"output_token_ids": [1]},
                        "honest_edited": {"output_token_ids": [2]},
                        "leyline": {"output_token_ids": [1]},
                    },
                    "pairwise": {
                        "full_honest_edited": {"first_token_agreement": False},
                        "full_leyline": {"common_prefix_tokens": 1},
                        "honest_edited_leyline": {"common_prefix_tokens": 0},
                    },
                }
                for _ in range(3)
            ]
            report_cases.append(
                {
                    "id": item["id"],
                    "family": item["family"],
                    "diagnostic_family": item["diagnostic_family"],
                    "claim_type": item["claim_type"],
                    "repetitions": runs,
                }
            )
        summary = retained_context_summary({"cases": report_cases})
        self.assertTrue(summary["passed"])
        report_cases[0]["repetitions"][0]["gates"]["honest_edited_matches"] = True
        self.assertFalse(retained_context_summary({"cases": report_cases})["passed"])

    def test_rollback_runner_emits_complete_post_write_evidence(self) -> None:
        tokenizer = CharacterTokenizer()
        workload_case = {
            **case(),
            "id": "rollback-case",
            "evaluation_mode": COMPLETION_TARGET,
            "family": "admissible",
            "claim_type": "admissible_target",
            "execution_expectation": "required",
            "minimum_transform_tokens": 128,
            "surviving_filler_unit": "neutral\n",
            "surviving_filler_max_repeat": 64,
            "expected_completion": " Y",
        }
        config = {
            "run_id": "rollback-run",
            "model": "model",
            "prompt_format": "raw",
            "evaluation": {"mode": COMPLETION_TARGET},
            "block_size": 128,
            "max_tokens": 2,
            "arms": {"honest_edited": "http://honest", "leyline": "http://leyline"},
        }
        environment = {
            "model": {"name": "model"},
            "repositories": {
                "vllm": {"path": "/src/vllm", "commit": "v"},
                "vllm_ascend": {"path": "/src/ascend", "commit": "a"},
            },
            "imported_modules": {
                "vllm": {"module_file": "/src/vllm/vllm/__init__.py"},
                "vllm_ascend": {"module_file": "/src/ascend/vllm_ascend/__init__.py"},
            },
            "runtime_config": {"tensor_parallel_size": 4, "block_size": 128},
            "topology": {"stdout": "TP4"},
            "cann_installation": {"version_files": {"/cann/version.info": "9.0"}},
        }

        def completion(_endpoint, _model, prompt_ids, **kwargs):
            directive = (kwargs.get("kv_transfer_params") or {}).get("leyline") or {}
            request_id = kwargs["request_id"]
            if directive.get("action") == "record":
                return {
                    "request_id": request_id,
                    "output_token_ids": [32, 89],
                    "kv_transfer_params": {"leyline": {"recorded": True}},
                }
            if directive.get("action") == "amortize":
                return {
                    "request_id": request_id,
                    "output_token_ids": [32, 89],
                    "kv_transfer_params": {
                        "leyline": {
                            "injection_reached": True,
                            "injected_rank": 1,
                            "injected_layer": 2,
                            "destination_writes": 1,
                            "applied": False,
                            "fallback_reason": "transform_failed",
                            "invalidated_destination_blocks": 1,
                            "local_apc_tokens": 0,
                            "normal_prefill_tokens": len(prompt_ids),
                            "cleanup": {
                                "sessions": 0,
                                "inflight": 0,
                                "pending": 0,
                                "matches": 0,
                                "transaction_owned_references": 0,
                            },
                        }
                    },
                }
            return {"request_id": request_id, "output_token_ids": [32, 89]}

        with patch.dict(
            "os.environ",
            {"VLLM_ASCEND_LEYLINE_FAULT_INJECTION": "validation-only"},
        ), patch(
            "benchmarks.leyline.scripts.run_rollback_validation.request_completion",
            completion,
        ):
            report = run_rollback(
                config,
                {"version": 2, "corpus_id": "test", "cases": [workload_case]},
                environment,
                tokenizer,
                case_id="rollback-case",
                fail_rank=1,
                fail_after_layer=2,
            )
        self.assertTrue(report["passed"])
        self.assertTrue(all(report["conditions"].values()))
        self.assertEqual(report["evidence_identity"]["run_id"], "rollback-run")

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

    def test_merge_preserves_cache_off_repetitions_and_variants(self) -> None:
        base_case = {
            **case(),
            "family": "counterfactual",
            "claim_type": "counterfactual_invariance",
            "evaluation": {"reference_tokens": None},
            "target_token_ids": [10],
        }
        source_runs = []
        cache_off_runs = []
        for repetition in range(3):
            source_runs.append(
                {
                    "arms": {
                        "full": result([10]),
                        "honest_edited": result([10]),
                        "leyline": result([10], applied=True),
                    },
                    "counterfactuals": [result([10]), result([10])],
                }
            )
            cache_off_runs.append(
                {
                    "arms": {"cache_off": result([10])},
                    "counterfactuals": [
                        {**result([10]), "variant": f"off-{repetition}-0"},
                        {**result([10]), "variant": f"off-{repetition}-1"},
                    ],
                }
            )
        source_case = {**deepcopy(base_case), "repetitions": source_runs}
        cache_off_case = {**deepcopy(base_case), "repetitions": cache_off_runs}
        common = {
            "schema_version": 2,
            "evaluation_contract": {
                "mode": COMPLETION_TARGET,
                "prompt_format": "raw",
            },
            "preflight": {"passed": True},
            "checkpoint_identity": {"name": "checkpoint", "revision": "pinned"},
        }
        merged = merge_documents(
            [
                {**common, "cases": [source_case]},
                {**common, "cases": [cache_off_case]},
            ],
            ["source.json", "cache-off.json"],
        )
        runs = merged["cases"][0]["repetitions"]
        self.assertEqual(len(runs), 3)
        self.assertTrue(all(len(run["counterfactuals"]) == 2 for run in runs))
        self.assertTrue(
            all(len(run["cache_off_counterfactuals"]) == 2 for run in runs)
        )
        self.assertEqual(
            runs[2]["cache_off_counterfactuals"][1]["variant"], "off-2-1"
        )

    def test_transform_smoke_and_split_stability(self) -> None:
        leyline = result([10], applied=True)
        execution = {
            "recorded": True,
            "applied": True,
            "transformed_tokens_positive": True,
            "no_fallback": True,
            "transform_complete": True,
            "output_token_ids_present": True,
            "valid": True,
        }
        smoke = evaluate_transform_smoke(
            {"arms": {"leyline": leyline}, "leyline_execution": execution},
            {"block_size": 128, "diagnostics": {"device_capture_enabled": False}},
        )
        self.assertTrue(smoke["passed"])
        capture_required = evaluate_transform_smoke(
            {"arms": {"leyline": leyline}, "leyline_execution": execution},
            {
                "block_size": 128,
                "diagnostics": {"device_capture_enabled": True},
            },
        )
        self.assertFalse(capture_required["passed"])
        self.assertEqual(
            capture_required["capture"]["reason"],
            "device_capture_dir_missing",
        )
        with tempfile.TemporaryDirectory() as directory:
            leyline["request_id"] = "smoke-request"
            metadata = leyline["kv_transfer_params"]["leyline"]
            metadata["expected_ranks"] = 1
            metadata["successful_ranks"] = 1
            Path(directory, "smoke.rank0.manifest.json").write_text(
                json.dumps(
                    {
                        "rank": 0,
                        "complete_for_rank": True,
                        "captures": [{"request_id": "smoke-request"}],
                    }
                )
            )
            captured = evaluate_transform_smoke(
                {"arms": {"leyline": leyline}, "leyline_execution": execution},
                {
                    "block_size": 128,
                    "diagnostics": {
                        "device_capture_enabled": True,
                        "device_capture_dir": directory,
                    },
                },
            )
            self.assertTrue(captured["passed"])
        execution["applied"] = False
        self.assertFalse(
            evaluate_transform_smoke(
                {"arms": {"leyline": leyline}, "leyline_execution": execution},
                {"block_size": 128},
            )["passed"]
        )

        gates = {
            "full_matches": True,
            "honest_edited_matches": True,
            "leyline_matches": True,
            "leyline_execution_valid": True,
        }
        first = {
            "gates": gates,
            "arms": {"leyline": result([10], applied=True)},
            "counterfactuals": [],
        }
        second = {
            "gates": gates,
            "arms": {"leyline": result([11], applied=True)},
            "counterfactuals": [],
        }
        stability = _stability_summary([first, second])
        self.assertTrue(stability["stable"])
        self.assertTrue(stability["execution_stable"])
        self.assertFalse(stability["generation_stable"])


if __name__ == "__main__":
    unittest.main()
