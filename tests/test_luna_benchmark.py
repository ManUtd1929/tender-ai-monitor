"""
Тесты Luna-ready triage и benchmark: тот же procurement-v2 prompt/schema/grounding без
model-specific правил, CLI --model, artifact со стоимостью, критерии допуска. Реальных
вызовов нет (fake client/analyzer, сеть закрыта safety_guards).

Запуск: python -m unittest tests.test_luna_benchmark -v
"""

import json
import unittest
from decimal import Decimal
from pathlib import Path
from unittest import mock

from src.ai import golden_set_evaluator as evaluator
from src.ai import openai_triage, pricing, tender_context as tender_context_module, triage_prompt
from tests.test_deep_prompt import make_tender_context
from tests.test_golden_set_evaluator import (
    API_KEY, RELEVANT, EvaluatorTestCase, FakeAnalyzer, evidence_item, make_response, payload, FakeClient,
)

LUNA = "gpt-5.6-luna"


class LunaAnalyzer(FakeAnalyzer):
    model = LUNA


def triage_context():
    return tender_context_module.build_triage_context(make_tender_context())


class LunaTriageSupportTests(unittest.TestCase):
    def analyzer(self, model):
        return openai_triage.OpenAITriageAnalyzer(model=model, reasoning_effort="medium", environ={})

    def test_luna_and_medium_effort_accepted(self):
        analyzer = self.analyzer(LUNA)
        self.assertEqual(analyzer.model, LUNA)
        self.assertEqual(analyzer.reasoning_effort, "medium")
        self.assertEqual(analyzer.prompt_version, "procurement-v2")

    def test_request_differs_from_terra_only_by_model(self):
        context = triage_context()
        luna, terra = self.analyzer(LUNA).build_request(context), self.analyzer("gpt-5.6-terra").build_request(context)
        self.assertEqual(luna["model"], LUNA)
        self.assertEqual({**luna, "model": None}, {**terra, "model": None})

    def test_structured_output_and_prompt_unchanged(self):
        request = self.analyzer(LUNA).build_request(triage_context())
        self.assertEqual(request["instructions"], triage_prompt.SYSTEM_PROMPT)
        self.assertEqual(request["text"], triage_prompt.build_text_format())
        self.assertEqual(request["text"]["format"]["type"], "json_schema")
        self.assertTrue(request["text"]["format"]["strict"])
        self.assertNotIn("tools", request)

    def test_no_luna_specific_rules_in_prompt_sources(self):
        for name in ("triage_prompt.py", "openai_triage.py"):
            source = (Path(triage_prompt.__file__).parent / name).read_text(encoding="utf-8")
            # допустима только константа модели по умолчанию, а не Luna-специфичные правила
            self.assertNotIn("luna", source.lower().replace("gpt-5.6-luna", ""), name)
        self.assertNotIn("luna", triage_prompt.SYSTEM_PROMPT.lower())

    def test_default_evaluator_model_is_not_silently_switched(self):
        self.assertEqual(openai_triage.DEFAULT_MODEL, "gpt-5.6-luna")
        self.assertEqual(openai_triage.load_settings({})["model"], "gpt-5.6-luna")

    def test_luna_runs_through_same_parsing_and_grounding(self):
        client = FakeClient(make_response(payload(category="computers", evidence=[evidence_item(text="Поставка компьютерной техники")])))
        analyzer = openai_triage.OpenAITriageAnalyzer(client=client, model=LUNA, environ={})
        outcome = analyzer.triage_with_metadata(triage_context())
        self.assertEqual(outcome["result"]["relevance_status"], "relevant")
        self.assertEqual(client.responses.calls[0]["model"], LUNA)


class BenchmarkCliTests(EvaluatorTestCase):
    def test_cli_model_flag_overrides_default_model(self):
        args = evaluator._build_arg_parser().parse_args(["--dry-run", "--model", LUNA, "--reasoning-effort", "medium"])
        with mock.patch("dotenv.load_dotenv"), mock.patch.dict("os.environ", {}, clear=True):
            analyzer = evaluator._default_analyzer(args)
        self.assertEqual((analyzer.model, analyzer.reasoning_effort), (LUNA, "medium"))

    def test_all_flag_with_model_is_accepted_by_parser(self):
        args = evaluator._build_arg_parser().parse_args(["--all", "--model", LUNA])
        self.assertTrue(args.all)
        self.assertEqual(args.model, LUNA)

    def test_dry_run_prints_conservative_cost_estimate_without_calls(self):
        self.make_golden(RELEVANT)
        analyzer = openai_triage.OpenAITriageAnalyzer(
            client=FakeClient(), model=LUNA, environ={"OPENAI_API_KEY": API_KEY},
        )
        code, output = self.run_main("--dry-run", analyzer=analyzer)
        self.assertEqual(code, 0, output)
        self.assertIn("Консервативная оценка стоимости", output)
        self.assertIn(LUNA, output)

    def test_dry_run_unknown_model_reports_no_price_instead_of_inventing_one(self):
        self.make_golden(RELEVANT)
        analyzer = openai_triage.OpenAITriageAnalyzer(
            client=FakeClient(), model="gpt-unknown", environ={"OPENAI_API_KEY": API_KEY},
        )
        _, output = self.run_main("--dry-run", analyzer=analyzer)
        self.assertIn("Оценка стоимости недоступна", output)

    def test_artifact_stores_model_prompt_effort_metrics_usage_and_cost(self):
        cases = self.make_golden(RELEVANT, RELEVANT)
        analyzer = LunaAnalyzer()
        records = evaluator.run_evaluation(cases, analyzer, db_path=self.db_path)
        artifact = evaluator.build_artifact(records, analyzer, "all", self.golden_path, evaluator._timestamp_now())
        self.assertEqual(artifact["model"], LUNA)
        self.assertEqual(artifact["prompt_version"], "test-prompt")
        self.assertEqual(artifact["reasoning_effort"], "medium")
        self.assertIn("usage_totals", artifact["metrics"])
        self.assertTrue(all(record["usage"] for record in artifact["cases"]))
        # по 100 in / 50 out на кейс: (100*0.20 + 50*1.20) / 1e6 = 0.00008; два кейса
        self.assertEqual(Decimal(artifact["cost"]["estimated_cost_usd"]), Decimal("0.00016"))
        self.assertEqual(artifact["cost"]["pricing_version"], pricing.PRICING_VERSION)
        json.dumps(artifact)  # сериализуется (Decimal хранится строкой)

    def test_artifact_cost_is_none_for_unknown_model(self):
        cases = self.make_golden(RELEVANT)
        analyzer = FakeAnalyzer()  # model "fake-model" не имеет цены
        records = evaluator.run_evaluation(cases, analyzer, db_path=self.db_path)
        artifact = evaluator.build_artifact(records, analyzer, "all", self.golden_path, evaluator._timestamp_now())
        self.assertIsNone(artifact["cost"]["estimated_cost_usd"])
        self.assertIn("fake-model", artifact["cost"]["error"])

    def test_case_run_prints_cost_and_criteria_and_calls_only_selected_model(self):
        cases = self.make_golden(RELEVANT)
        client = FakeClient(make_response(payload(category="computers", evidence=[evidence_item(text="Тендер 1")])))
        analyzer = openai_triage.OpenAITriageAnalyzer(
            client=client, model=LUNA, environ={"OPENAI_API_KEY": API_KEY}, sleep=lambda seconds: None,
        )
        code, output = self.run_main("--case-id", cases[0]["case_id"], analyzer=analyzer)
        self.assertEqual(code, 0, output)
        self.assertEqual([call["model"] for call in client.responses.calls], [LUNA])
        self.assertIn("Критерии допуска к production triage", output)
        self.assertIn("Оценка стоимости прогона", output)
        (artifact_path,) = list(self.runs_dir.iterdir())
        self.assertIn(LUNA, artifact_path.name)


class ProductionCriteriaTests(EvaluatorTestCase):
    def evaluate(self, analyzer, n=2):
        cases = self.make_golden(*[RELEVANT] * n)
        records = evaluator.run_evaluation(cases, analyzer, db_path=self.db_path)
        return evaluator.production_criteria(evaluator.compute_metrics(records))

    def test_all_correct_passes(self):
        analyzer = LunaAnalyzer(default=payload(category="computers"))
        result = self.evaluate(analyzer)
        self.assertTrue(result["passed"], result)

    def test_category_mismatch_does_not_block_production_criteria(self):
        result = self.evaluate(LunaAnalyzer(default=payload(category="furniture")))
        self.assertTrue(result["passed"], result)

    def test_false_negative_fails(self):
        from tests.test_openai_triage import UNRELATED

        result = self.evaluate(LunaAnalyzer(default=payload(**UNRELATED)))
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["false_negatives_zero"])
        self.assertFalse(result["checks"]["core_all_correct"])

    def test_api_error_fails(self):
        error = openai_triage.TriageError(openai_triage.KIND_TIMEOUT, "timeout")
        result = self.evaluate(LunaAnalyzer(default=error))
        self.assertFalse(result["checks"]["api_validation_errors_zero"])
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
