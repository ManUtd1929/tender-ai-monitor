"""
Тесты src.ai.golden_set_evaluator на временной SQLite БД, временных golden set/runs и
fake analyzer/client: реального OpenAI/сети нет, production data/tenders.db закрыт общей
защитой (tests/safety_guards.py).

Запуск из корня проекта:
    python -m unittest tests.test_golden_set_evaluator -v
"""

import hashlib
import io
import json
import re
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from src.ai import golden_set, golden_set_evaluator as evaluator, openai_triage
from src.database import enrichment_repository
from tests import safety_guards
from tests.test_golden_set import GoldenSetDbTestCase, make_expected
from tests.test_openai_triage import (
    FakeClient, make_response, payload, evidence_item, OTHER_SERVICE, UNRELATED, MAYBE_UNCLEAR,
)

API_KEY = "sk-test-secret-key-123"

USAGE = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150, "cached_tokens": 0, "reasoning_tokens": 20}


def prediction(**overrides) -> dict:
    return payload(**overrides)


class FakeAnalyzer:
    """Analyzer с заданными ответами по resource_url; записывает полученные context'ы."""

    model = "fake-model"
    reasoning_effort = "medium"
    prompt_version = "test-prompt"
    provider = "fake"

    def __init__(self, outcomes=None, default=None):
        self.outcomes = outcomes or {}
        self.default = default if default is not None else prediction()
        self.contexts = []

    def triage_with_metadata(self, triage_context):
        self.contexts.append(triage_context)
        outcome = self.outcomes.get(triage_context["resource_url"], self.default)
        if isinstance(outcome, Exception):
            raise outcome
        return {"result": outcome, "usage": dict(USAGE), "response_id": "resp_1", "attempts": 1}


class EvaluatorTestCase(GoldenSetDbTestCase):
    def setUp(self):
        super().setUp()
        self.runs_dir = self.tmp_dir / "runs"

    def make_golden(self, *expected_list) -> list:
        """Добавляет тендеры в БД и сохраняет golden set (валидный hash) во временный файл."""
        cases = []
        for number, expected in enumerate(expected_list, start=1):
            url = self.add(number)
            cases.append(golden_set.build_golden_case(url, expected, db_path=self.db_path))
        golden_set.save_golden_set(cases, self.golden_path)
        return cases

    def run_main(self, *args, analyzer):
        argv = [*args, "--path", str(self.golden_path), "--db-path", str(self.db_path), "--runs-dir", str(self.runs_dir)]
        out = io.StringIO()
        with redirect_stdout(out):
            code = evaluator.main(argv, analyzer=analyzer)
        return code, out.getvalue()

    def db_hash(self) -> str:
        return hashlib.sha256(self.db_path.read_bytes()).hexdigest()


RELEVANT = make_expected()
NOT_RELEVANT_SERVICE = make_expected(
    relevance_status="not_relevant", opportunity_type="other_service", category="cleaning_services",
)
NOT_RELEVANT_UNRELATED = make_expected(
    relevance_status="not_relevant", opportunity_type="unrelated", category=None,
)
MAYBE = make_expected(relevance_status="maybe", opportunity_type="unclear", category=None)

RELEVANT_PREDICTION = prediction(category="computers")


class CompareTests(unittest.TestCase):
    def test_all_fields_match(self):
        checks = evaluator.compare_prediction(evaluator.expected_core({"expected": RELEVANT}), RELEVANT_PREDICTION)
        self.assertTrue(all(checks.values()))

    def test_category_mismatch_only(self):
        expected = evaluator.expected_core({"expected": RELEVANT})
        checks = evaluator.compare_prediction(expected, prediction(category="furniture"))
        self.assertTrue(checks["relevance_status"] and checks["opportunity_type"] and checks["core"])
        self.assertFalse(checks["category"])
        self.assertFalse(checks["full"])

    def test_null_expected_category_is_not_evaluated_but_counts_for_full(self):
        expected = evaluator.expected_core({"expected": NOT_RELEVANT_UNRELATED})
        checks = evaluator.compare_prediction(expected, prediction(**UNRELATED))
        self.assertFalse(checks["category_evaluated"])
        self.assertTrue(checks["full"])
        wrong = evaluator.compare_prediction(expected, prediction(**{**OTHER_SERVICE}))
        self.assertFalse(wrong["full"])

    def test_reason_is_not_compared(self):
        expected = evaluator.expected_core({"expected": RELEVANT})
        checks = evaluator.compare_prediction(expected, {**RELEVANT_PREDICTION, "reason": "совсем другой текст"})
        self.assertTrue(checks["full"])

    def test_mismatch_classification(self):
        relevant = evaluator.expected_core({"expected": RELEVANT})
        service = evaluator.expected_core({"expected": NOT_RELEVANT_SERVICE})
        maybe = evaluator.expected_core({"expected": MAYBE})
        table = (
            (relevant, prediction(**OTHER_SERVICE), evaluator.MISMATCH_FALSE_NEGATIVE),
            (service, prediction(), evaluator.MISMATCH_FALSE_POSITIVE),
            (maybe, prediction(), evaluator.MISMATCH_MAYBE),
            (relevant, prediction(**MAYBE_UNCLEAR), evaluator.MISMATCH_MAYBE),
            (service, prediction(**MAYBE_UNCLEAR), evaluator.MISMATCH_MAYBE),
            (service, prediction(**UNRELATED), evaluator.MISMATCH_LABEL),
            (relevant, prediction(category="other"), evaluator.MISMATCH_LABEL),
            (relevant, RELEVANT_PREDICTION, None),
        )
        for expected, predicted, mismatch in table:
            with self.subTest(expected=expected, mismatch=mismatch):
                self.assertEqual(evaluator.classify_mismatch(expected, predicted), mismatch)


class EvaluateCaseTests(EvaluatorTestCase):
    def test_one_case_comparison(self):
        (case,) = self.make_golden(RELEVANT)
        analyzer = FakeAnalyzer(default=prediction(category="furniture"))

        record = evaluator.evaluate_case(case, analyzer, db_path=self.db_path)

        self.assertEqual(record["status"], evaluator.RECORD_SCORED)
        self.assertEqual(record["prediction"]["category"], "furniture")
        self.assertEqual(record["expected"], {
            "relevance_status": "relevant", "opportunity_type": "procurement", "category": "computers",
        })
        self.assertTrue(record["checks"]["relevance_status"])
        self.assertTrue(record["checks"]["opportunity_type"])
        self.assertFalse(record["checks"]["category"])
        self.assertEqual(record["usage"], USAGE)
        self.assertEqual(record["response_id"], "resp_1")
        self.assertEqual(record["input_hash"], case["input_hash"])
        self.assertIsNone(record["error"])

    def test_analyzer_receives_context_without_expected_labels(self):
        (case,) = self.make_golden(make_expected(expected_reason="СЕКРЕТНЫЙ-ЭТАЛОН", notes="СЕКРЕТНАЯ-ЗАМЕТКА"))
        analyzer = FakeAnalyzer()

        evaluator.evaluate_case(case, analyzer, db_path=self.db_path)

        (context,) = analyzer.contexts
        text = json.dumps(context, ensure_ascii=False)
        for secret in ("СЕКРЕТНЫЙ-ЭТАЛОН", "СЕКРЕТНАЯ-ЗАМЕТКА", "expected", "computers"):
            self.assertNotIn(secret, text)

    def test_false_negative_and_false_positive_detected(self):
        cases = self.make_golden(RELEVANT, NOT_RELEVANT_SERVICE)
        analyzer = FakeAnalyzer(outcomes={
            cases[0]["resource_url"]: prediction(**OTHER_SERVICE),
            cases[1]["resource_url"]: prediction(),
        })

        records = evaluator.run_evaluation(cases, analyzer, db_path=self.db_path)

        self.assertEqual(records[0]["mismatch"], evaluator.MISMATCH_FALSE_NEGATIVE)
        self.assertEqual(records[1]["mismatch"], evaluator.MISMATCH_FALSE_POSITIVE)
        metrics = evaluator.compute_metrics(records)
        self.assertEqual(metrics["mismatches"][evaluator.MISMATCH_FALSE_NEGATIVE], [cases[0]["case_id"]])
        self.assertEqual(metrics["mismatches"][evaluator.MISMATCH_FALSE_POSITIVE], [cases[1]["case_id"]])

    def test_stale_hash_blocks_api_call(self):
        (case,) = self.make_golden(RELEVANT)
        enrichment_repository.save_enrichment(
            case["resource_url"], golden_set_enrichment(description="Тендер изменился"), self.db_path,
        )
        analyzer = FakeAnalyzer()

        record = evaluator.evaluate_case(case, analyzer, db_path=self.db_path)

        self.assertEqual(analyzer.contexts, [])
        self.assertEqual(record["status"], evaluator.RECORD_SKIPPED)
        self.assertEqual(record["error"]["kind"], evaluator.SKIP_STALE)
        self.assertIsNone(record["prediction"])

    def test_tender_missing_from_db_blocks_api_call(self):
        (case,) = self.make_golden(RELEVANT)
        ghost = {**case, "case_id": "case-ghost", "resource_url": "https://example.test/resource/404"}
        analyzer = FakeAnalyzer()

        record = evaluator.evaluate_case(ghost, analyzer, db_path=self.db_path)

        self.assertEqual(analyzer.contexts, [])
        self.assertEqual(record["error"]["kind"], evaluator.SKIP_MISSING)

    def test_triage_error_is_recorded_with_usage(self):
        (case,) = self.make_golden(RELEVANT)
        error = openai_triage.TriageError(
            openai_triage.KIND_VALIDATION, "bad result", usage=dict(USAGE), response_id="resp_9", attempts=1,
        )

        record = evaluator.evaluate_case(case, FakeAnalyzer(default=error), db_path=self.db_path)

        self.assertEqual(record["status"], evaluator.RECORD_ERROR)
        self.assertEqual(record["error"], {"kind": "validation", "message": "bad result"})
        self.assertEqual(record["usage"], USAGE)
        self.assertEqual(record["response_id"], "resp_9")


def golden_set_enrichment(**overrides) -> dict:
    from tests.test_evaluation_dataset import make_enrichment
    return make_enrichment(**overrides)


class RunEvaluationTests(EvaluatorTestCase):
    def test_one_failing_case_does_not_abort_run(self):
        cases = self.make_golden(RELEVANT, RELEVANT, RELEVANT, RELEVANT)
        analyzer = FakeAnalyzer(outcomes={
            cases[1]["resource_url"]: openai_triage.TriageError(openai_triage.KIND_TIMEOUT, "timed out"),
            cases[2]["resource_url"]: RuntimeError("boom"),
        }, default=RELEVANT_PREDICTION)
        progress = []

        records = evaluator.run_evaluation(cases, analyzer, db_path=self.db_path, progress=progress.append)

        self.assertEqual([record["status"] for record in records], ["scored", "error", "error", "scored"])
        self.assertEqual(records[1]["error"]["kind"], "timeout")
        self.assertEqual(records[2]["error"]["kind"], evaluator.ERROR_UNEXPECTED)
        self.assertEqual(len(analyzer.contexts), 4)
        self.assertEqual(len(progress), 4)

    def test_metrics(self):
        cases = self.make_golden(RELEVANT, RELEVANT, NOT_RELEVANT_SERVICE, NOT_RELEVANT_UNRELATED, MAYBE, RELEVANT)
        urls = [case["resource_url"] for case in cases]
        analyzer = FakeAnalyzer(outcomes={
            urls[0]: RELEVANT_PREDICTION,                                   # full ok
            urls[1]: prediction(category="other_goods"),                     # core ok, category wrong
            urls[2]: prediction(**OTHER_SERVICE),                            # full ok
            urls[3]: prediction(**UNRELATED),                                # full ok (null category)
            urls[4]: prediction(),                                           # expected maybe -> relevant
            urls[5]: openai_triage.TriageError(openai_triage.KIND_REFUSAL, "no"),
        })

        records = evaluator.run_evaluation(cases, analyzer, db_path=self.db_path)
        metrics = evaluator.compute_metrics(records)

        self.assertEqual(metrics["total_cases"], 6)
        self.assertEqual(metrics["scored_cases"], 5)
        self.assertEqual(metrics["error_cases"], 1)
        self.assertEqual(metrics["error_kinds"], {"refusal": 1})
        self.assertEqual(metrics["relevance_status"], {"correct": 4, "total": 5, "accuracy": 0.8})
        self.assertEqual(metrics["opportunity_type"]["correct"], 4)
        # category: только expected != null (кейсы 0, 1, 2), верны 0 и 2
        self.assertEqual(metrics["category"], {"correct": 2, "total": 3, "accuracy": 2 / 3})
        self.assertEqual(metrics["core"]["correct"], 4)
        self.assertEqual(metrics["full"]["correct"], 3)
        confusion = metrics["confusion_relevance_status"]
        self.assertEqual(confusion["relevant"], {"relevant": 2, "maybe": 0, "not_relevant": 0})
        self.assertEqual(confusion["not_relevant"], {"relevant": 0, "maybe": 0, "not_relevant": 2})
        self.assertEqual(confusion["maybe"], {"relevant": 1, "maybe": 0, "not_relevant": 0})
        self.assertEqual(metrics["mismatches"][evaluator.MISMATCH_MAYBE], [cases[4]["case_id"]])
        self.assertEqual(metrics["mismatches"][evaluator.MISMATCH_LABEL], [cases[1]["case_id"]])
        self.assertEqual(metrics["usage_totals"]["total_tokens"], 150 * 5)

    def test_empty_metrics_have_no_accuracy(self):
        metrics = evaluator.compute_metrics([])
        self.assertIsNone(metrics["core"]["accuracy"])

    def test_report_lists_failures_with_reason(self):
        cases = self.make_golden(RELEVANT)
        analyzer = FakeAnalyzer(default=prediction(**{**OTHER_SERVICE, "reason": "Модель считает это услугой"}))
        records = evaluator.run_evaluation(cases, analyzer, db_path=self.db_path)

        report = evaluator.format_report(records, evaluator.compute_metrics(records))

        self.assertIn("FALSE NEGATIVE", report)
        self.assertIn(cases[0]["case_id"], report)
        self.assertIn(cases[0]["title"], report)
        self.assertIn("Модель считает это услугой", report)
        self.assertIn("Confusion matrix", report)


class DryRunTests(EvaluatorTestCase):
    def real_analyzer(self, environ, client=None):
        return openai_triage.OpenAITriageAnalyzer(client=client, environ=environ)

    def test_dry_run_makes_no_calls_and_writes_nothing(self):
        cases = self.make_golden(RELEVANT, NOT_RELEVANT_SERVICE)
        client = FakeClient()
        analyzer = self.real_analyzer({"OPENAI_API_KEY": API_KEY}, client=client)
        golden_before, db_before = self.golden_path.read_bytes(), self.db_hash()

        with mock.patch.object(analyzer, "triage_with_metadata", side_effect=AssertionError("API call")), \
                mock.patch.object(analyzer, "ensure_client", side_effect=AssertionError("client created")):
            code, output = self.run_main("--dry-run", analyzer=analyzer)

        self.assertEqual(code, 0, output)
        self.assertEqual(client.responses.calls, [])
        self.assertIn("2 кейс(ов)", output)
        self.assertIn("'ok': 2", output)
        self.assertIn("gpt-5.6-terra", output)
        self.assertIn("medium", output)
        self.assertIn("API-вызовов не выполнено", output)
        self.assertNotIn(API_KEY, output)
        self.assertFalse(self.runs_dir.exists())
        self.assertEqual(self.golden_path.read_bytes(), golden_before)
        self.assertEqual(self.db_hash(), db_before)
        self.assertEqual(len(cases), 2)

    def test_dry_run_reports_missing_api_key(self):
        self.make_golden(RELEVANT)
        client = FakeClient()
        analyzer = self.real_analyzer({})
        code, output = self.run_main("--dry-run", analyzer=analyzer)
        self.assertEqual(code, 1)
        self.assertIn("MISSING", output)
        self.assertEqual(client.responses.calls, [])

    def test_dry_run_reports_stale_cases(self):
        (case,) = self.make_golden(RELEVANT)
        enrichment_repository.save_enrichment(
            case["resource_url"], golden_set_enrichment(description="Изменён"), self.db_path,
        )
        code, output = self.run_main("--dry-run", analyzer=self.real_analyzer({"OPENAI_API_KEY": API_KEY}))
        self.assertEqual(code, 1)
        self.assertIn("STALE", output)

    def test_invalid_golden_set_is_reported(self):
        self.golden_path.write_text("[{\"case_id\": \"x\"}]", encoding="utf-8")
        code, output = self.run_main("--dry-run", analyzer=FakeAnalyzer())
        self.assertEqual(code, 2)
        self.assertIn("невалиден", output)

    def test_mode_is_required(self):
        with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), mock.patch("sys.stderr", io.StringIO()):
            evaluator.main([], analyzer=FakeAnalyzer())


class CliRunTests(EvaluatorTestCase):
    def test_case_id_runs_only_that_case_and_saves_artifact(self):
        cases = self.make_golden(RELEVANT, NOT_RELEVANT_SERVICE)
        analyzer = FakeAnalyzer(default=prediction(category="furniture"))
        golden_before, db_before = self.golden_path.read_bytes(), self.db_hash()

        code, output = self.run_main("--case-id", cases[0]["case_id"], analyzer=analyzer)

        self.assertEqual(code, 0, output)
        self.assertEqual([c["resource_url"] for c in analyzer.contexts], [cases[0]["resource_url"]])
        self.assertIn("prediction:", output)
        self.assertIn("expected:", output)
        self.assertRegex(output, r"relevance_status\s+PASS")
        self.assertRegex(output, r"opportunity_type\s+PASS")
        self.assertRegex(output, r"category\s+FAIL")
        self.assertIn("usage:", output)
        self.assertEqual(self.golden_path.read_bytes(), golden_before)
        self.assertEqual(self.db_hash(), db_before)

        (artifact_path,) = list(self.runs_dir.iterdir())
        self.assertRegex(artifact_path.name, r"^\d{8}T\d{6}Z_fake-model\.json$")
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        self.assertEqual(artifact["evaluator_version"], evaluator.EVALUATOR_VERSION)
        self.assertEqual(artifact["prompt_version"], "test-prompt")
        self.assertEqual(artifact["model"], "fake-model")
        self.assertEqual(artifact["reasoning_effort"], "medium")
        self.assertEqual(artifact["mode"], "case")
        self.assertRegex(artifact["timestamp"], r"^\d{4}-\d{2}-\d{2}T")
        (record,) = artifact["cases"]
        self.assertEqual(record["input_hash"], cases[0]["input_hash"])
        self.assertEqual(record["prediction"]["category"], "furniture")
        self.assertEqual(record["expected"]["category"], "computers")
        self.assertEqual(record["usage"], USAGE)
        self.assertFalse(record["checks"]["category"])
        self.assertIsNone(record["error"])

    def test_unknown_case_id_makes_no_calls(self):
        self.make_golden(RELEVANT)
        analyzer = FakeAnalyzer()
        code, output = self.run_main("--case-id", "case-nope", analyzer=analyzer)
        self.assertEqual(code, 2)
        self.assertEqual(analyzer.contexts, [])
        self.assertFalse(self.runs_dir.exists())

    def test_all_runs_every_case_and_survives_failures(self):
        cases = self.make_golden(RELEVANT, RELEVANT, NOT_RELEVANT_SERVICE)
        analyzer = FakeAnalyzer(outcomes={
            cases[0]["resource_url"]: openai_triage.TriageError(openai_triage.KIND_RATE_LIMIT, "429"),
            cases[1]["resource_url"]: RELEVANT_PREDICTION,
            cases[2]["resource_url"]: prediction(**OTHER_SERVICE),
        })

        code, output = self.run_main("--all", analyzer=analyzer)

        self.assertEqual(code, 1, "есть ошибочный кейс")
        self.assertEqual(len(analyzer.contexts), 3)
        self.assertIn("Confusion matrix", output)
        (artifact_path,) = list(self.runs_dir.iterdir())
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        self.assertEqual(artifact["mode"], "all")
        self.assertEqual(artifact["metrics"]["error_cases"], 1)
        self.assertEqual(artifact["metrics"]["scored_cases"], 2)
        self.assertEqual(artifact["cases"][0]["error"]["kind"], "rate_limit")

    def test_all_with_stale_case_skips_only_that_case(self):
        cases = self.make_golden(RELEVANT, RELEVANT)
        enrichment_repository.save_enrichment(
            cases[0]["resource_url"], golden_set_enrichment(description="Изменён"), self.db_path,
        )
        analyzer = FakeAnalyzer(default=RELEVANT_PREDICTION)

        code, output = self.run_main("--all", analyzer=analyzer)

        self.assertEqual(code, 1)
        self.assertEqual([c["resource_url"] for c in analyzer.contexts], [cases[1]["resource_url"]])
        self.assertIn(evaluator.SKIP_STALE, output)

    def test_missing_api_key_stops_before_any_call_or_artifact(self):
        cases = self.make_golden(RELEVANT)
        analyzer = openai_triage.OpenAITriageAnalyzer(environ={})
        with mock.patch("openai.OpenAI") as client_class:
            code, output = self.run_main("--case-id", cases[0]["case_id"], analyzer=analyzer)
        self.assertEqual(code, 2)
        self.assertIn("OPENAI_API_KEY", output)
        client_class.assert_not_called()
        self.assertFalse(self.runs_dir.exists())

    def test_artifact_never_contains_api_key(self):
        cases = self.make_golden(RELEVANT)
        # evidence обязан дословно цитировать title, который реально уходит модели
        client = FakeClient(make_response(payload(category="computers", evidence=[evidence_item(text="Тендер 1")])))
        analyzer = openai_triage.OpenAITriageAnalyzer(
            client=client, environ={"OPENAI_API_KEY": API_KEY}, sleep=lambda seconds: None,
        )

        code, output = self.run_main("--case-id", cases[0]["case_id"], analyzer=analyzer)

        self.assertEqual(code, 0, output)
        self.assertEqual(len(client.responses.calls), 1)
        sent = client.responses.calls[0]
        self.assertNotIn("СЕКРЕТ", sent["input"])
        self.assertNotIn(cases[0]["expected"]["expected_reason"], sent["input"])
        (artifact_path,) = list(self.runs_dir.iterdir())
        self.assertNotIn(API_KEY, artifact_path.read_text(encoding="utf-8"))
        self.assertNotIn(API_KEY, output)
        self.assertIn("gpt-5.6-terra", artifact_path.name)

    def test_artifact_is_not_overwritten(self):
        cases = self.make_golden(RELEVANT)
        records = evaluator.run_evaluation(cases, FakeAnalyzer(), db_path=self.db_path)
        artifact = evaluator.build_artifact(
            records, FakeAnalyzer(), "all", self.golden_path, evaluator._timestamp_now(),
        )
        evaluator.save_artifact(artifact, self.runs_dir)
        with self.assertRaises(FileExistsError):
            evaluator.save_artifact(artifact, self.runs_dir)


class SafetyTests(unittest.TestCase):
    def test_production_db_stays_blocked(self):
        case = {
            "case_id": "case-x", "resource_url": "https://example.test/x", "title": "x",
            "input_hash": "a" * 64, "expected": make_expected(),
        }
        with safety_guards.expect_violation(self, safety_guards.DATABASE_MESSAGE):
            evaluator.prepare_case(case, db_path=safety_guards.PRODUCTION_DB_PATH)

    def test_real_golden_set_is_valid_and_tracked_path(self):
        cases = evaluator.load_valid_cases(golden_set.DEFAULT_GOLDEN_SET_PATH)
        self.assertEqual(len(cases), 24)
        self.assertEqual(golden_set.DEFAULT_GOLDEN_SET_PATH.parent.name, "evaluation")

    def test_runs_dir_is_outside_golden_set_file(self):
        self.assertEqual(evaluator.DEFAULT_RUNS_DIR.name, "runs")
        self.assertNotEqual(evaluator.DEFAULT_RUNS_DIR, golden_set.DEFAULT_GOLDEN_SET_PATH)


if __name__ == "__main__":
    unittest.main()
