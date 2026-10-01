"""
Тесты src.ai.deep_analysis_evaluator на временной SQLite БД, временных golden set/runs и
fake analyzer: реального OpenAI/сети нет, production data/tenders.db закрыт общей защитой
(tests/safety_guards.py). Нет human-labeled deep golden set — здесь не проверяется "deep
accuracy", только правильная работа evaluator'а (hydrate/hash/eligibility/API-call/artifact).

Запуск из корня проекта:
    python -m unittest tests.test_deep_analysis_evaluator -v
"""

import hashlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest import mock

from src.ai import deep_analysis_evaluator as evaluator
from src.ai import golden_set, openai_deep_analysis
from src.database import enrichment_repository
from tests import safety_guards
from tests.test_evaluation_dataset import make_enrichment
from tests.test_golden_set import GoldenSetDbTestCase, make_expected

API_KEY = "sk-test-secret-key-123"


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

USAGE = {"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500, "cached_tokens": 0, "reasoning_tokens": 200}

RELEVANT = make_expected()  # relevant/procurement/computers — eligible for deep analysis
NOT_RELEVANT = make_expected(
    relevance_status="not_relevant", opportunity_type="unrelated", category=None,
)
MAYBE_UNCLEAR = make_expected(relevance_status="maybe", opportunity_type="unclear", category=None)


def procurement_block(**overrides) -> dict:
    from src.ai import relevance_schema

    block = {name: None for name in relevance_schema.PROCUREMENT_SCALAR_FIELDS}
    for name in relevance_schema.PROCUREMENT_OBJECT_FIELDS:
        block[name] = None
    for name in relevance_schema.PROCUREMENT_LIST_FIELDS:
        block[name] = []
    block["subject"] = "Поставка товара"
    block.update(overrides)
    return block


def procurement_result(title: str, **overrides) -> dict:
    result = {
        "summary": "Закупка товара.",
        "opportunity_type": "procurement",
        "category": "computer_equipment",
        "why_interesting": "Физический товар, потенциально можно импортировать.",
        "contracting_authority": None,
        "procedure_code": None,
        "confidence": "high",
        "participation_barriers": [],
        "missing_information": [],
        "source_conflicts": [],
        "manual_review_required": False,
        "evidence": [{
            "source_type": "announcement", "field": "title", "download_id": None,
            "member_name": None, "text": title,
        }],
        "procurement": procurement_block(),
        "logistics": None,
    }
    result.update(overrides)
    return result


class FakeDeepAnalyzer:
    model = "fake-model"
    reasoning_effort = "high"
    prompt_version = "test-deep-prompt"
    provider = "fake"

    def __init__(self, outcomes=None, default=None):
        self.outcomes = outcomes or {}
        self.default = default
        self.tender_contexts = []

    def deep_analyze_with_metadata(self, tender_context, deep_analysis_context, triage_result):
        self.tender_contexts.append(tender_context)
        outcome = self.outcomes.get(
            tender_context["resource_url"],
            self.default if self.default is not None else procurement_result(tender_context["announcement"]["title"]),
        )
        if isinstance(outcome, Exception):
            raise outcome
        return {"result": outcome, "usage": dict(USAGE), "response_id": "resp_deep_1", "attempts": 1}


class EvaluatorTestCase(GoldenSetDbTestCase):
    def setUp(self):
        super().setUp()
        self.deep_runs_dir = self.tmp_dir / "deep_runs"

    def make_golden(self, *expected_list) -> list:
        cases = []
        for number, expected in enumerate(expected_list, start=1):
            url = self.add(number)
            cases.append(golden_set.build_golden_case(url, expected, db_path=self.db_path))
        golden_set.save_golden_set(cases, self.golden_path)
        return cases

    def run_main(self, *args, analyzer):
        argv = [
            *args, "--path", str(self.golden_path), "--db-path", str(self.db_path),
            "--runs-dir", str(self.deep_runs_dir),
        ]
        out = io.StringIO()
        with redirect_stdout(out):
            code = evaluator.main(argv, analyzer=analyzer)
        return code, out.getvalue()


class PrepareCaseTests(EvaluatorTestCase):
    def test_relevant_procurement_is_eligible(self):
        (case,) = self.make_golden(RELEVANT)
        prepared = evaluator.prepare_case(case, db_path=self.db_path)
        self.assertEqual(prepared["status"], golden_set.HASH_OK)
        self.assertIsNotNone(prepared["tender_context"])
        self.assertIsNotNone(prepared["deep_analysis_context"])
        self.assertTrue(prepared["triage_result"]["requires_deep_analysis"])

    def test_maybe_unclear_is_eligible(self):
        (case,) = self.make_golden(MAYBE_UNCLEAR)
        prepared = evaluator.prepare_case(case, db_path=self.db_path)
        self.assertEqual(prepared["status"], golden_set.HASH_OK)

    def test_not_relevant_is_not_eligible(self):
        (case,) = self.make_golden(NOT_RELEVANT)
        prepared = evaluator.prepare_case(case, db_path=self.db_path)
        self.assertEqual(prepared["status"], evaluator.SKIP_NOT_ELIGIBLE)
        self.assertIsNone(prepared["tender_context"])

    def test_stale_hash_is_reported(self):
        (case,) = self.make_golden(RELEVANT)
        enrichment_repository.save_enrichment(
            case["resource_url"], make_enrichment(description="Изменён"), self.db_path,
        )
        prepared = evaluator.prepare_case(case, db_path=self.db_path)
        self.assertEqual(prepared["status"], golden_set.HASH_STALE)

    def test_missing_tender_is_reported(self):
        (case,) = self.make_golden(RELEVANT)
        ghost = {**case, "case_id": "case-ghost", "resource_url": "https://example.test/resource/404"}
        prepared = evaluator.prepare_case(ghost, db_path=self.db_path)
        self.assertEqual(prepared["status"], golden_set.HASH_MISSING)


class EvaluateCaseTests(EvaluatorTestCase):
    def test_scored_case_has_full_record(self):
        (case,) = self.make_golden(RELEVANT)
        analyzer = FakeDeepAnalyzer()

        record = evaluator.evaluate_case(case, analyzer, db_path=self.db_path)

        self.assertEqual(record["status"], evaluator.RECORD_SCORED)
        self.assertEqual(record["case_id"], case["case_id"])
        self.assertEqual(record["triage_label"], {
            "relevance_status": "relevant", "opportunity_type": "procurement", "category": "computers",
        })
        self.assertIsNotNone(record["document_coverage"])
        self.assertGreater(record["input_size_chars"], 0)
        self.assertEqual(record["prediction"]["opportunity_type"], "procurement")
        self.assertEqual(record["usage"], USAGE)
        self.assertEqual(record["response_id"], "resp_deep_1")
        self.assertEqual(record["attempts"], 1)
        self.assertIsNone(record["error"])
        self.assertEqual(len(analyzer.tender_contexts), 1)

    def test_not_relevant_case_is_skipped_without_api_call(self):
        (case,) = self.make_golden(NOT_RELEVANT)
        analyzer = FakeDeepAnalyzer()

        record = evaluator.evaluate_case(case, analyzer, db_path=self.db_path)

        self.assertEqual(record["status"], evaluator.RECORD_SKIPPED)
        self.assertEqual(record["error"]["kind"], evaluator.SKIP_NOT_ELIGIBLE)
        self.assertEqual(analyzer.tender_contexts, [])

    def test_stale_hash_blocks_api_call(self):
        (case,) = self.make_golden(RELEVANT)
        enrichment_repository.save_enrichment(
            case["resource_url"], make_enrichment(description="Изменён"), self.db_path,
        )
        analyzer = FakeDeepAnalyzer()

        record = evaluator.evaluate_case(case, analyzer, db_path=self.db_path)

        self.assertEqual(analyzer.tender_contexts, [])
        self.assertEqual(record["status"], evaluator.RECORD_SKIPPED)
        self.assertEqual(record["error"]["kind"], evaluator.SKIP_STALE)

    def test_missing_tender_blocks_api_call(self):
        (case,) = self.make_golden(RELEVANT)
        ghost = {**case, "case_id": "case-ghost", "resource_url": "https://example.test/resource/404"}
        analyzer = FakeDeepAnalyzer()

        record = evaluator.evaluate_case(ghost, analyzer, db_path=self.db_path)

        self.assertEqual(analyzer.tender_contexts, [])
        self.assertEqual(record["error"]["kind"], evaluator.SKIP_MISSING)

    def test_deep_analysis_error_is_recorded_with_usage(self):
        (case,) = self.make_golden(RELEVANT)
        error = openai_deep_analysis.DeepAnalysisError(
            openai_deep_analysis.KIND_VALIDATION, "bad result",
            usage=dict(USAGE), response_id="resp_9", attempts=1,
        )
        analyzer = FakeDeepAnalyzer(default=error)

        record = evaluator.evaluate_case(case, analyzer, db_path=self.db_path)

        self.assertEqual(record["status"], evaluator.RECORD_ERROR)
        self.assertEqual(record["error"], {"kind": "validation", "message": "bad result"})
        self.assertEqual(record["usage"], USAGE)
        self.assertEqual(record["response_id"], "resp_9")

    def test_unexpected_exception_does_not_propagate(self):
        (case,) = self.make_golden(RELEVANT)
        analyzer = FakeDeepAnalyzer(default=RuntimeError("boom"))

        record = evaluator.evaluate_case(case, analyzer, db_path=self.db_path)

        self.assertEqual(record["status"], evaluator.RECORD_ERROR)
        self.assertEqual(record["error"]["kind"], evaluator.ERROR_UNEXPECTED)

    def test_expected_labels_not_leaked_into_analyzer_context(self):
        (case,) = self.make_golden(make_expected(expected_reason="СЕКРЕТНЫЙ-ЭТАЛОН", notes="СЕКРЕТНАЯ-ЗАМЕТКА"))
        analyzer = FakeDeepAnalyzer()

        evaluator.evaluate_case(case, analyzer, db_path=self.db_path)

        (tender_context,) = analyzer.tender_contexts
        text = json.dumps(tender_context, ensure_ascii=False)
        for secret in ("СЕКРЕТНЫЙ-ЭТАЛОН", "СЕКРЕТНАЯ-ЗАМЕТКА"):
            self.assertNotIn(secret, text)


class DryRunTests(EvaluatorTestCase):
    def real_analyzer(self, environ):
        return openai_deep_analysis.OpenAIDeepAnalysisAnalyzer(environ=environ)

    def test_dry_run_makes_no_calls_and_writes_nothing(self):
        self.make_golden(RELEVANT, NOT_RELEVANT, MAYBE_UNCLEAR)
        analyzer = self.real_analyzer({"OPENAI_API_KEY": API_KEY})
        golden_before, db_before = self.golden_path.read_bytes(), self.db_path.read_bytes()

        with mock.patch.object(analyzer, "ensure_client", side_effect=AssertionError("client created")):
            code, output = self.run_main("--dry-run", analyzer=analyzer)

        self.assertEqual(code, 0, output)
        self.assertIn("3 кейс(ов)", output)
        self.assertIn("gpt-5.6-luna", output)
        self.assertIn("high", output)
        self.assertIn("готовых к deep analysis", output)
        self.assertIn("API-вызовов не выполнено", output)
        self.assertNotIn(API_KEY, output)
        self.assertFalse(self.deep_runs_dir.exists())
        self.assertEqual(self.golden_path.read_bytes(), golden_before)
        self.assertEqual(self.db_path.read_bytes(), db_before)

    def test_dry_run_reports_missing_api_key(self):
        self.make_golden(RELEVANT)
        analyzer = self.real_analyzer({})
        code, output = self.run_main("--dry-run", analyzer=analyzer)
        self.assertEqual(code, 1)
        self.assertIn("MISSING", output)

    def test_dry_run_reports_stale_cases(self):
        (case,) = self.make_golden(RELEVANT)
        enrichment_repository.save_enrichment(
            case["resource_url"], make_enrichment(description="Изменён"), self.db_path,
        )
        code, output = self.run_main("--dry-run", analyzer=self.real_analyzer({"OPENAI_API_KEY": API_KEY}))
        self.assertIn("STALE", output)

    def test_invalid_golden_set_is_reported(self):
        self.golden_path.write_text("[{\"case_id\": \"x\"}]", encoding="utf-8")
        code, output = self.run_main("--dry-run", analyzer=FakeDeepAnalyzer())
        self.assertEqual(code, 2)
        self.assertIn("невалиден", output)

    def test_mode_is_required(self):
        with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), mock.patch("sys.stderr", io.StringIO()):
            evaluator.main([], analyzer=FakeDeepAnalyzer())

    def test_no_all_flag_exists(self):
        with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), mock.patch("sys.stderr", io.StringIO()):
            evaluator.main(["--all"], analyzer=FakeDeepAnalyzer())


class CliRunTests(EvaluatorTestCase):
    def test_case_id_runs_only_that_case_and_saves_artifact(self):
        cases = self.make_golden(RELEVANT, NOT_RELEVANT)
        analyzer = FakeDeepAnalyzer()
        golden_before, db_before = self.golden_path.read_bytes(), self.db_path.read_bytes()

        code, output = self.run_main("--case-id", cases[0]["case_id"], analyzer=analyzer)

        self.assertEqual(code, 0, output)
        self.assertEqual(len(analyzer.tender_contexts), 1)
        self.assertIn("deep prediction:", output)
        self.assertIn("manual_review_required", output)
        self.assertIn("missing_information", output)
        self.assertIn("participation_barriers", output)
        self.assertIn("document_coverage", output)
        self.assertEqual(self.golden_path.read_bytes(), golden_before)
        self.assertEqual(self.db_path.read_bytes(), db_before)

        (artifact_path,) = list(self.deep_runs_dir.iterdir())
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        self.assertEqual(artifact["evaluator_version"], evaluator.EVALUATOR_VERSION)
        self.assertEqual(artifact["model"], "fake-model")
        self.assertEqual(artifact["reasoning_effort"], "high")
        self.assertEqual(artifact["case"]["case_id"], cases[0]["case_id"])
        self.assertEqual(artifact["case"]["status"], "scored")
        # legacy fake usage без cache_write_tokens -> 0; usage/cost единообразно на верхнем уровне
        self.assertEqual(artifact["usage"]["cache_write_tokens"], 0)
        self.assertEqual(artifact["usage"]["input_tokens"], 1000)
        # у fake-model нет цены: стоимость не выдумывается (None + причина), evaluator не падает
        self.assertIsNone(artifact["cost"]["estimated_cost_usd"])
        self.assertIn("fake-model", artifact["cost"]["error"])

    def test_unknown_case_id_makes_no_calls(self):
        self.make_golden(RELEVANT)
        analyzer = FakeDeepAnalyzer()
        code, output = self.run_main("--case-id", "case-nope", analyzer=analyzer)
        self.assertEqual(code, 2)
        self.assertEqual(analyzer.tender_contexts, [])
        self.assertFalse(self.deep_runs_dir.exists())

    def test_missing_api_key_stops_before_any_call_or_artifact(self):
        cases = self.make_golden(RELEVANT)
        analyzer = openai_deep_analysis.OpenAIDeepAnalysisAnalyzer(environ={})
        with mock.patch("openai.OpenAI") as client_class:
            code, output = self.run_main("--case-id", cases[0]["case_id"], analyzer=analyzer)
        self.assertEqual(code, 2)
        self.assertIn("OPENAI_API_KEY", output)
        client_class.assert_not_called()
        self.assertFalse(self.deep_runs_dir.exists())

    def test_not_relevant_case_id_makes_no_api_call(self):
        cases = self.make_golden(NOT_RELEVANT)
        analyzer = FakeDeepAnalyzer()
        code, output = self.run_main("--case-id", cases[0]["case_id"], analyzer=analyzer)
        self.assertEqual(code, 1)
        self.assertEqual(analyzer.tender_contexts, [])
        self.assertIn(evaluator.SKIP_NOT_ELIGIBLE, output)

    def test_artifact_never_contains_api_key(self):
        (case,) = self.make_golden(RELEVANT)
        analyzer = FakeDeepAnalyzer()
        code, output = self.run_main("--case-id", case["case_id"], analyzer=analyzer)
        self.assertEqual(code, 0, output)
        (artifact_path,) = list(self.deep_runs_dir.iterdir())
        self.assertNotIn(API_KEY, artifact_path.read_text(encoding="utf-8"))
        self.assertNotIn(API_KEY, output)

    def test_artifact_is_not_overwritten(self):
        (case,) = self.make_golden(RELEVANT)
        record = evaluator.evaluate_case(case, FakeDeepAnalyzer(), db_path=self.db_path)
        artifact = evaluator.build_artifact(
            record, FakeDeepAnalyzer(), self.golden_path, evaluator._timestamp_now(),
        )
        evaluator.save_artifact(artifact, self.deep_runs_dir)
        with self.assertRaises(FileExistsError):
            evaluator.save_artifact(artifact, self.deep_runs_dir)


class MaterializedArtifactTests(EvaluatorTestCase):
    """Реальный OpenAIDeepAnalysisAnalyzer с fake client: artifact хранит materialized evidence."""

    def run_with_ids(self, evidence_ids):
        from tests.test_openai_deep_analysis import FakeClient, make_response, payload

        (case,) = self.make_golden(RELEVANT)
        raw = payload(evidence_ids=evidence_ids, procurement=procurement_block(), category="computer_equipment")
        client = FakeClient(make_response(raw))
        analyzer = openai_deep_analysis.OpenAIDeepAnalysisAnalyzer(client=client, environ={}, sleep=lambda _: None)
        code, output = self.run_main("--case-id", case["case_id"], analyzer=analyzer)
        return code, output, client, raw

    def test_artifact_contains_materialized_exact_evidence_and_raw_ids(self):
        code, output, client, raw = self.run_with_ids(["ev_ann_title_0"])
        self.assertEqual(code, 0, output)
        self.assertEqual(len(client.responses.calls), 1)
        (path,) = self.deep_runs_dir.glob("*.json")
        artifact = json.loads(path.read_text(encoding="utf-8"))
        case = artifact["case"]
        self.assertEqual(artifact["prompt_version"], "procurement-deep-v6")
        (evidence,) = case["prediction"]["evidence"]
        self.assertEqual(evidence["evidence_id"], "ev_ann_title_0")
        self.assertEqual(evidence["source_type"], "announcement")
        self.assertTrue(evidence["text"])  # точный заголовок тендера, взятый из каталога
        self.assertEqual(case["raw_model_output"]["evidence_ids"], ["ev_ann_title_0"])
        self.assertNotIn("evidence", case["raw_model_output"])
        self.assertGreater(case["evidence_units"], 0)
        self.assertNotIn("evidence_catalog", json.dumps(artifact))

    def test_unknown_id_is_recorded_as_validation_error(self):
        code, output, _, _ = self.run_with_ids(["ev_doc_1_ffffff_0000"])
        self.assertEqual(code, 1)
        (path,) = self.deep_runs_dir.glob("*.json")
        case = json.loads(path.read_text(encoding="utf-8"))["case"]
        self.assertEqual(case["status"], evaluator.RECORD_ERROR)
        self.assertEqual(case["error"]["kind"], "validation")
        self.assertIsNone(case["prediction"])


class SafetyTests(unittest.TestCase):
    def test_production_db_stays_blocked(self):
        case = {
            "case_id": "case-x", "resource_url": "https://example.test/x", "title": "x",
            "input_hash": "a" * 64, "expected": make_expected(),
        }
        with safety_guards.expect_violation(self, safety_guards.DATABASE_MESSAGE):
            evaluator.prepare_case(case, db_path=safety_guards.PRODUCTION_DB_PATH)

    @unittest.skipUnless(safety_guards.PRODUCTION_DB_PATH.exists(), "нет локальной production БД для копирования")
    def test_dry_run_on_real_golden_set_and_db_copy_reaches_read_path_and_mutates_nothing(self):
        """
        Настоящий golden set + КОПИЯ production БД (файловое копирование, sqlite к оригиналу не
        подключается: это запрещено защитой тестов). Проверяем, что dry-run дошёл до конца (hydrate
        по БД + итоговая строка), а не упал раньше на выводе армянского текста в кодировку консоли.
        """
        golden_path = golden_set.DEFAULT_GOLDEN_SET_PATH
        self.assertTrue(golden_path.exists(), "в репозитории должен быть golden set")
        with tempfile.TemporaryDirectory() as tmp:
            db_copy = Path(tmp) / "tenders_copy.db"
            shutil.copyfile(safety_guards.PRODUCTION_DB_PATH, db_copy)
            hashes_before = {
                "production_db": _sha256(safety_guards.PRODUCTION_DB_PATH),
                "db_copy": _sha256(db_copy),
                "golden_set": _sha256(golden_path),
            }
            violations_before = safety_guards.violations()
            analyzer = openai_deep_analysis.OpenAIDeepAnalysisAnalyzer(environ={"OPENAI_API_KEY": API_KEY})
            # UTF-8, а не консольная cp1252: армянский текст не должен ронять вывод.
            raw = io.BytesIO()
            stdout = io.TextIOWrapper(raw, encoding="utf-8", write_through=True)
            with mock.patch.object(analyzer, "ensure_client", side_effect=AssertionError("client created")),                     mock.patch("openai.OpenAI", side_effect=AssertionError("OpenAI client created")),                     redirect_stdout(stdout):
                code = evaluator.main(["--dry-run", "--db-path", str(db_copy)], analyzer=analyzer)
            output = raw.getvalue().decode("utf-8")

            self.assertEqual(code, 0, output)
            # дошли до конца: golden set прочитан, БД прочитана (hydrate), конфиг проверен, итог напечатан
            self.assertIn("кейс(ов), структура валидна", output)
            self.assertIn("Hydrate/hash/eligibility:", output)
            self.assertIn("API-вызовов не выполнено, client не создавался", output)
            self.assertNotIn(API_KEY, output)
            self.assertEqual({
                "production_db": _sha256(safety_guards.PRODUCTION_DB_PATH),
                "db_copy": _sha256(db_copy),
                "golden_set": _sha256(golden_path),
            }, hashes_before)
            # ни сети, ни production БД: список нарушений защиты не вырос
            self.assertEqual(safety_guards.violations(), violations_before)

    def test_dry_run_survives_non_ascii_output_and_reaches_db(self):
        """Hermetic-вариант: армянский заголовок в golden set, вывод в UTF-8, БД (пустая, временная) читается."""
        with tempfile.TemporaryDirectory() as tmp:
            case = golden_set.load_golden_set(golden_set.DEFAULT_GOLDEN_SET_PATH)[0]
            case["title"] = "Համակարգիչների մատակարարում"
            path = Path(tmp) / "golden.json"
            path.write_text(json.dumps([case], ensure_ascii=False), encoding="utf-8")
            raw = io.BytesIO()
            stdout = io.TextIOWrapper(raw, encoding="utf-8", write_through=True)
            analyzer = openai_deep_analysis.OpenAIDeepAnalysisAnalyzer(environ={"OPENAI_API_KEY": API_KEY})
            missing_db = Path(tmp) / "empty.db"
            with redirect_stdout(stdout):
                code = evaluator.main(
                    ["--dry-run", "--path", str(path), "--db-path", str(missing_db)], analyzer=analyzer,
                )
            output = raw.getvalue().decode("utf-8")
            self.assertIn("Hydrate/hash/eligibility:", output, output)
            self.assertIn("API-вызовов не выполнено", output)
            self.assertIn(code, (0, 1))

    def test_no_all_mode_in_arg_parser(self):
        parser = evaluator._build_arg_parser()
        option_strings = {option for action in parser._actions for option in action.option_strings}
        self.assertNotIn("--all", option_strings)


if __name__ == "__main__":
    unittest.main()
