"""
Тесты src.ai.relevance_pipeline на временной SQLite БД с fake analyzer (см. правило
задачи: production FakeAnalyzer не создаётся, fake analyzer — только в tests).
Реального AI/HTTP здесь нет.

Запуск из корня проекта:
    python -m unittest tests.test_relevance_pipeline -v
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.ai import relevance_pipeline as pipeline
from src.database import analysis_repository, announcement_repository, enrichment_repository

LOGISTICS_FIELDS = (
    "service", "cargo", "origin", "destination", "transport_mode", "weight",
    "volume", "frequency", "customs_requirements", "insurance_requirements", "special_conditions",
)
PROCUREMENT_SCALAR_FIELDS = (
    "subject", "quantity_summary", "brand_or_equivalent", "delivery_location",
    "delivery_deadline", "warranty", "estimated_value_amd",
)
PROCUREMENT_LIST_FIELDS = ("items", "lots", "technical_requirements", "country_of_origin_requirements", "certifications")


def make_announcement(number, **overrides) -> dict:
    announcement = {
        "title": f"Тендер {number}",
        "source_section": "open_competition",
        "source_section_name": "Открытый конкурс",
        "source_page_url": "https://gnumner.minfin.am/ru/page/test/",
        "resource_url": f"https://example.test/resource/{number}",
        "resource_type": "armeps_documents_page",
    }
    announcement.update(overrides)
    return announcement


def make_enrichment(**overrides) -> dict:
    enrichment = {"enrichment_status": "success", "description": "Тендер", "documents": []}
    enrichment.update(overrides)
    return enrichment


class FakeAnalyzer:
    """
    Test-only analyzer: triage_by_url / deep_by_url — {resource_url: raw_result}. Без
    записи о конкретном URL возвращает default_triage / default_deep. side_effect можно
    задать per-call через очередь (queue), как unittest.mock side_effect.
    """

    def __init__(self, triage_by_url=None, deep_by_url=None, default_triage=None, default_deep=None):
        self.triage_by_url = triage_by_url or {}
        self.deep_by_url = deep_by_url or {}
        self.default_triage = default_triage
        self.default_deep = default_deep
        self.triage_calls = []
        self.deep_calls = []

    def triage(self, triage_context: dict) -> dict:
        resource_url = triage_context["resource_url"]
        self.triage_calls.append(resource_url)
        result = self.triage_by_url.get(resource_url, self.default_triage)
        if isinstance(result, Exception):
            raise result
        return result

    def deep_analyze(self, tender_context: dict, deep_analysis_context: dict, triage_result: dict) -> dict:
        resource_url = tender_context["resource_url"]
        self.deep_calls.append(resource_url)
        result = self.deep_by_url.get(resource_url, self.default_deep)
        if isinstance(result, Exception):
            raise result
        return result


def not_relevant_triage(**overrides) -> dict:
    result = {
        "relevance_status": "not_relevant", "opportunity_type": "unrelated", "category": None,
        "confidence": "high", "reason": "Строительные работы без поставки товара",
        "requires_deep_analysis": False, "evidence": [],
    }
    result.update(overrides)
    return result


def relevant_logistics_triage(**overrides) -> dict:
    result = {
        "relevance_status": "relevant", "opportunity_type": "logistics", "category": "international_freight",
        "confidence": "high", "reason": "Международная перевозка груза",
        "requires_deep_analysis": True, "evidence": [],
    }
    result.update(overrides)
    return result


def maybe_procurement_triage(**overrides) -> dict:
    result = {
        "relevance_status": "maybe", "opportunity_type": "procurement", "category": "electronics",
        "confidence": "medium", "reason": "Закупка компьютерной техники, требуется дилерская авторизация",
        "requires_deep_analysis": True, "evidence": [],
    }
    result.update(overrides)
    return result


def logistics_deep(**overrides) -> dict:
    result = {
        "summary": "Международная перевозка груза автомобильным транспортом",
        "opportunity_type": "logistics", "category": "international_freight",
        "why_interesting": "Логистическая услуга", "participation_barriers": [],
        "missing_information": [], "manual_review_required": False, "evidence": [],
        "procurement": None,
        "logistics": {name: None for name in LOGISTICS_FIELDS} | {"service": "Международная перевозка"},
    }
    result.update(overrides)
    return result


def procurement_deep(**overrides) -> dict:
    block = {name: None for name in PROCUREMENT_SCALAR_FIELDS} | {name: [] for name in PROCUREMENT_LIST_FIELDS}
    block["subject"] = "Поставка компьютерной техники"
    result = {
        "summary": "Закупка компьютерной техники с последующим импортом",
        "opportunity_type": "procurement", "category": "electronics",
        "why_interesting": "Возможна закупка у зарубежного поставщика",
        "participation_barriers": ["official_dealer_required"],
        "missing_information": [], "manual_review_required": True, "evidence": [],
        "procurement": block, "logistics": None,
    }
    result.update(overrides)
    return result


class RelevancePipelineTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "test.db"
        analysis_repository.init_db(self.db_path)

    def add(self, number, **enrichment_overrides) -> str:
        announcement = make_announcement(number)
        announcement_repository.save_announcement(announcement, self.db_path)
        enrichment_repository.save_enrichment(
            announcement["resource_url"], make_enrichment(**enrichment_overrides), self.db_path,
        )
        return announcement["resource_url"]


class TriageStageTests(RelevancePipelineTestCase):
    def test_relevant_logistics_is_saved(self):
        url = self.add(1)
        analyzer = FakeAnalyzer(default_triage=relevant_logistics_triage())

        summary = pipeline.run_triage(analyzer, db_path=self.db_path)

        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["relevant_count"], 1)
        row = analysis_repository.get_triage(url, db_path=self.db_path)
        self.assertEqual(row["relevance_status"], "relevant")
        self.assertEqual(row["opportunity_type"], "logistics")

    def test_maybe_procurement_is_saved(self):
        url = self.add(1)
        analyzer = FakeAnalyzer(default_triage=maybe_procurement_triage())

        summary = pipeline.run_triage(analyzer, db_path=self.db_path)

        self.assertEqual(summary["maybe_count"], 1)
        row = analysis_repository.get_triage(url, db_path=self.db_path)
        self.assertEqual(row["relevance_status"], "maybe")
        self.assertEqual(row["opportunity_type"], "procurement")

    def test_not_relevant_is_saved(self):
        url = self.add(1)
        analyzer = FakeAnalyzer(default_triage=not_relevant_triage())

        summary = pipeline.run_triage(analyzer, db_path=self.db_path)

        self.assertEqual(summary["not_relevant_count"], 1)
        self.assertFalse(analysis_repository.get_triage(url, db_path=self.db_path)["requires_deep_analysis"])

    def test_invalid_result_fails_safely(self):
        self.add(1)
        analyzer = FakeAnalyzer(default_triage={"relevance_status": "definitely"})

        summary = pipeline.run_triage(analyzer, db_path=self.db_path)

        self.assertEqual(summary["success_count"], 0)
        self.assertEqual(summary["failed_count"], 1)
        self.assertEqual(analysis_repository.count_triage(self.db_path), 0)

    def test_one_failure_does_not_stop_batch(self):
        url1 = self.add(1)
        url2 = self.add(2)
        analyzer = FakeAnalyzer(triage_by_url={
            url1: RuntimeError("analyzer down"),
            url2: relevant_logistics_triage(),
        })

        summary = pipeline.run_triage(analyzer, db_path=self.db_path)

        self.assertEqual(summary["candidate_count"], 2)
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["failed_count"], 1)
        self.assertEqual(summary["failures"][0]["resource_url"], url1)
        self.assertEqual(summary["failures"][0]["error_type"], "RuntimeError")
        self.assertIsNotNone(analysis_repository.get_triage(url2, db_path=self.db_path))
        self.assertIsNone(analysis_repository.get_triage(url1, db_path=self.db_path))

    def test_provider_and_model_are_stored(self):
        url = self.add(1)
        analyzer = FakeAnalyzer(default_triage=relevant_logistics_triage())

        pipeline.run_triage(analyzer, db_path=self.db_path, provider="openai", model="gpt-x", prompt_version="v2")

        row = analysis_repository.get_triage(url, db_path=self.db_path)
        self.assertEqual(row["provider"], "openai")
        self.assertEqual(row["model"], "gpt-x")
        self.assertEqual(row["prompt_version"], "v2")

    def test_analyzer_receives_compact_triage_context_not_full_context(self):
        self.add(1)
        analyzer = FakeAnalyzer(default_triage=relevant_logistics_triage())

        pipeline.run_triage(analyzer, db_path=self.db_path)

        self.assertEqual(len(analyzer.triage_calls), 1)

    def test_no_candidates_calls_analyzer_zero_times(self):
        analyzer = FakeAnalyzer(default_triage=relevant_logistics_triage())
        summary = pipeline.run_triage(analyzer, db_path=self.db_path)
        self.assertEqual(summary["candidate_count"], 0)
        self.assertEqual(analyzer.triage_calls, [])

    def test_rerun_with_unchanged_context_is_existing(self):
        self.add(1)
        analyzer = FakeAnalyzer(default_triage=relevant_logistics_triage())
        pipeline.run_triage(analyzer, db_path=self.db_path)

        summary = pipeline.run_triage(analyzer, db_path=self.db_path)

        self.assertEqual(summary["candidate_count"], 0)


class DeepAnalysisStageTests(RelevancePipelineTestCase):
    def run_triage_first(self, url, triage_result):
        analyzer = FakeAnalyzer(triage_by_url={url: triage_result})
        pipeline.run_triage(analyzer, db_path=self.db_path)

    def test_not_relevant_skips_deep_analysis(self):
        url = self.add(1)
        self.run_triage_first(url, not_relevant_triage())
        analyzer = FakeAnalyzer(default_deep=logistics_deep())

        summary = pipeline.run_deep_analysis(analyzer, db_path=self.db_path)

        self.assertEqual(summary["candidate_count"], 0)
        self.assertEqual(analyzer.deep_calls, [])
        self.assertIsNone(analysis_repository.get_deep_analysis(url, db_path=self.db_path))

    def test_relevant_logistics_deep_analysis_is_saved(self):
        url = self.add(1)
        self.run_triage_first(url, relevant_logistics_triage())
        analyzer = FakeAnalyzer(default_deep=logistics_deep())

        summary = pipeline.run_deep_analysis(analyzer, db_path=self.db_path)

        self.assertEqual(summary["success_count"], 1)
        row = analysis_repository.get_deep_analysis(url, db_path=self.db_path)
        self.assertEqual(row["result"]["opportunity_type"], "logistics")
        self.assertIsNotNone(row["result"]["logistics"])
        self.assertIsNone(row["result"]["procurement"])

    def test_maybe_procurement_deep_analysis_is_saved(self):
        url = self.add(1)
        self.run_triage_first(url, maybe_procurement_triage())
        analyzer = FakeAnalyzer(default_deep=procurement_deep())

        summary = pipeline.run_deep_analysis(analyzer, db_path=self.db_path)

        self.assertEqual(summary["success_count"], 1)
        row = analysis_repository.get_deep_analysis(url, db_path=self.db_path)
        self.assertEqual(row["result"]["opportunity_type"], "procurement")
        self.assertIsNotNone(row["result"]["procurement"])

    def test_analyzer_receives_triage_result(self):
        url = self.add(1)
        self.run_triage_first(url, maybe_procurement_triage())
        received = {}

        class RecordingAnalyzer(FakeAnalyzer):
            def deep_analyze(self, tender_context, deep_analysis_context, triage_result):
                received["triage_result"] = triage_result
                return procurement_deep()

        pipeline.run_deep_analysis(RecordingAnalyzer(), db_path=self.db_path)

        self.assertEqual(received["triage_result"], maybe_procurement_triage())

    def test_invalid_deep_result_fails_safely(self):
        url = self.add(1)
        self.run_triage_first(url, relevant_logistics_triage())
        analyzer = FakeAnalyzer(default_deep={"summary": "x"})

        summary = pipeline.run_deep_analysis(analyzer, db_path=self.db_path)

        self.assertEqual(summary["failed_count"], 1)
        self.assertIsNone(analysis_repository.get_deep_analysis(url, db_path=self.db_path))

    def test_one_failure_does_not_stop_batch(self):
        url1 = self.add(1)
        url2 = self.add(2)
        self.run_triage_first(url1, relevant_logistics_triage())
        self.run_triage_first(url2, maybe_procurement_triage())
        analyzer = FakeAnalyzer(deep_by_url={
            url1: RuntimeError("analyzer down"),
            url2: procurement_deep(),
        })

        summary = pipeline.run_deep_analysis(analyzer, db_path=self.db_path)

        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["failed_count"], 1)
        self.assertIsNotNone(analysis_repository.get_deep_analysis(url2, db_path=self.db_path))
        self.assertIsNone(analysis_repository.get_deep_analysis(url1, db_path=self.db_path))


class FullPipelineTests(RelevancePipelineTestCase):
    def test_run_relevance_pipeline_runs_both_stages(self):
        url = self.add(1)
        analyzer = FakeAnalyzer(
            triage_by_url={url: relevant_logistics_triage()},
            deep_by_url={url: logistics_deep()},
        )

        result = pipeline.run_relevance_pipeline(analyzer, db_path=self.db_path)

        self.assertEqual(result["triage"]["success_count"], 1)
        self.assertEqual(result["deep_analysis"]["success_count"], 1)
        self.assertIsNotNone(analysis_repository.get_triage(url, db_path=self.db_path))
        self.assertIsNotNone(analysis_repository.get_deep_analysis(url, db_path=self.db_path))

    def test_not_relevant_never_reaches_deep_analysis_in_full_run(self):
        url = self.add(1)
        analyzer = FakeAnalyzer(
            triage_by_url={url: not_relevant_triage()},
            default_deep=logistics_deep(),
        )

        pipeline.run_relevance_pipeline(analyzer, db_path=self.db_path)

        self.assertEqual(analyzer.deep_calls, [])


class MainTests(unittest.TestCase):
    def test_main_only_prints_message(self):
        with mock.patch("builtins.print") as fake_print:
            pipeline.main()
        fake_print.assert_called_once()


if __name__ == "__main__":
    unittest.main()
