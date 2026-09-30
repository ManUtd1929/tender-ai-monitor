"""
Тесты src.ai.analysis_pipeline на временной SQLite БД с fake analyzers. Реального OpenAI, сети и
production БД здесь нет: fake-анализаторы наследуют настоящие (build_request/model/prompt_version для
preflight реальные), но triage_with_metadata / deep_analyze_with_metadata заменены скриптами, а
openai.OpenAI на время тестов запрещён (создание client = failure).

Запуск из корня проекта:
    python -m unittest tests.test_analysis_pipeline -v
"""

import hashlib
import io
import sqlite3
import tempfile
import unittest
from contextlib import closing
from decimal import Decimal
from pathlib import Path
from unittest import mock

from src.ai import analysis_pipeline, commercial_gate, escalation, openai_deep_analysis, openai_triage
from src.ai.budget_settings import BudgetSettings
from src.database import (
    ai_usage_repository, analysis_repository, announcement_repository, document_repository, enrichment_repository,
    pipeline_state_repository as state_repo,
)
from tests import safety_guards

USAGE = {
    "input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500,
    "cached_tokens": 200, "cache_write_tokens": 100, "reasoning_tokens": 120,
}


def triage_result(status="relevant", **overrides) -> dict:
    result = {
        "relevance_status": status,
        "opportunity_type": {"relevant": "procurement", "maybe": "unclear", "not_relevant": "unrelated"}[status],
        "category": {"relevant": "electronics", "maybe": None, "not_relevant": None}[status],
        "confidence": "high", "reason": f"reason-{status}", "requires_deep_analysis": status != "not_relevant",
        "evidence": [],
    }
    result.update(overrides)
    return result


def deep_result(**overrides) -> dict:
    result = {"summary": "s", "opportunity_type": "procurement", "category": "electronics",
              "confidence": "high", "manual_review_required": False}
    result.update(overrides)
    return result


class FakeTriage(openai_triage.OpenAITriageAnalyzer):
    """outcomes: {resource_url: result | Exception}; default — default_outcome."""

    def __init__(self, default_outcome=None, usage=USAGE):
        super().__init__(environ={})
        self.outcomes = {}
        self.default_outcome = default_outcome if default_outcome is not None else triage_result("relevant")
        self.usage = usage
        self.calls = []

    def triage_with_metadata(self, triage_context):
        url = triage_context["resource_url"]
        self.calls.append(url)
        outcome = self.outcomes.get(url, self.default_outcome)
        if isinstance(outcome, Exception):
            raise outcome
        return {"result": outcome, "usage": self.usage, "response_id": f"resp-triage-{len(self.calls)}", "attempts": 1}


class FakeDeep(openai_deep_analysis.OpenAIDeepAnalysisAnalyzer):
    def __init__(self, default_outcome=None, usage=USAGE, response_id=None):
        super().__init__(environ={})
        self.outcomes = {}
        self.default_outcome = default_outcome if default_outcome is not None else deep_result()
        self.usage = usage
        self.fixed_response_id = response_id
        self.calls = []

    def deep_analyze_with_metadata(self, tender_context, deep_analysis_context, triage_result):
        url = tender_context["resource_url"]
        self.calls.append(url)
        outcome = self.outcomes.get(url, self.default_outcome)
        if isinstance(outcome, Exception):
            raise outcome
        return {
            "result": outcome, "raw_model_output": {"evidence_ids": []}, "usage": self.usage,
            "response_id": self.fixed_response_id or f"resp-deep-{len(self.calls)}", "attempts": 1,
        }


def gate_pass(triage, ctx, min_value):
    return {
        "gate_decision": commercial_gate.DEEP_CANDIDATE, "gate_reason": "test pass", "facts_used": {},
        "estimated_value_amd": None, "total_quantity": None, "total_lots": None,
        "category": triage.get("category"), "confidence": "low",
    }


class PipelineTestCase(unittest.TestCase):
    settings = BudgetSettings()

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "test.db"
        state_repo.init_db(self.db_path)
        ai_usage_repository.init_db(self.db_path)
        client = mock.patch("openai.OpenAI", side_effect=AssertionError("OpenAI client must not be created"))
        self.openai_client = client.start()
        self.addCleanup(client.stop)
        self.triage = FakeTriage()
        self.deep = FakeDeep()

    def add(self, number, **enrichment) -> str:
        announcement = {
            "title": f"Тендер {number}", "source_section": "open_competition",
            "source_section_name": "Открытый конкурс", "source_page_url": "https://gnumner.minfin.am/ru/page/test/",
            "resource_url": f"https://example.test/resource/{number}", "resource_type": "armeps_documents_page",
        }
        announcement_repository.save_announcement(announcement, self.db_path)
        data = {"enrichment_status": "success", "description": "Тендер", "documents": []} | enrichment
        enrichment_repository.save_enrichment(announcement["resource_url"], data, self.db_path)
        return announcement["resource_url"]

    def make(self, gate=gate_pass, settings=None, **kwargs):
        return analysis_pipeline.AnalysisPipeline(
            self.triage, self.deep, settings or self.settings, db_path=self.db_path, gate=gate, **kwargs,
        )

    def state(self, url):
        return state_repo.get_state(url, db_path=self.db_path)

    def ledger_rows(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            return conn.execute(
                "SELECT analysis_type, model, response_id, input_tokens, cached_input_tokens, cache_write_tokens, "
                "output_tokens, reasoning_tokens, estimated_cost_usd, success, error_kind FROM ai_usage_events "
                "ORDER BY id"
            ).fetchall()


class RoutingTests(PipelineTestCase):
    def test_relevant_tender_goes_triage_gate_deep_and_is_persisted(self):
        url = self.add(1, estimated_value_amd="5000000")
        outcome = self.make(gate=commercial_gate.evaluate).process_announcement(url)

        self.assertEqual(outcome["state"], state_repo.STATE_DEEP_COMPLETED)
        self.assertEqual(outcome["api_calls"], {"triage": 1, "deep": 1})
        triage_row = analysis_repository.get_triage(url, db_path=self.db_path)
        self.assertEqual(triage_row["relevance_status"], "relevant")
        self.assertEqual(triage_row["model"], self.triage.model)
        deep_row = analysis_repository.get_deep_analysis(url, db_path=self.db_path)
        self.assertEqual(deep_row["result"]["summary"], "s")
        self.assertEqual(deep_row["prompt_version"], self.deep.prompt_version)
        self.assertEqual(deep_row["input_hash"], triage_row["input_hash"])
        state = self.state(url)
        self.assertEqual(state["state"], state_repo.STATE_DEEP_COMPLETED)
        self.assertEqual(state["model"], self.deep.model)
        self.assertEqual(state["details"]["gate"]["gate_decision"], commercial_gate.DEEP_CANDIDATE)
        self.assertEqual(state["details"]["raw_model_output"], {"evidence_ids": []})
        self.assertEqual(state["details"]["response_id"], "resp-deep-1")
        self.assertIsNotNone(state["details"]["estimated_cost_usd"])
        self.assertEqual(state["details"]["admission"]["deep_analysis_status"], "ready_for_deep")

    def test_not_relevant_stops_before_gate_and_deep(self):
        url = self.add(1)
        self.triage.default_outcome = triage_result("not_relevant")
        gate = mock.Mock(side_effect=AssertionError("gate must not be called"))

        outcome = self.make(gate=gate).process_announcement(url)

        self.assertEqual(outcome["state"], state_repo.STATE_STOPPED_NOT_RELEVANT)
        self.assertEqual(outcome["reason_code"], "not_relevant")
        gate.assert_not_called()
        self.assertEqual(self.deep.calls, [])
        self.assertIsNone(analysis_repository.get_deep_analysis(url, db_path=self.db_path))
        self.assertEqual(self.state(url)["message"], "reason-not_relevant")

    def test_maybe_is_not_discarded_and_reaches_deep(self):
        url = self.add(1)
        self.triage.default_outcome = triage_result("maybe")
        seen = []

        def gate(triage, ctx, min_value):
            seen.append(triage["relevance_status"])
            return gate_pass(triage, ctx, min_value)

        outcome = self.make(gate=gate).process_announcement(url)

        self.assertEqual(seen, ["maybe"])
        self.assertEqual(outcome["state"], state_repo.STATE_DEEP_COMPLETED)
        self.assertEqual(self.deep.calls, [url])
        self.assertEqual(analysis_repository.get_triage(url, db_path=self.db_path)["relevance_status"], "maybe")

    def test_gate_skip_keeps_relevance_and_skips_deep(self):
        url = self.add(1, estimated_value_amd="100")
        settings = BudgetSettings(min_deep_value_amd=1000)

        outcome = self.make(gate=commercial_gate.evaluate, settings=settings).process_announcement(url)

        self.assertEqual(outcome["state"], state_repo.STATE_COMMERCIAL_GATE_SKIP)
        self.assertEqual(outcome["reason_code"], commercial_gate.SKIP_LOW_VALUE)
        self.assertEqual(self.deep.calls, [])
        self.assertEqual(analysis_repository.get_triage(url, db_path=self.db_path)["relevance_status"], "relevant")
        state = self.state(url)
        self.assertEqual(state["reason_code"], commercial_gate.SKIP_LOW_VALUE)
        self.assertEqual(state["details"]["gate"]["gate_decision"], commercial_gate.SKIP_LOW_VALUE)
        self.assertEqual(state["details"]["relevance_status"], "relevant")

    def test_without_min_value_gate_does_not_skip(self):
        url = self.add(1, estimated_value_amd="100")
        outcome = self.make(gate=commercial_gate.evaluate).process_announcement(url)
        self.assertEqual(outcome["state"], state_repo.STATE_DEEP_COMPLETED)


class AdmissionTests(PipelineTestCase):
    def test_triage_budget_deferred_makes_no_analyzer_call(self):
        url = self.add(1)
        settings = BudgetSettings(monthly_budget_usd=Decimal("0.0001"), soft_limit_usd=Decimal("0.0001"))

        outcome = self.make(settings=settings).process_announcement(url)

        self.assertEqual(outcome["state"], state_repo.STATE_TRIAGE_DEFERRED_BUDGET)
        self.assertEqual(outcome["reason_code"], escalation.BUDGET_DEFERRED)
        self.assertEqual(self.triage.calls, [])
        self.assertEqual(self.deep.calls, [])
        self.assertIsNone(analysis_repository.get_triage(url, db_path=self.db_path))
        self.assertEqual(self.ledger_rows(), [])
        self.assertFalse(outcome["failed"])

    def test_deep_budget_deferred_keeps_triage_and_skips_deep(self):
        url = self.add(1)
        settings = BudgetSettings(soft_limit_usd=Decimal("0.01"))
        ai_usage_repository.record_usage(
            "triage", "gpt-5.6-luna", {"input_tokens": 5_000_000, "output_tokens": 0}, db_path=self.db_path,
        )
        self.assertGreater(ai_usage_repository.current_month_cost(db_path=self.db_path), Decimal("0.01"))

        outcome = self.make(settings=settings).process_announcement(url)

        self.assertEqual(outcome["state"], state_repo.STATE_DEEP_DEFERRED_BUDGET)
        self.assertEqual(outcome["reason_code"], escalation.BUDGET_DEFERRED)
        self.assertEqual(self.triage.calls, [url])  # soft limit: дешёвый triage разрешён
        self.assertEqual(self.deep.calls, [])
        self.assertIsNotNone(analysis_repository.get_triage(url, db_path=self.db_path))
        self.assertEqual(len(self.ledger_rows()), 2)  # seed + triage; Deep ничего не потратил

    def test_deep_input_limit_exceeded_makes_no_deep_call(self):
        url = self.add(1)
        settings = BudgetSettings(max_deep_input_tokens=1)

        outcome = self.make(settings=settings).process_announcement(url)

        self.assertEqual(outcome["state"], state_repo.STATE_DEEP_DEFERRED_INPUT)
        self.assertEqual(outcome["reason_code"], escalation.DEEP_INPUT_LIMIT_EXCEEDED)
        self.assertEqual(self.deep.calls, [])
        state = self.state(url)
        self.assertEqual(state["details"]["max_deep_input_tokens"], 1)
        self.assertGreater(state["details"]["admission"]["preflight"]["estimated_input_tokens"], 1)

    def test_deferred_deep_retried_without_calls_until_condition_changes(self):
        url = self.add(1)
        self.make(settings=BudgetSettings(max_deep_input_tokens=1)).process_announcement(url)
        self.assertEqual(self.deep.calls, [])

        # тот же лимит: тендер settled и не попадает в batch
        self.assertEqual(self.make(settings=BudgetSettings(max_deep_input_tokens=1)).select_candidates(), [])
        # лимит изменился: снова кандидат, Deep вызывается
        pipeline = self.make()
        self.assertEqual([c["reason"] for c in pipeline.select_candidates()], ["retry_deep"])
        self.assertEqual(pipeline.process_announcement(url)["state"], state_repo.STATE_DEEP_COMPLETED)
        self.assertEqual(self.triage.calls, [url])  # triage не повторялся


class IdempotencyTests(PipelineTestCase):
    def test_unchanged_rerun_makes_no_additional_calls(self):
        url = self.add(1)
        pipeline = self.make()
        pipeline.process_announcement(url)
        rows_before = self.ledger_rows()

        second = pipeline.process_announcement(url)

        self.assertEqual(self.triage.calls, [url])
        self.assertEqual(self.deep.calls, [url])
        self.assertEqual(second["api_calls"], {"triage": 0, "deep": 0})
        self.assertTrue(second["reused"]["triage"] and second["reused"]["deep"])
        self.assertEqual(self.ledger_rows(), rows_before)
        self.assertEqual(self.state(url)["details"]["response_id"], "resp-deep-1")  # raw/usage не затёрты
        self.assertEqual(pipeline.select_candidates(), [])
        # рестарт процесса = новый объект пайплайна на той же БД
        self.assertEqual(self.make().select_candidates(), [])

    def test_changed_input_hash_makes_tender_eligible_again(self):
        url = self.add(1)
        pipeline = self.make()
        pipeline.process_announcement(url)
        enrichment_repository.save_enrichment(
            url, {"enrichment_status": "success", "description": "Изменённое описание", "documents": []}, self.db_path,
        )
        self.assertEqual([c["reason"] for c in pipeline.select_candidates()], ["retry_triage"])

        pipeline.process_announcement(url)

        self.assertEqual(self.triage.calls, [url, url])
        self.assertEqual(self.deep.calls, [url, url])

    def test_triage_prompt_version_change_triggers_reanalysis(self):
        url = self.add(1)
        pipeline = self.make()
        pipeline.process_announcement(url)
        self.triage.prompt_version = "procurement-v-next"

        self.assertEqual(len(pipeline.select_candidates()), 1)
        pipeline.process_announcement(url)

        self.assertEqual(self.triage.calls, [url, url])
        self.assertEqual(analysis_repository.get_triage(url, db_path=self.db_path)["prompt_version"], "procurement-v-next")
        self.assertEqual(self.deep.calls, [url])  # deep-промпт не менялся, hash тот же -> reuse

    def test_deep_prompt_version_change_reruns_only_deep(self):
        url = self.add(1)
        pipeline = self.make()
        pipeline.process_announcement(url)
        self.deep.prompt_version = "procurement-deep-next"

        pipeline.process_announcement(url)

        self.assertEqual(self.triage.calls, [url])
        self.assertEqual(self.deep.calls, [url, url])

    def test_model_change_makes_analysis_eligible_again(self):
        url = self.add(1)
        pipeline = self.make()
        pipeline.process_announcement(url)
        self.deep.model = "gpt-5.6-terra"

        pipeline.process_announcement(url)

        self.assertEqual(self.deep.calls, [url, url])

    def test_not_relevant_is_settled(self):
        url = self.add(1)
        self.triage.default_outcome = triage_result("not_relevant")
        pipeline = self.make()
        pipeline.process_announcement(url)
        self.assertEqual(pipeline.select_candidates(), [])
        pipeline.process_announcement(url)
        self.assertEqual(self.triage.calls, [url])


class ErrorTests(PipelineTestCase):
    def test_one_tender_error_does_not_stop_batch(self):
        first, second = self.add(1), self.add(2)
        def gate(triage, ctx, min_value):  # ошибка до платного Deep-вызова: резерва нет, batch продолжается
            if ctx["resource_url"] == first:
                raise RuntimeError("boom")
            return gate_pass(triage, ctx, min_value)

        summary = self.make(gate=gate).process_batch()

        self.assertEqual(summary["processed_count"], 2)
        self.assertEqual(summary["failed_count"], 1)
        self.assertEqual(summary["failures"][0]["resource_url"], first)
        self.assertEqual(summary["failures"][0]["error_type"], "RuntimeError")
        self.assertEqual(self.state(first)["state"], state_repo.STATE_DEEP_ERROR)
        self.assertEqual(self.state(first)["reason_code"], analysis_pipeline.REASON_UNEXPECTED_ERROR)
        self.assertEqual(self.state(second)["state"], state_repo.STATE_DEEP_COMPLETED)

    def test_triage_api_error_persists_error_state(self):
        url = self.add(1)
        self.triage.outcomes[url] = openai_triage.TriageError(openai_triage.KIND_TIMEOUT, "timeout", attempts=3)

        outcome = self.make().process_announcement(url)

        self.assertEqual(outcome["state"], state_repo.STATE_TRIAGE_ERROR)
        self.assertTrue(outcome["failed"])
        state = self.state(url)
        self.assertEqual((state["reason_code"], state["error_kind"]), ("timeout", "timeout"))
        self.assertIsNone(analysis_repository.get_triage(url, db_path=self.db_path))
        self.assertEqual(self.deep.calls, [])
        self.assertEqual(self.ledger_rows(), [])  # usage нет — ничего не выдумываем

    def test_triage_transient_error_is_retried_but_validation_error_is_not(self):
        url = self.add(1)
        self.triage.outcomes[url] = openai_triage.TriageError(openai_triage.KIND_RATE_LIMIT, "429")
        pipeline = self.make()
        pipeline.process_announcement(url)
        self.assertEqual([c["reason"] for c in pipeline.select_candidates()], ["retry_triage"])

        self.triage.outcomes[url] = openai_triage.TriageError(
            openai_triage.KIND_VALIDATION, "bad", usage=USAGE, response_id="r-bad",
        )
        pipeline.process_announcement(url)
        self.assertEqual(pipeline.select_candidates(), [])
        pipeline.process_announcement(url)  # прямой повторный вызов тоже не платит
        self.assertEqual(self.triage.calls, [url, url])

    def test_deep_api_error_keeps_triage_and_records_escalation_candidate(self):
        url = self.add(1)
        self.deep.outcomes[url] = openai_deep_analysis.DeepAnalysisError(
            openai_triage.KIND_API_ERROR, "500", usage=USAGE, response_id="r-fail",
        )

        outcome = self.make().process_announcement(url)

        self.assertEqual(outcome["state"], state_repo.STATE_ESCALATION_CANDIDATE)
        self.assertEqual(outcome["escalation_reason"], escalation.API_FAILURE)
        state = self.state(url)
        self.assertEqual(state["escalation_reason"], escalation.API_FAILURE)
        self.assertEqual(state["details"]["gate"]["gate_decision"], commercial_gate.DEEP_CANDIDATE)
        self.assertEqual(analysis_repository.get_triage(url, db_path=self.db_path)["relevance_status"], "relevant")
        self.assertIsNone(analysis_repository.get_deep_analysis(url, db_path=self.db_path))
        self.assertEqual(self.deep.calls, [url])  # никакого второго (Terra) вызова
        self.assertEqual([row[9] for row in self.ledger_rows() if row[0] == "deep"], [0])  # success=0
        self.assertEqual(self.ledger_rows()[-1][10], "api_error")

    def test_validation_failure_is_escalation_candidate_without_terra_call(self):
        url = self.add(1)
        self.deep.outcomes[url] = openai_deep_analysis.DeepAnalysisError(
            openai_triage.KIND_VALIDATION, "bad evidence", usage=USAGE, response_id="r-val",
        )
        pipeline = self.make()

        outcome = pipeline.process_announcement(url)
        pipeline.process_announcement(url)

        self.assertEqual(outcome["escalation_reason"], escalation.VALIDATION_FAILURE)
        self.assertEqual(self.deep.calls, [url])  # ни Terra, ни повторного Luna с тем же входом
        self.assertEqual(pipeline.select_candidates(), [])
        self.openai_client.assert_not_called()

    def test_deep_transport_error_is_retried_on_next_run(self):
        url = self.add(1)
        self.deep.outcomes[url] = openai_deep_analysis.DeepAnalysisError(openai_triage.KIND_RATE_LIMIT, "429")
        pipeline = self.make()
        pipeline.process_announcement(url)
        self.deep.outcomes.pop(url)

        outcome = pipeline.process_announcement(url)

        self.assertEqual(outcome["state"], state_repo.STATE_DEEP_COMPLETED)
        self.assertEqual(self.deep.calls, [url, url])

    def test_config_error_is_not_an_escalation_candidate(self):
        url = self.add(1)
        self.deep.outcomes[url] = openai_deep_analysis.DeepAnalysisError(openai_triage.KIND_CONFIG, "no key")
        outcome = self.make().process_announcement(url)
        self.assertEqual(outcome["state"], state_repo.STATE_DEEP_ERROR)
        self.assertIsNone(outcome["escalation_reason"])

    def test_manual_review_required_alone_is_not_escalation(self):
        url = self.add(1)
        self.deep.default_outcome = deep_result(manual_review_required=True)
        outcome = self.make().process_announcement(url)
        self.assertEqual(outcome["state"], state_repo.STATE_DEEP_COMPLETED)
        self.assertIsNone(self.state(url)["escalation_reason"])
        self.assertEqual(self.deep.calls, [url])

    def test_medium_confidence_alone_is_not_escalation(self):
        url = self.add(1)
        self.triage.default_outcome = triage_result("relevant", confidence="medium")
        self.deep.default_outcome = deep_result(confidence="medium")
        outcome = self.make().process_announcement(url)
        self.assertEqual(outcome["state"], state_repo.STATE_DEEP_COMPLETED)
        self.assertIsNone(self.state(url)["escalation_reason"])
        self.assertEqual(self.deep.calls, [url])

    def test_escalation_mapping_is_explicit(self):
        self.assertFalse(escalation.is_escalation_reason("manual_review_required"))
        self.assertIsNone(escalation.reason_for_error_kind("config"))
        self.assertIsNone(escalation.reason_for_error_kind("authentication"))
        self.assertEqual(escalation.reason_for_error_kind("validation"), escalation.VALIDATION_FAILURE)


class UsageLedgerTests(PipelineTestCase):
    def test_exactly_one_row_per_response_with_cache_fields(self):
        url = self.add(1)
        self.make().process_announcement(url)

        rows = self.ledger_rows()
        self.assertEqual([row[0] for row in rows], ["triage", "deep"])
        self.assertEqual([row[2] for row in rows], ["resp-triage-1", "resp-deep-1"])
        for row in rows:
            self.assertEqual(row[3:8], (1000, 200, 100, 500, 120))  # in, cached, cache_write, out, reasoning
            self.assertEqual(row[9], 1)
            self.assertGreater(Decimal(row[8]), 0)
        self.assertEqual(rows[1][1], self.deep.model)
        self.assertEqual(
            self.state(url)["details"]["estimated_cost_usd"], rows[1][8],
        )

    def test_duplicate_response_id_is_not_recorded_twice(self):
        self.deep.fixed_response_id = "same-response"
        first, second = self.add(1), self.add(2)
        self.make().process_batch()
        self.assertEqual([row[2] for row in self.ledger_rows() if row[0] == "deep"], ["same-response"])
        self.assertEqual(self.deep.calls, [first, second])

    def test_no_usage_means_no_ledger_row(self):
        self.triage.usage = None
        self.deep.usage = None
        self.make().process_announcement(self.add(1))
        self.assertEqual(self.ledger_rows(), [])

    def test_ledger_write_failure_does_not_lose_triage_result(self):
        url = self.add(1)
        with mock.patch.object(ai_usage_repository, "_insert_event", side_effect=sqlite3.OperationalError("locked")):
            outcome = self.make().process_announcement(url)
        # Оплаченный triage сохранён, но Deep не стартовал: учёт сломан (см. AccountingFailClosedTests).
        self.assertIsNotNone(analysis_repository.get_triage(url, db_path=self.db_path))
        self.assertEqual(outcome["state"], analysis_pipeline.OUTCOME_ACCOUNTING_BLOCKED)
        self.assertEqual(self.deep.calls, [])


class AccountingFailClosedTests(PipelineTestCase):
    """Durable pre-call reservation + fail-closed accounting. Только fake analyzers, OpenAI недоступен."""

    def fail_finalization(self):
        # settle_reservation пишет событие через _insert_event: отказ здесь = ledger/finalization не удался
        return mock.patch.object(ai_usage_repository, "_insert_event", side_effect=sqlite3.OperationalError("locked"))

    def reservations(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            return conn.execute(
                "SELECT id, analysis_type, reference, model, prompt_version, input_hash, estimated_cost_usd, status,"
                " actual_cost_usd FROM ai_call_reservations ORDER BY id"
            ).fetchall()

    def blocks(self):
        return ai_usage_repository.unresolved_accounting_blocks(db_path=self.db_path)

    def effective(self):
        return ai_usage_repository.effective_committed_spend(db_path=self.db_path)

    # 1
    def test_reservation_exists_before_analyzer_call(self):
        url = self.add(1)
        seen = []
        original = self.triage.triage_with_metadata

        def spy(triage_context):
            seen.append(ai_usage_repository.outstanding_reservations(db_path=self.db_path))
            return original(triage_context)

        self.triage.triage_with_metadata = spy
        self.make().process_announcement(url)

        (during,) = seen[:1]
        self.assertEqual(len(during), 1)
        self.assertEqual((during[0]["analysis_type"], during[0]["reference"], during[0]["status"]), ("triage", url, "reserved"))
        self.assertEqual(during[0]["model"], self.triage.model)
        self.assertEqual(during[0]["prompt_version"], self.triage.prompt_version)
        self.assertEqual(len(during[0]["input_hash"]), 64)
        self.assertGreater(Decimal(during[0]["estimated_cost_usd"]), 0)

    # 2
    def test_reservation_write_failure_means_no_analyzer_call(self):
        url = self.add(1)
        with mock.patch.object(ai_usage_repository, "reserve_call", side_effect=sqlite3.OperationalError("locked")):
            outcome = self.make().process_announcement(url)
        self.assertEqual(self.triage.calls, [])
        self.assertEqual(outcome["state"], analysis_pipeline.OUTCOME_ACCOUNTING_BLOCKED)
        self.assertEqual(outcome["api_calls"], {"triage": 0, "deep": 0})
        self.assertEqual(self.ledger_rows(), [])

    def test_deep_reservation_write_failure_means_no_deep_call(self):
        url = self.add(1)
        real = ai_usage_repository.reserve_call

        def fail_deep(analysis_type, *args, **kwargs):
            if analysis_type == "deep":
                raise sqlite3.OperationalError("locked")
            return real(analysis_type, *args, **kwargs)

        with mock.patch.object(ai_usage_repository, "reserve_call", side_effect=fail_deep):
            outcome = self.make().process_announcement(url)
        self.assertEqual(self.triage.calls, [url])
        self.assertEqual(self.deep.calls, [])
        self.assertEqual(outcome["state"], analysis_pipeline.OUTCOME_ACCOUNTING_BLOCKED)

    # 3, 12
    def test_success_records_actual_usage_and_settles_reservation(self):
        url = self.add(1)
        outcome = self.make().process_announcement(url)
        self.assertEqual(outcome["state"], state_repo.STATE_DEEP_COMPLETED)
        rows = self.reservations()
        self.assertEqual([(r[1], r[7]) for r in rows], [("triage", "settled"), ("deep", "settled")])
        self.assertEqual(len(self.ledger_rows()), 2)
        self.assertEqual(ai_usage_repository.outstanding_reservations(db_path=self.db_path), [])
        # нет двойного счёта: после settle учитывается только факт
        self.assertEqual(self.effective(), ai_usage_repository.current_month_cost(db_path=self.db_path))
        self.assertEqual(ai_usage_repository.outstanding_reserved_cost(db_path=self.db_path), Decimal(0))

    # 4
    def test_actual_below_reservation_leaves_no_residual_estimate(self):
        url = self.add(1)
        self.make().process_announcement(url)
        for row in self.reservations():
            estimated, actual = Decimal(row[6]), Decimal(row[8])
            self.assertLess(actual, estimated)  # факт (1000/500 токенов) меньше резерва (8000/16000 output)
        ledger_total = sum(Decimal(r[8]) for r in self.ledger_rows())
        self.assertEqual(self.effective(), ledger_total)
        self.assertLess(self.effective(), sum(Decimal(r[6]) for r in self.reservations()))

    # 5
    def test_finalization_failure_keeps_reservation_durable(self):
        url = self.add(1)
        with self.fail_finalization():
            self.make().process_announcement(url)
        (row,) = [r for r in self.reservations()]
        self.assertIn(row[7], ("reserved", "unresolved"))
        self.assertEqual(self.ledger_rows(), [])
        self.assertEqual(self.effective(), Decimal(row[6]))  # резерв входит в committed spend

    def test_finalization_failure_works_even_if_block_and_unresolved_marks_fail(self):
        first, second = self.add(1), self.add(2)
        pipeline = self.make()
        with self.fail_finalization(), mock.patch.object(
            ai_usage_repository, "open_accounting_block", side_effect=sqlite3.OperationalError("locked"),
        ), mock.patch.object(
            ai_usage_repository, "mark_reservation_unresolved", side_effect=sqlite3.OperationalError("locked"),
        ):
            pipeline.process_announcement(first)
        self.assertEqual(self.blocks(), [])
        (row,) = self.reservations()
        self.assertEqual(row[7], "reserved")  # без единой дополнительной записи резерв уже в БД
        outcome = self.make().process_announcement(second)  # полностью новый экземпляр
        self.assertEqual(outcome["state"], analysis_pipeline.OUTCOME_ACCOUNTING_BLOCKED)
        self.assertEqual(self.triage.calls, [first])

    # 6
    def test_restart_reservation_still_blocks_next_paid_call(self):
        first, second = self.add(1), self.add(2)
        with self.fail_finalization():
            self.make().process_announcement(first)
        # «перезапуск»: новые pipeline, новые fake analyzers, ни байта памяти
        triage, deep = FakeTriage(), FakeDeep()
        restarted = analysis_pipeline.AnalysisPipeline(
            triage, deep, self.settings, db_path=self.db_path, gate=gate_pass,
        )
        self.assertEqual(len(ai_usage_repository.outstanding_reservations(db_path=self.db_path)), 1)
        outcome = restarted.process_announcement(second)
        self.assertEqual(outcome["state"], analysis_pipeline.OUTCOME_ACCOUNTING_BLOCKED)
        self.assertEqual(triage.calls, [])
        self.assertEqual(deep.calls, [])

    def test_next_tender_in_same_process_is_blocked_after_finalization_failure(self):
        first, second = self.add(1), self.add(2)
        pipeline = self.make()
        with self.fail_finalization():
            pipeline.process_announcement(first)
        outcome = pipeline.process_announcement(second)
        self.assertTrue(outcome["failed"])
        self.assertEqual(self.triage.calls, [first])
        self.assertIsNone(self.state(second))

    def test_batch_stops_after_finalization_failure(self):
        urls = [self.add(n) for n in range(1, 4)]
        with self.fail_finalization():
            result = self.make().process_batch()
        self.assertTrue(result["accounting_blocked"])
        self.assertEqual(self.triage.calls, urls[:1])
        self.assertEqual(self.deep.calls, [])

    def test_deep_finalization_failure_keeps_deep_result_and_blocks_next(self):
        first, second = self.add(1), self.add(2)
        real = ai_usage_repository._insert_event

        def fail_deep_only(conn, analysis_type, *args, **kwargs):
            if analysis_type == "deep":
                raise sqlite3.OperationalError("disk full")
            return real(conn, analysis_type, *args, **kwargs)

        with mock.patch.object(ai_usage_repository, "_insert_event", side_effect=fail_deep_only):
            pipeline = self.make()
            outcome = pipeline.process_announcement(first)
        self.assertEqual(outcome["state"], state_repo.STATE_DEEP_COMPLETED)  # результат Deep не потерян
        self.assertIsNotNone(analysis_repository.get_deep_analysis(first, db_path=self.db_path))
        self.assertEqual([r["analysis_type"] for r in ai_usage_repository.outstanding_reservations(db_path=self.db_path)], ["deep"])
        pipeline.process_announcement(second)
        self.assertEqual(self.triage.calls, [first])

    # 7, 8, 9
    def test_effective_spend_includes_outstanding_reservations(self):
        ai_usage_repository.record_usage(
            "triage", "gpt-5.6-luna", {"input_tokens": 1_000_000, "output_tokens": 0}, db_path=self.db_path,
        )
        actual = ai_usage_repository.current_month_cost(db_path=self.db_path)
        ai_usage_repository.reserve_call("triage", "gpt-5.6-luna", Decimal("0.5"), reference="x", db_path=self.db_path)
        self.assertEqual(self.effective(), actual + Decimal("0.5"))
        self.assertEqual(self.make().store.month_spend(), actual + Decimal("0.5"))

    def test_hard_limit_cannot_be_exceeded_by_actual_plus_reservations(self):
        guard = analysis_pipeline.BudgetGuard(self.settings)  # hard 15, soft 12
        ai_usage_repository.record_usage(
            "triage", "gpt-5.6-luna", {"input_tokens": 0, "output_tokens": 12_000_000}, db_path=self.db_path,
        )  # 14.4 факт
        ai_usage_repository.reserve_call("deep", "gpt-5.6-luna", Decimal("0.5"), reference="y", db_path=self.db_path)
        spend = self.make().store.month_spend()  # 14.9
        self.assertEqual(spend, Decimal("14.9"))
        self.assertFalse(guard.check(spend, Decimal("0.2"), "triage", "gpt-5.6-luna").allowed)  # 14.9 + 0.2 > 15
        self.assertTrue(guard.check(spend, Decimal("0.05"), "triage", "gpt-5.6-luna").allowed)
        # без учёта резерва 0.2 бы «влезло» — именно поэтому резервы входят в spend
        self.assertTrue(guard.check(Decimal("14.4"), Decimal("0.2"), "triage", "gpt-5.6-luna").allowed)

    def test_soft_limit_uses_effective_committed_spend(self):
        guard = analysis_pipeline.BudgetGuard(self.settings)
        ai_usage_repository.record_usage(
            "triage", "gpt-5.6-luna", {"input_tokens": 0, "output_tokens": 9_900_000}, db_path=self.db_path,
        )  # 11.88 факт: ниже soft 12
        self.assertEqual(guard.check(self.effective(), Decimal("0.01"), "deep", "m").decision, "allowed")
        ai_usage_repository.reserve_call("deep", "gpt-5.6-luna", Decimal("0.2"), reference="z", db_path=self.db_path)
        decision = guard.check(self.effective(), Decimal("0.01"), "deep", "m")  # 12.08 >= soft
        self.assertEqual((decision.decision, decision.allowed), ("soft_limit_mode", False))

    def test_pipeline_defers_when_effective_spend_leaves_no_room(self):
        url = self.add(1)
        ai_usage_repository.record_usage(
            "triage", "gpt-5.6-luna", {"input_tokens": 0, "output_tokens": 12_499_000}, db_path=self.db_path,
        )
        outcome = self.make(settings=BudgetSettings(monthly_budget_usd=Decimal("15"))).process_announcement(url)
        self.assertEqual(outcome["state"], state_repo.STATE_TRIAGE_DEFERRED_BUDGET)
        self.assertEqual(self.triage.calls, [])
        self.assertEqual(self.reservations(), [])  # отложенный вызов резерва не создаёт

    # 10
    def test_error_without_usage_and_no_billing_releases_reservation(self):
        first, second = self.add(1), self.add(2)
        self.triage.outcomes[first] = openai_triage.TriageError(
            openai_triage.KIND_RATE_LIMIT, "429", usage=None, response_id=None, attempts=3,
        )
        pipeline = self.make()
        outcome = pipeline.process_announcement(first)
        self.assertEqual(outcome["state"], state_repo.STATE_TRIAGE_ERROR)
        (row,) = self.reservations()
        self.assertEqual((row[7], row[8]), ("released", "0"))
        self.assertEqual(self.effective(), Decimal(0))
        self.assertEqual(self.ledger_rows(), [])
        # легитимный повтор после освобождения работает
        self.triage.outcomes.pop(first)
        self.assertEqual(pipeline.process_announcement(first)["state"], state_repo.STATE_DEEP_COMPLETED)
        self.assertEqual(pipeline.process_announcement(second)["state"], state_repo.STATE_DEEP_COMPLETED)

    def test_ambiguous_billing_timeout_stays_unresolved_and_blocks(self):
        first, second = self.add(1), self.add(2)
        self.triage.outcomes[first] = openai_triage.TriageError(
            openai_triage.KIND_TIMEOUT, "timeout", usage=None, response_id=None, attempts=3,
        )
        pipeline = self.make()
        pipeline.process_announcement(first)
        (row,) = self.reservations()
        self.assertEqual(row[7], "unresolved")
        self.assertEqual(pipeline.process_announcement(second)["state"], analysis_pipeline.OUTCOME_ACCOUNTING_BLOCKED)
        self.assertEqual(self.triage.calls, [first])
        self.assertEqual(self.effective(), Decimal(row[6]))

    def test_unexpected_analyzer_exception_stays_reserved_fail_closed(self):
        first, second = self.add(1), self.add(2)
        self.triage.outcomes[first] = RuntimeError("boom")
        pipeline = self.make()
        outcome = pipeline.process_announcement(first)
        self.assertEqual(outcome["reason_code"], analysis_pipeline.REASON_UNEXPECTED_ERROR)
        self.assertEqual([r[7] for r in self.reservations()], ["reserved"])
        self.assertEqual(pipeline.process_announcement(second)["state"], analysis_pipeline.OUTCOME_ACCOUNTING_BLOCKED)

    # 11
    def test_error_with_usage_records_actual_cost_and_settles(self):
        first, second = self.add(1), self.add(2)
        self.triage.outcomes[first] = openai_triage.TriageError(
            openai_triage.KIND_VALIDATION, "bad", usage=dict(USAGE), response_id="resp-bad", attempts=1,
        )
        pipeline = self.make()
        pipeline.process_announcement(first)
        (ledger,) = self.ledger_rows()
        self.assertEqual((ledger[2], ledger[9], ledger[10]), ("resp-bad", 0, openai_triage.KIND_VALIDATION))
        (row,) = self.reservations()
        self.assertEqual((row[7], Decimal(row[8])), ("settled", Decimal(ledger[8])))
        self.assertEqual(self.effective(), Decimal(ledger[8]))
        self.assertEqual(pipeline.process_announcement(second)["state"], state_repo.STATE_DEEP_COMPLETED)

    def test_paid_error_with_finalization_failure_keeps_reservation(self):
        url = self.add(1)
        self.triage.outcomes[url] = openai_triage.TriageError(
            openai_triage.KIND_VALIDATION, "bad", usage=dict(USAGE), response_id="resp-err", attempts=1,
        )
        with self.fail_finalization():
            self.make().process_announcement(url)
        (block,) = self.blocks()
        self.assertEqual((block["success"], block["error_kind"]), (0, openai_triage.KIND_VALIDATION))
        self.assertEqual(len(ai_usage_repository.outstanding_reservations(db_path=self.db_path)), 1)

    # 13
    def test_duplicate_response_id_is_safe(self):
        first, second = self.add(1), self.add(2)
        self.deep.fixed_response_id = "same-response"
        pipeline = self.make()
        pipeline.process_announcement(first)
        outcome = pipeline.process_announcement(second)
        self.assertEqual(outcome["state"], state_repo.STATE_DEEP_COMPLETED)
        self.assertEqual([r[2] for r in self.ledger_rows() if r[0] == "deep"], ["same-response"])
        self.assertEqual(ai_usage_repository.outstanding_reservations(db_path=self.db_path), [])
        self.assertEqual(self.blocks(), [])

    # идемпотентность резервов
    def test_no_duplicate_outstanding_reservations_but_retry_after_release_works(self):
        def reserve():
            return ai_usage_repository.reserve_call(
                "triage", "m", Decimal("0.01"), reference="u", prompt_version="p", input_hash="h", db_path=self.db_path,
            )

        first = reserve()
        with self.assertRaises(ai_usage_repository.ReservationBlocked):
            reserve()
        self.assertEqual(len(self.reservations()), 1)
        ai_usage_repository.release_reservation(first, "retry test", db_path=self.db_path)
        self.assertGreater(reserve(), first)

    def test_reservation_cannot_be_settled_twice(self):
        rid = ai_usage_repository.reserve_call("triage", "gpt-5.6-luna", Decimal("0.01"), reference="u", db_path=self.db_path)
        ai_usage_repository.settle_reservation(rid, "triage", "gpt-5.6-luna", USAGE, response_id="r1", db_path=self.db_path)
        with self.assertRaises(ValueError):
            ai_usage_repository.settle_reservation(
                rid, "triage", "gpt-5.6-luna", USAGE, response_id="r2", db_path=self.db_path,
            )
        self.assertEqual(len(self.ledger_rows()), 1)

    def test_settle_without_usage_is_refused_and_keeps_reservation(self):
        rid = ai_usage_repository.reserve_call("triage", "gpt-5.6-luna", Decimal("0.01"), reference="u", db_path=self.db_path)
        with self.assertRaises(ValueError):
            ai_usage_repository.settle_reservation(rid, "triage", "gpt-5.6-luna", None, db_path=self.db_path)
        self.assertEqual(self.reservations()[0][7], "reserved")

    def test_block_check_error_fails_closed(self):
        url = self.add(1)
        with mock.patch.object(
            ai_usage_repository, "outstanding_reservations", side_effect=sqlite3.OperationalError("locked"),
        ):
            outcome = self.make().process_announcement(url)
        self.assertTrue(outcome["failed"])
        self.assertEqual(self.triage.calls, [])

    # 14: reconcile
    def test_reconcile_settles_reservation_and_block_atomically_and_resumes(self):
        first, second = self.add(1), self.add(2)
        with self.fail_finalization():
            self.make().process_announcement(first)
        (block,) = self.blocks()
        self.assertEqual(block["usage"], USAGE)
        self.assertIsNotNone(block["reservation_id"])

        (item,) = ai_usage_repository.reconcile_accounting_blocks("locked DB fixed", db_path=self.db_path)

        self.assertGreater(item["cost_usd"], 0)
        self.assertEqual(self.blocks(), [])
        self.assertEqual(ai_usage_repository.outstanding_reservations(db_path=self.db_path), [])
        (ledger,) = self.ledger_rows()
        self.assertEqual((ledger[0], ledger[2], ledger[3]), ("triage", "resp-triage-1", 1000))
        self.assertEqual(self.reservations()[0][7], "settled")
        self.assertEqual(self.effective(), Decimal(ledger[8]))
        self.assertEqual(self.make().process_announcement(second)["state"], state_repo.STATE_DEEP_COMPLETED)

    def test_reconcile_failure_rolls_back_everything(self):
        url = self.add(1)
        with self.fail_finalization():
            self.make().process_announcement(url)
        with mock.patch.object(ai_usage_repository.pricing, "cost_from_usage", side_effect=ValueError("no price")):
            with self.assertRaises(ValueError):
                ai_usage_repository.reconcile_accounting_blocks("try", db_path=self.db_path)
        self.assertEqual(len(self.blocks()), 1)
        self.assertEqual(len(ai_usage_repository.outstanding_reservations(db_path=self.db_path)), 1)
        self.assertEqual(self.ledger_rows(), [])

    def test_reconcile_does_not_double_count_already_recorded_response(self):
        url = self.add(1)
        with self.fail_finalization():
            self.make().process_announcement(url)
        ai_usage_repository.record_usage(
            "triage", self.triage.model, USAGE, response_id="resp-triage-1", db_path=self.db_path,
        )
        (item,) = ai_usage_repository.reconcile_accounting_blocks("already there", db_path=self.db_path)
        self.assertTrue(item["duplicate"])
        self.assertEqual(len(self.ledger_rows()), 1)
        self.assertEqual(ai_usage_repository.outstanding_reservations(db_path=self.db_path), [])

    def test_reconcile_requires_note_and_never_touches_unknown_usage_reservations(self):
        rid = ai_usage_repository.reserve_call("triage", "gpt-5.6-luna", Decimal("0.01"), reference="u", db_path=self.db_path)
        with self.assertRaises(ValueError):
            ai_usage_repository.reconcile_accounting_blocks("  ", db_path=self.db_path)
        ai_usage_repository.reconcile_accounting_blocks("nothing to do", db_path=self.db_path)
        self.assertEqual(self.reservations()[0][7], "reserved")  # без известного usage резерв не трогается
        with self.assertRaises(ValueError):
            ai_usage_repository.release_reservation(rid, " ", db_path=self.db_path)
        ai_usage_repository.release_reservation(rid, "operator: подтверждено в dashboard, биллинга нет", db_path=self.db_path)
        self.assertEqual(self.reservations()[0][7], "released")

    def test_reconcile_cli_list_and_release_require_note(self):
        from src.ai import accounting_reconcile

        rid = ai_usage_repository.reserve_call("triage", "gpt-5.6-luna", Decimal("0.01"), reference="u", db_path=self.db_path)
        out = io.StringIO()
        with mock.patch("sys.stdout", out):
            accounting_reconcile.main(["--list", "--db-path", str(self.db_path)])
        self.assertIn(f"резерв #{rid}", out.getvalue())
        with mock.patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            accounting_reconcile.main(["--release", str(rid), "--db-path", str(self.db_path)])
        self.assertEqual(self.reservations()[0][7], "reserved")
        with mock.patch("sys.stdout", io.StringIO()):
            accounting_reconcile.main(["--release", str(rid), "--note", "проверено", "--db-path", str(self.db_path)])
        self.assertEqual(self.reservations()[0][7], "released")

    # 15
    def test_dry_run_creates_no_reservations(self):
        self.add(1)
        self.add(2)
        analysis_pipeline.dry_run(db_path=self.db_path, environ={})
        self.assertEqual(self.reservations(), [])
        self.assertEqual(self.triage.calls, [])

    def test_dry_run_store_reports_outstanding_reservation_readonly(self):
        rid = ai_usage_repository.reserve_call("triage", "gpt-5.6-luna", Decimal("0.5"), reference="u", db_path=self.db_path)
        store = analysis_pipeline._ReadOnlyStore(self.db_path)
        self.assertEqual([r["id"] for r in store.outstanding_reservations()], [rid])
        self.assertEqual(store.month_spend(), Decimal("0.5"))
        report = analysis_pipeline.dry_run(db_path=self.db_path, environ={})
        self.assertEqual(len(report["outstanding_reservations"]), 1)


class BatchTests(PipelineTestCase):
    def test_batch_limit_and_settled_second_run(self):
        urls = [self.add(n) for n in range(1, 4)]
        pipeline = self.make()

        first = pipeline.process_batch(limit=2)
        self.assertEqual((first["candidate_count"], first["processed_count"]), (2, 2))
        self.assertEqual(self.triage.calls, urls[:2])
        second = pipeline.process_batch()
        self.assertEqual(second["candidate_count"], 1)  # только оставшийся, без лимита в день
        third = pipeline.process_batch()
        self.assertEqual(third["candidate_count"], 0)
        self.assertEqual(len(self.triage.calls), 3)
        self.assertEqual(third["api_calls"], {"triage": 0, "deep": 0})

    def test_invalid_limit_rejected(self):
        for bad in (0, -1, True, "5"):
            with self.assertRaises(ValueError):
                self.make().select_candidates(bad)

    def test_fresh_tenders_come_before_retries(self):
        retry = self.add(1)
        self.triage.outcomes[retry] = openai_triage.TriageError(openai_triage.KIND_TIMEOUT, "t")
        pipeline = self.make()
        pipeline.process_announcement(retry)
        fresh = self.add(2)
        self.assertEqual([c["resource_url"] for c in pipeline.select_candidates()], [fresh, retry])


class DryRunTests(PipelineTestCase):
    def file_digest(self):
        return hashlib.sha256(self.db_path.read_bytes()).hexdigest()

    def test_dry_run_creates_no_client_calls_nothing_and_does_not_mutate(self):
        url = self.add(1)
        self.add(2)
        self.make().process_announcement(url)  # url: deep_completed; 2: pending
        digest = self.file_digest()

        with mock.patch.object(openai_triage.OpenAITriageAnalyzer, "ensure_client", side_effect=AssertionError("client")), \
             mock.patch.object(openai_deep_analysis.OpenAIDeepAnalysisAnalyzer, "ensure_client", side_effect=AssertionError("client")):
            report = analysis_pipeline.dry_run(db_path=self.db_path, environ={})

        self.openai_client.assert_not_called()
        self.assertEqual(self.file_digest(), digest)
        self.assertEqual(report["enriched_total"], 2)
        self.assertEqual(report["candidate_count"], 1)
        self.assertEqual(report["routing_preview"], {"pending_triage": 1})
        self.assertGreater(report["estimated_pending_triage_cost_usd"], 0)
        self.assertEqual(report["config"]["deep_fallback_model"], "gpt-5.6-terra")
        self.assertEqual(report["config"]["max_deep_input_tokens"], 140_000)
        self.assertEqual(len(self.ledger_rows()), 2)

    def test_dry_run_triage_preflight_breakdown_is_consistent(self):
        from src.ai import preflight as preflight_module

        for number in range(1, 4):
            self.add(number)
        report = analysis_pipeline.dry_run(db_path=self.db_path, environ={})
        t = report["triage_preflight"]

        per_request = []
        for url in (f"https://example.test/resource/{n}" for n in range(1, 4)):
            ctx = analysis_pipeline.tender_context_module.build_tender_context(url, db_path=self.db_path)
            request = self.triage.build_request(analysis_pipeline.tender_context_module.build_triage_context(ctx))
            per_request.append(preflight_module.estimate_request_tokens(request))
        input_total = sum(i for i, _ in per_request)
        output_total = sum(o for _, o in per_request)

        self.assertEqual(t["candidate_count"], 3)
        self.assertEqual(t["estimated_input_tokens_total"], input_total)
        self.assertEqual(t["estimated_output_tokens_total"], output_total)
        self.assertEqual(output_total, 3 * 8000)  # резерв triage, а не Deep-размер (16000)
        # ставки USD за 1M токенов: input по max(input, cache_write) = 0.25, output = 1.20 (Luna, short)
        self.assertEqual(t["estimated_ordinary_or_cache_write_cost_usd"], Decimal(input_total) * Decimal("0.25") / 1_000_000)
        self.assertEqual(t["estimated_output_cost_usd"], Decimal(output_total) * Decimal("1.20") / 1_000_000)
        self.assertEqual(
            t["estimated_total_cost_usd"],
            t["estimated_ordinary_or_cache_write_cost_usd"] + t["estimated_output_cost_usd"],
        )
        self.assertEqual(report["estimated_pending_triage_cost_usd"], t["estimated_total_cost_usd"])
        self.assertEqual(t["rate_tiers"], {"short": 3})  # long-context тариф для обычного triage не применяется

    def test_cli_dry_run_prints_triage_preflight_estimate(self):
        self.add(1)
        out = io.StringIO()
        with mock.patch("dotenv.load_dotenv"), mock.patch("sys.stdout", out):
            analysis_pipeline.main(["--dry-run", "--db-path", str(self.db_path)])
        text = out.getvalue()
        for name in (
            "TRIAGE PREFLIGHT ESTIMATE", "candidate_count: 1", "estimated_input_tokens_total",
            "estimated_output_tokens_total", "estimated_ordinary_or_cache_write_cost",
            "estimated_output_cost", "estimated_total_cost", "average_estimated_cost_per_candidate",
        ):
            self.assertIn(name, text)

    def test_dry_run_previews_routing_from_saved_triage(self):
        url = self.add(1, estimated_value_amd="5000000")
        pipeline = self.make(gate=commercial_gate.evaluate, settings=BudgetSettings(max_deep_input_tokens=1))
        pipeline.process_announcement(url)  # deep_deferred_input
        digest = self.file_digest()

        report = analysis_pipeline.dry_run(db_path=self.db_path, environ={})  # лимит 140000 -> Deep был бы разрешён

        self.assertEqual(report["routing_preview"], {"deep:ready_for_deep": 1})
        self.assertEqual(self.file_digest(), digest)
        self.assertEqual(self.deep.calls, [])

    def test_dry_run_on_database_without_analysis_tables_creates_nothing(self):
        path = Path(self._tmp.name) / "fresh.db"
        document_repository.init_db(path)
        enrichment_repository.init_db(path)
        announcement = {
            "title": "T", "source_section": "s", "source_section_name": "S", "source_page_url": "https://x.test/",
            "resource_url": "https://example.test/resource/9", "resource_type": "armeps_documents_page",
        }
        announcement_repository.save_announcement(announcement, path)
        enrichment_repository.save_enrichment(
            announcement["resource_url"], {"enrichment_status": "success", "documents": []}, path,
        )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()

        report = analysis_pipeline.dry_run(db_path=path, environ={})

        self.assertEqual(report["candidate_count"], 1)
        self.assertEqual(report["month_spend_usd"], 0)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        with closing(sqlite3.connect(path)) as conn:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        self.assertNotIn("tender_triage", tables)
        self.assertNotIn("tender_pipeline_state", tables)
        self.assertNotIn("ai_usage_events", tables)

    def test_cli_requires_dry_run(self):
        # приватный stdout/stderr: main() не должен перенастраивать глобальный sys.stdout процесса тестов
        with mock.patch("sys.stdout", io.StringIO()), mock.patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                analysis_pipeline.main([])

    def test_cli_dry_run_prints_report(self):
        self.add(1)
        out = io.StringIO()
        with mock.patch("dotenv.load_dotenv"), mock.patch("sys.stdout", out):
            code = analysis_pipeline.main(["--dry-run", "--limit", "5", "--db-path", str(self.db_path)])
        self.assertEqual(code, 0)
        self.assertIn("DRY RUN", out.getvalue())
        self.assertIn("deep_fallback_model: gpt-5.6-terra", out.getvalue())
        self.openai_client.assert_not_called()


class RunAiAnalysisTests(PipelineTestCase):
    def test_without_api_key_nothing_is_called(self):
        self.add(1)
        result = analysis_pipeline.run_ai_analysis(db_path=self.db_path, environ={})
        self.assertEqual(result, {"status": "skipped_no_api_key"})
        self.openai_client.assert_not_called()


class SafetyGuardTests(unittest.TestCase):
    def test_production_database_stays_blocked(self):
        with safety_guards.expect_violation(self, safety_guards.DATABASE_MESSAGE):
            sqlite3.connect(safety_guards.PRODUCTION_DB_PATH)

    def test_external_network_stays_blocked(self):
        import socket

        with safety_guards.expect_violation(self, safety_guards.NETWORK_MESSAGE):
            socket.create_connection(("api.openai.com", 443))


class PipelineStateRepositoryTests(PipelineTestCase):
    def test_unknown_state_and_unknown_announcement_rejected(self):
        url = self.add(1)
        with self.assertRaises(ValueError):
            state_repo.save_state(url, "bogus", db_path=self.db_path)
        with self.assertRaises(ValueError):
            state_repo.save_state("https://nope.test/", state_repo.STATE_DEEP_COMPLETED, db_path=self.db_path)

    def test_state_is_overwritten_and_counted(self):
        url = self.add(1)
        state_repo.save_state(url, state_repo.STATE_PENDING_TRIAGE, db_path=self.db_path)
        state_repo.save_state(url, state_repo.STATE_DEEP_COMPLETED, reason_code="x", details={"a": 1}, db_path=self.db_path)
        self.assertEqual(state_repo.get_state(url, db_path=self.db_path)["details"], {"a": 1})
        self.assertEqual(state_repo.count_by_state(self.db_path), {state_repo.STATE_DEEP_COMPLETED: 1})


if __name__ == "__main__":
    unittest.main()
