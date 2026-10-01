"""
Тесты автоматического AI batch в monitor (AI_ANALYSIS_BATCH_LIMIT): только fake analyzers, временная БД,
ZERO OpenAI, реальный .env не читается и не меняется.

    python -m unittest tests.test_monitor_ai_batch -v
"""

import hashlib
import sqlite3
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

from src import monitor
from src.ai import analysis_pipeline, openai_deep_analysis, openai_triage
from src.ai.analysis_pipeline import BatchLimitConfigError, load_monitor_batch_limit
from src.database import ai_usage_repository, pipeline_state_repository as state_repo
from tests.test_analysis_pipeline import gate_pass, triage_result
from tests.test_operational_eligibility import FUTURE, PAST, DeadlineCase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"


def _env_digest():
    return hashlib.sha256(ENV_FILE.read_bytes()).hexdigest() if ENV_FILE.exists() else None


class LimitParsingTests(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(load_monitor_batch_limit({"AI_ANALYSIS_BATCH_LIMIT": " 5 "}), 5)

    def test_missing_or_empty_is_not_unlimited(self):
        for env in ({}, {"AI_ANALYSIS_BATCH_LIMIT": ""}, {"AI_ANALYSIS_BATCH_LIMIT": "  "}):
            with self.subTest(env=env), self.assertRaises(BatchLimitConfigError):
                load_monitor_batch_limit(env)

    def test_zero_negative_and_non_integer_rejected(self):
        for raw in ("0", "-1", "-5", "abc", "2.5", "5x", "none", "unlimited"):
            with self.subTest(raw=raw), self.assertRaises(BatchLimitConfigError):
                load_monitor_batch_limit({"AI_ANALYSIS_BATCH_LIMIT": raw})

    def test_env_example_documents_disabled_and_limit_five(self):
        text = "\n" + (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn("\nAI_ANALYSIS_ENABLED=false\n", text)
        self.assertIn("\nAI_ANALYSIS_BATCH_LIMIT=5\n", text)


class MonitorAiBatchTests(DeadlineCase):
    def setUp(self):
        super().setUp()
        self._env_before = _env_digest()
        self.addCleanup(lambda: self.assertEqual(self._env_before, _env_digest()))  # реальный .env не тронут
        self.gate = gate_pass
        for analyzer in (self.triage, self.deep):
            analyzer._client = object()  # has_api_key=True без реального client; вызовы идут в fake-методы
        real = analysis_pipeline.AnalysisPipeline
        patches = [
            mock.patch.object(analysis_pipeline, "build_analyzers", return_value=(self.triage, self.deep)),
            mock.patch.object(
                analysis_pipeline, "AnalysisPipeline",
                side_effect=lambda *a, **k: real(*a, **{**k, "clock": self.clock, "gate": self.gate}),
            ),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_ai(self, limit="5"):
        env = {} if limit is None else {"AI_ANALYSIS_BATCH_LIMIT": limit}
        return analysis_pipeline.run_monitor_ai_batch(db_path=self.db_path, environ=env)

    def test_limit_five_of_twenty_and_next_run_continues(self):
        urls = [self.add_deadline(n, FUTURE) for n in range(1, 21)]
        result = self.run_ai("5")
        self.assertEqual((result["batch_limit"], result["selected"], result["processed"]), (5, 5, 5))
        self.assertEqual(self.triage.calls, urls[:5])
        for url in urls[5:]:  # шестой и далее: analyzer не вызван, состояния нет
            self.assertNotIn(url, self.triage.calls)
            self.assertIsNone(self.state(url))
        self.assertEqual((result["triage_api_calls"], result["deep_api_calls"], result["deep_completed"]), (5, 5, 5))
        self.assertIsNotNone(result["batch_cost_usd"])
        # следующий отдельный запуск продолжает backlog; обработанные не оплачиваются повторно
        result2 = self.run_ai("5")
        self.assertEqual(self.triage.calls, urls[:10])
        self.assertEqual(result2["selected"], 5)
        self.assertEqual(len(self.deep.calls), 10)

    def test_expired_do_not_consume_limit(self):
        expired = [self.add_deadline(n, PAST) for n in range(1, 6)]
        active = [self.add_deadline(n, FUTURE) for n in range(100, 110)]
        self.run_ai("3")
        self.assertEqual(self.triage.calls, active[:3])
        for url in expired:
            self.assertEqual(self.state(url)["state"], state_repo.STATE_SKIPPED_EXPIRED)

    def test_reusable_result_creates_no_paid_call(self):
        self.add_deadline(1, FUTURE)
        self.run_ai("5")
        calls = (len(self.triage.calls), len(self.deep.calls))
        result = self.run_ai("5")
        self.assertEqual((len(self.triage.calls), len(self.deep.calls)), calls)
        self.assertEqual((result["selected"], result["triage_api_calls"], result["deep_api_calls"]), (0, 0, 0))

    def test_invalid_limit_builds_nothing_and_calls_nothing(self):
        self.add_deadline(1, FUTURE)
        for limit in (None, "", "0", "-3", "1.5", "x"):
            with self.subTest(limit=limit):
                self.assertEqual(self.run_ai(limit)["status"], "config_error")
        analysis_pipeline.build_analyzers.assert_not_called()
        self.assertEqual((self.triage.calls, self.deep.calls), ([], []))
        self.openai_client.assert_not_called()

    def test_accounting_block_stops_further_paid_calls(self):
        urls = [self.add_deadline(n, FUTURE) for n in range(1, 5)]
        real = ai_usage_repository.reserve_call
        count = []

        def failing(*a, **k):  # сбой резерва на втором платном вызове -> accounting_blocked
            count.append(1)
            if len(count) >= 2:
                raise RuntimeError("db locked")
            return real(*a, **k)

        with mock.patch.object(ai_usage_repository, "reserve_call", side_effect=failing), \
                self.assertLogs(analysis_pipeline.logger, "CRITICAL"):
            result = self.run_ai("4")
        self.assertTrue(result["accounting_blocked"])
        self.assertEqual(len(self.triage.calls), 1)
        self.assertEqual(len(count), 2)  # после блока новых резервов/вызовов нет
        self.assertEqual(result["processed"], 1)  # резерв Deep не создан -> блок, остальные не начаты
        for url in urls[1:]:
            self.assertIsNone(self.state(url))

    def test_ordinary_tender_errors_do_not_break_batch(self):
        urls = [self.add_deadline(n, FUTURE) for n in range(1, 4)]
        self.triage.outcomes[urls[0]] = triage_result("not_relevant")

        def gate(triage, ctx, min_value):  # ошибка до платного Deep-вызова: резерва нет, batch продолжается
            if ctx["resource_url"] == urls[1]:
                raise RuntimeError("boom")
            return gate_pass(triage, ctx, min_value)

        self.gate = gate
        result = self.run_ai("3")
        self.assertEqual(result["processed"], 3)
        self.assertEqual(result["not_relevant"], 1)
        self.assertEqual(result["errors"], 1)
        self.assertEqual(self.state(urls[2])["state"], state_repo.STATE_DEEP_COMPLETED)

    def test_post_response_validation_error_with_known_usage_does_not_block_batch(self):
        a, b = [self.add_deadline(n, FUTURE) for n in (1, 2)]
        # платный ответ получен (usage известен), затем контент не прошёл валидацию: это НЕ ambiguous billing
        self.deep.outcomes[a] = openai_deep_analysis.DeepAnalysisError(
            openai_triage.KIND_VALIDATION, "bad evidence", usage=self.deep.usage, response_id="r-val-a",
        )
        result = self.run_ai("2")
        self.assertFalse(result["accounting_blocked"])
        self.assertEqual(result["processed"], 2)
        self.assertEqual(self.state(a)["state"], state_repo.STATE_ESCALATION_CANDIDATE)
        self.assertEqual(self.deep.calls, [a, b])  # Deep Tender B реально вызван
        self.assertEqual(self.state(b)["state"], state_repo.STATE_DEEP_COMPLETED)
        # резервов нет в reserved/unresolved, usage A записан в ledger как success=0/validation
        self.assertEqual(ai_usage_repository.outstanding_reservations(db_path=self.db_path), [])
        with closing(sqlite3.connect(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT success, error_kind FROM ai_usage_events WHERE analysis_type='deep' ORDER BY id"
            ).fetchall()
        self.assertEqual(rows, [(0, openai_triage.KIND_VALIDATION), (1, None)])


class MonitorIntegrationTests(unittest.TestCase):
    def run_stage(self, env):
        with mock.patch.dict("os.environ", env, clear=True):
            return monitor._run_ai_analysis_if_enabled()

    def test_disabled_makes_zero_ai_calls(self):
        with mock.patch.object(analysis_pipeline, "run_monitor_ai_batch") as batch, \
                mock.patch.object(analysis_pipeline, "run_ai_analysis") as legacy, \
                mock.patch.object(analysis_pipeline, "build_analyzers") as build, \
                mock.patch("openai.OpenAI") as client:
            for env in ({}, {"AI_ANALYSIS_ENABLED": "false", "AI_ANALYSIS_BATCH_LIMIT": "5"}):
                self.assertIsNone(self.run_stage(env))
        for m in (batch, legacy, build, client):
            m.assert_not_called()

    def test_enabled_uses_batch_entry_never_unlimited_run(self):
        with mock.patch.object(analysis_pipeline, "run_monitor_ai_batch", return_value={"status": "ok"}) as batch, \
                mock.patch.object(analysis_pipeline, "run_ai_analysis") as legacy:
            result = self.run_stage({"AI_ANALYSIS_ENABLED": "true", "AI_ANALYSIS_BATCH_LIMIT": "5"})
        self.assertEqual(result, {"status": "ok"})
        batch.assert_called_once()
        legacy.assert_not_called()

    def test_enabled_passes_limit_to_pipeline(self):
        with mock.patch.object(analysis_pipeline, "run_ai_analysis", return_value={"status": "skipped_no_api_key"}) as run:
            result = self.run_stage({"AI_ANALYSIS_ENABLED": "true", "AI_ANALYSIS_BATCH_LIMIT": "5"})
        self.assertEqual(run.call_args.kwargs["limit"], 5)
        self.assertEqual(result["batch_limit"], 5)

    def test_enabled_without_limit_is_config_error_and_no_ai(self):
        with mock.patch.object(analysis_pipeline, "run_ai_analysis") as run, \
                mock.patch.object(analysis_pipeline, "build_analyzers") as build:
            result = self.run_stage({"AI_ANALYSIS_ENABLED": "true"})
        self.assertEqual(result["status"], "config_error")
        run.assert_not_called()
        build.assert_not_called()

    def test_ai_exception_is_contained_by_monitor_stage(self):
        with mock.patch.object(analysis_pipeline, "run_monitor_ai_batch", side_effect=RuntimeError("boom")):
            result = self.run_stage({"AI_ANALYSIS_ENABLED": "true", "AI_ANALYSIS_BATCH_LIMIT": "5"})
        self.assertEqual(result["status"], "error")  # run_monitor продолжает и возвращает scraping-результат


if __name__ == "__main__":
    unittest.main()
