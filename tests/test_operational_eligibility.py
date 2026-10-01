"""
Тесты Operational Eligibility Gate (deadline): чистая логика, пайплайн, batch, dry-run, one-shot, CLI.
Только fake analyzers и временная БД; время задаётся явно (никакой зависимости от текущей даты).

    python -m unittest tests.test_operational_eligibility -v
"""

import hashlib
import io
import os
import sqlite3
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest import mock

from src import monitor
from src.ai import analysis_pipeline, commercial_gate, one_shot, operational_eligibility as oe
from src.ai.budget_settings import BudgetSettings
from src.database import (
    ai_usage_repository, analysis_repository, announcement_repository, enrichment_repository,
    pipeline_state_repository as state_repo,
)
from tests.test_analysis_pipeline import PipelineTestCase

YEREVAN = timezone(timedelta(hours=4))
NOW = datetime(2026, 10, 1, 12, 0, 0, tzinfo=YEREVAN)
PAST = "2026-09-30 11:00:00"
FUTURE = "2026-10-10 11:00:00"


class ParseTests(unittest.TestCase):
    def test_naive_datetime_is_yerevan(self):
        parsed = oe.parse_deadline("2026-10-05 14:30:00")
        self.assertEqual(parsed, datetime(2026, 10, 5, 10, 30, tzinfo=timezone.utc))
        self.assertEqual(parsed.utcoffset(), timedelta(hours=4))

    def test_aware_keeps_offset(self):
        parsed = oe.parse_deadline("2026-10-05T14:30:00+02:00")
        self.assertEqual(parsed.utcoffset(), timedelta(hours=2))
        self.assertEqual(parsed, datetime(2026, 10, 5, 12, 30, tzinfo=timezone.utc))
        self.assertEqual(oe.parse_deadline("2026-10-05T10:30:00Z"), datetime(2026, 10, 5, 10, 30, tzinfo=timezone.utc))

    def test_date_only_is_end_of_local_day(self):
        parsed = oe.parse_deadline("2026-10-05")
        self.assertEqual((parsed.hour, parsed.minute, parsed.second), (23, 59, 59))
        self.assertEqual(parsed.utcoffset(), timedelta(hours=4))

    def test_other_project_formats(self):
        self.assertEqual(oe.parse_deadline("05/10/2026 14:30"), oe.parse_deadline("2026-10-05 14:30:00"))

    def test_garbage_is_none(self):
        for value in (None, "", "   ", "Бессрочный", "2026-13-45 99:99:99", "tomorrow", 12345, "05.10.26"):
            self.assertIsNone(oe.parse_deadline(value), value)

    def test_now_must_be_aware(self):
        with self.assertRaises(ValueError):
            oe.evaluate(FUTURE, None, datetime(2026, 10, 1, 12, 0))


class EvaluateTests(unittest.TestCase):
    def status(self, list_value, detail_value=None, now=NOW):
        return oe.evaluate(list_value, detail_value, now)

    def test_future_is_eligible(self):
        result = self.status(FUTURE)
        self.assertEqual(result.status, oe.STATUS_ELIGIBLE)
        self.assertFalse(result.expired)

    def test_past_is_expired(self):
        result = self.status(PAST)
        self.assertEqual(result.status, oe.STATUS_EXPIRED)
        self.assertTrue(result.expired)
        self.assertEqual(result.resolved_deadline, datetime(2026, 9, 30, 11, 0, tzinfo=YEREVAN))

    def test_boundary_deadline_equal_now_is_expired(self):
        self.assertEqual(self.status("2026-10-01 12:00:00").status, oe.STATUS_EXPIRED)
        self.assertEqual(self.status("2026-10-01 12:00:01").status, oe.STATUS_ELIGIBLE)

    def test_unknown_when_no_deadline(self):
        for value in (None, "", "  "):
            result = self.status(value, value)
            self.assertEqual(result.status, oe.STATUS_UNKNOWN)
            self.assertIsNone(result.resolved_deadline)

    def test_malformed_is_unknown_not_expired(self):
        result = self.status("Бессрочный")
        self.assertEqual(result.status, oe.STATUS_UNKNOWN)
        self.assertEqual(result.deadline_sources[0][2], None)

    def test_malformed_beside_expired_is_not_reliable(self):
        self.assertEqual(self.status(PAST, "мусор").status, oe.STATUS_UNKNOWN)

    def test_malformed_beside_future_uses_parseable(self):
        self.assertEqual(self.status(FUTURE, "мусор").status, oe.STATUS_ELIGIBLE)

    def test_naive_deadline_uses_yerevan_not_server_tz(self):
        # now = 07:30 UTC = 11:30 Yerevan. 08:00 Yerevan = 04:00 UTC -> истёк; 12:00 Yerevan = 08:00 UTC -> активен.
        now_utc = datetime(2026, 10, 1, 7, 30, tzinfo=timezone.utc)
        self.assertEqual(self.status("2026-10-01 12:00:00", now=now_utc).status, oe.STATUS_ELIGIBLE)
        self.assertEqual(self.status("2026-10-01 08:00:00", now=now_utc).status, oe.STATUS_EXPIRED)
        # Если бы наивное значение считалось UTC, 08:00 (UTC) ещё не наступило бы (07:30 UTC) — тест бы упал.

    def test_aware_deadline_compares_correctly(self):
        now_utc = datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc)
        self.assertEqual(self.status("2026-10-01T11:00:00+02:00", now=now_utc).status, oe.STATUS_EXPIRED)  # 09:00Z
        self.assertEqual(self.status("2026-10-01T13:00:00+02:00", now=now_utc).status, oe.STATUS_ELIGIBLE)  # 11:00Z

    def test_date_only_expires_after_end_of_local_day(self):
        self.assertEqual(self.status("2026-10-01", now=NOW).status, oe.STATUS_ELIGIBLE)  # 12:00, день не закончен
        self.assertEqual(self.status("2026-10-01", now=datetime(2026, 10, 1, 23, 59, 58, tzinfo=YEREVAN)).status,
                         oe.STATUS_ELIGIBLE)
        self.assertEqual(self.status("2026-10-01", now=datetime(2026, 10, 2, 0, 0, 0, tzinfo=YEREVAN)).status,
                         oe.STATUS_EXPIRED)

    def test_both_sources_expired(self):
        result = self.status(PAST, "2026-09-29 10:00:00")
        self.assertEqual(result.status, oe.STATUS_EXPIRED)
        self.assertEqual(result.resolved_deadline, datetime(2026, 9, 30, 11, 0, tzinfo=YEREVAN))  # самый поздний

    def test_list_past_detail_future_is_conflict_not_skip(self):
        result = self.status(PAST, FUTURE)
        self.assertEqual(result.status, oe.STATUS_CONFLICT)
        self.assertFalse(result.expired)
        self.assertEqual(result.resolved_deadline, datetime(2026, 10, 10, 11, 0, tzinfo=YEREVAN))

    def test_list_future_detail_past_is_conflict_not_skip(self):
        self.assertEqual(self.status(FUTURE, PAST).status, oe.STATUS_CONFLICT)

    def test_agreeing_sources_eligible(self):
        self.assertEqual(self.status(FUTURE, FUTURE).status, oe.STATUS_ELIGIBLE)
        self.assertEqual(self.status(FUTURE, "2026-10-10T07:00:00Z").status, oe.STATUS_ELIGIBLE)  # тот же момент

    def test_to_dict_is_compact_and_deterministic(self):
        facts = self.status(PAST, "2026-09-29 10:00:00").to_dict()
        self.assertEqual(set(facts), {"status", "resolved_deadline", "deadline_sources", "evaluated_at"})
        self.assertEqual(facts["evaluated_at"], NOW.isoformat())
        self.assertEqual([s["source"] for s in facts["deadline_sources"]], [oe.SOURCE_LIST, oe.SOURCE_DETAIL])


class DeadlineCase(PipelineTestCase):
    """Фикстуры с явными сроками и фиксированными часами."""

    now = NOW

    def clock(self):
        return self.now

    def add_deadline(self, number, list_deadline=None, detail_deadline=None, **enrichment) -> str:
        announcement = {
            "title": f"Тендер {number}", "source_section": "open_competition",
            "source_section_name": "Открытый конкурс", "source_page_url": "https://gnumner.minfin.am/ru/page/test/",
            "resource_url": f"https://example.test/resource/{number}", "resource_type": "armeps_documents_page",
            "deadline_at": list_deadline,
        }
        announcement_repository.save_announcement(announcement, self.db_path)
        data = {"enrichment_status": "success", "description": "Тендер", "documents": [],
                "deadline_at_detail": detail_deadline} | enrichment
        enrichment_repository.save_enrichment(announcement["resource_url"], data, self.db_path)
        return announcement["resource_url"]

    def make(self, gate=None, **kwargs):
        from tests.test_analysis_pipeline import gate_pass
        kwargs.setdefault("clock", self.clock)
        return super().make(gate=gate or gate_pass, **kwargs)

    def count(self, table):
        with closing(sqlite3.connect(self.db_path)) as conn:
            return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def assert_zero_spend(self):
        self.assertEqual(self.triage.calls, [])
        self.assertEqual(self.deep.calls, [])
        self.assertEqual(self.count("ai_call_reservations"), 0)
        self.assertEqual(self.count("ai_usage_events"), 0)
        self.openai_client.assert_not_called()


class PipelineGateTests(DeadlineCase):
    def test_active_tender_runs_normally(self):
        url = self.add_deadline(1, FUTURE)
        outcome = self.make().process_announcement(url)
        self.assertEqual(outcome["state"], state_repo.STATE_DEEP_COMPLETED)
        self.assertEqual(outcome["operational_eligibility"], oe.STATUS_ELIGIBLE)

    def test_expired_tender_costs_nothing(self):
        url = self.add_deadline(1, PAST)
        outcome = self.make().process_announcement(url)
        self.assertEqual(outcome["state"], state_repo.STATE_SKIPPED_EXPIRED)
        self.assertEqual(outcome["reason_code"], "deadline_expired")
        self.assertFalse(outcome["failed"])
        self.assertEqual(outcome["api_calls"], {"triage": 0, "deep": 0})
        self.assert_zero_spend()  # нет analyzer-вызовов, резервов, ledger, OpenAI client
        self.assertIsNone(analysis_repository.get_triage(url, db_path=self.db_path))

    def test_expired_state_has_deterministic_facts(self):
        url = self.add_deadline(1, PAST, "2026-09-29 10:00:00")
        self.make().process_announcement(url)
        state = self.state(url)
        self.assertEqual(state["state"], "skipped_expired")
        self.assertEqual(state["reason_code"], "deadline_expired")
        self.assertEqual(state["details"]["resolved_deadline"], "2026-09-30T11:00:00+04:00")
        self.assertEqual(state["details"]["evaluated_at"], NOW.isoformat())
        self.assertEqual(len(state["details"]["deadline_sources"]), 2)

    def test_not_relevant_label_is_not_applied(self):
        url = self.add_deadline(1, PAST)
        self.make().process_announcement(url)
        self.assertNotEqual(self.state(url)["state"], state_repo.STATE_STOPPED_NOT_RELEVANT)

    def test_unknown_and_conflict_continue_to_ai(self):
        unknown = self.add_deadline(1, None)
        conflict = self.add_deadline(2, PAST, FUTURE)
        malformed = self.add_deadline(3, "Бессрочный")
        pipeline = self.make()
        for url in (unknown, conflict, malformed):
            self.assertEqual(pipeline.process_announcement(url)["state"], state_repo.STATE_DEEP_COMPLETED)
        self.assertEqual(len(self.triage.calls), 3)

    def test_extension_makes_tender_eligible_again(self):
        url = self.add_deadline(1, "2026-09-30 11:00:00")
        pipeline = self.make()
        self.assertEqual(pipeline.process_announcement(url)["state"], state_repo.STATE_SKIPPED_EXPIRED)
        self.assertEqual(pipeline.select_candidates(), [])  # settled: повторно не пишется

        # скрапер обновил срок (announcement_repository обновляет запись по resource_url)
        self.add_deadline(1, "2026-10-10 11:00:00")
        candidates = pipeline.select_candidates()
        self.assertEqual([c["resource_url"] for c in candidates], [url])
        self.assertEqual(candidates[0]["operational_status"], oe.STATUS_ELIGIBLE)
        self.assertEqual(pipeline.process_announcement(url)["state"], state_repo.STATE_DEEP_COMPLETED)
        self.assertEqual(self.triage.calls, [url])

    def test_deep_completed_survives_later_expiry(self):
        url = self.add_deadline(1, "2026-10-05 11:00:00")
        self.make().process_announcement(url)  # now = 2026-10-01: активен
        deep_before = analysis_repository.get_deep_analysis(url, db_path=self.db_path)
        triage_before = analysis_repository.get_triage(url, db_path=self.db_path)
        self.assertEqual(self.state(url)["state"], state_repo.STATE_DEEP_COMPLETED)

        self.now = datetime(2026, 10, 6, 12, 0, tzinfo=YEREVAN)  # срок прошёл
        pipeline = self.make()
        self.assertEqual(pipeline.select_candidates(), [])
        outcome = pipeline.process_announcement(url)
        self.assertEqual(outcome["state"], state_repo.STATE_DEEP_COMPLETED)
        self.assertTrue(outcome["reused"]["deep"])
        self.assertEqual(self.state(url)["state"], state_repo.STATE_DEEP_COMPLETED)
        self.assertEqual(analysis_repository.get_deep_analysis(url, db_path=self.db_path), deep_before)
        self.assertEqual(analysis_repository.get_triage(url, db_path=self.db_path), triage_before)
        self.assertEqual((len(self.triage.calls), len(self.deep.calls)), (1, 1))

    def test_expired_before_deep_keeps_triage_and_makes_no_deep_reservation(self):
        url = self.add_deadline(1, "2026-10-05 11:00:00")
        self.make().process_announcement(url)
        # Deep-результат устарел (другой prompt_version), срок прошёл: новая платная работа запрещена.
        self.deep.prompt_version = "other-version"
        self.now = datetime(2026, 10, 6, 12, 0, tzinfo=YEREVAN)
        reservations_before = self.count("ai_call_reservations")
        outcome = self.make().process_announcement(url)
        self.assertEqual(outcome["state"], state_repo.STATE_SKIPPED_EXPIRED)
        self.assertEqual(len(self.deep.calls), 1)
        self.assertEqual(self.count("ai_call_reservations"), reservations_before)
        self.assertIsNotNone(analysis_repository.get_deep_analysis(url, db_path=self.db_path))
        self.assertIsNotNone(analysis_repository.get_triage(url, db_path=self.db_path))

    def test_allow_expired_bypasses_only_deadline(self):
        url = self.add_deadline(1, PAST)
        outcome = self.make(allow_expired=True).process_announcement(url)
        self.assertEqual(outcome["state"], state_repo.STATE_DEEP_COMPLETED)
        self.assertEqual(outcome["operational_eligibility"], oe.STATUS_EXPIRED)
        self.assertEqual((self.triage.calls, self.deep.calls), ([url], [url]))

    def test_allow_expired_does_not_bypass_budget_guard(self):
        url = self.add_deadline(1, PAST)
        tiny = BudgetSettings(monthly_budget_usd=Decimal("0.0001"), soft_limit_usd=Decimal("0.0001"))
        outcome = self.make(allow_expired=True, settings=tiny).process_announcement(url)
        self.assertEqual(outcome["state"], state_repo.STATE_TRIAGE_DEFERRED_BUDGET)
        self.assertEqual(self.triage.calls, [])

    def test_allow_expired_does_not_bypass_accounting_block(self):
        url = self.add_deadline(1, PAST)
        ai_usage_repository.reserve_call("triage", "m", Decimal("0.01"), reference="other", db_path=self.db_path)
        outcome = self.make(allow_expired=True).process_announcement(url)
        self.assertEqual(outcome["reason_code"], analysis_pipeline.REASON_ACCOUNTING_BLOCKED)
        self.assertEqual(self.triage.calls, [])

    def test_allow_expired_does_not_bypass_commercial_gate(self):
        url = self.add_deadline(1, PAST, estimated_value_amd="100")
        settings = BudgetSettings(min_deep_value_amd=1000)
        outcome = self.make(allow_expired=True, settings=settings, gate=commercial_gate.evaluate).process_announcement(url)
        self.assertEqual(outcome["state"], state_repo.STATE_COMMERCIAL_GATE_SKIP)
        self.assertEqual(self.deep.calls, [])


class BatchTests(DeadlineCase):
    def test_batch_continues_past_expired_rows_to_active_candidates(self):
        expired = [self.add_deadline(n, PAST) for n in (1, 2, 3)]
        active = [self.add_deadline(n, FUTURE) for n in (4, 5, 6)]
        pipeline = self.make()

        candidates = pipeline.select_candidates(limit=2)
        reasons = [(c["resource_url"], c["reason"]) for c in candidates]
        self.assertEqual(reasons[:2], [(active[0], "needs_triage"), (active[1], "needs_triage")])
        self.assertEqual({u for u, r in reasons[2:]}, set(expired))
        self.assertTrue(all(r == "skip_expired" for _, r in reasons[2:]))

        result = pipeline.process_batch(limit=2)
        self.assertEqual(self.triage.calls, active[:2])
        self.assertEqual(result["state_counts"], {
            state_repo.STATE_DEEP_COMPLETED: 2, state_repo.STATE_SKIPPED_EXPIRED: 3,
        })
        self.assertEqual(result["failed_count"], 0)
        self.assertIsNone(self.state(active[2]))  # за limit
        # expired settled: повторный прогон их не трогает
        self.assertEqual([c["resource_url"] for c in pipeline.select_candidates()], [active[2]])

    def test_expired_do_not_reserve(self):
        for n in (1, 2):
            self.add_deadline(n, PAST)
        self.make().process_batch()
        self.assert_zero_spend()

    def test_fresh_before_retry_order_unchanged(self):
        retry = self.add_deadline(1, FUTURE)
        self.make().process_announcement(retry)
        self.deep.prompt_version = "v-new"  # deep устарел -> retry_deep
        fresh = self.add_deadline(2, FUTURE)
        pipeline = self.make()
        self.assertEqual([(c["resource_url"], c["reason"]) for c in pipeline.select_candidates()],
                         [(fresh, "needs_triage"), (retry, "retry_deep")])


class DryRunTests(DeadlineCase):
    def file_hash(self):
        return hashlib.sha256(self.db_path.read_bytes()).hexdigest()

    def test_dry_run_reports_statuses_and_is_read_only(self):
        self.add_deadline(1, PAST)
        self.add_deadline(2, PAST, "2026-09-29 10:00:00")
        self.add_deadline(3, FUTURE)
        self.add_deadline(4, None)
        self.add_deadline(5, PAST, FUTURE)
        before = self.file_hash()

        report = analysis_pipeline.dry_run(db_path=self.db_path, limit=1, environ={}, now=NOW)

        self.assertEqual(self.file_hash(), before)
        self.openai_client.assert_not_called()
        self.assertEqual(report["operational_eligibility"]["status_counts"], {
            "expired": 2, "eligible": 1, "unknown": 1, "conflict": 1,
        })
        self.assertEqual(report["operational_eligibility"]["pending_by_status"], {
            "skip_expired": 2, "needs_triage_active": 1, "needs_triage_deadline_unknown": 1,
            "needs_triage_deadline_conflict": 1,
        })
        # limit=1 относится к платным кандидатам; два skip_expired сверх limit
        self.assertEqual(report["candidate_count"], 3)
        self.assertEqual(report["routing_preview"]["skip_expired"], 2)
        self.assertEqual(report["evaluated_at"], NOW.isoformat())
        # triage-оценка стоимости не включает истёкшие
        self.assertEqual(report["triage_preflight"]["candidate_count"], 1)

    def test_dry_run_cli_prints_operational_section(self):
        self.add_deadline(1, PAST)
        with mock.patch("sys.stdout", io.StringIO()) as stdout:
            code = analysis_pipeline.main(["--dry-run", "--db-path", str(self.db_path)])
        self.assertEqual(code, 0)
        text = stdout.getvalue()
        self.assertIn("Operational Eligibility", text)
        self.assertIn("skip_expired", text)


class OneShotTests(DeadlineCase):
    def run_one(self, url, confirm=False, allow_expired=False, settings=None):
        lines = []
        code = one_shot.run_one(
            url, self.triage, self.deep, settings or self.settings, db_path=self.db_path, confirm=confirm,
            out=lines.append, allow_expired=allow_expired, clock=self.clock,
        )
        return code, "\n".join(lines)

    def test_preflight_shows_operational_block_with_zero_writes(self):
        url = self.add_deadline(1, PAST, "2026-09-29 10:00:00")
        before = hashlib.sha256(self.db_path.read_bytes()).hexdigest()
        code, text = self.run_one(url)
        self.assertEqual(code, 0)
        for label in ("Operational Eligibility:", "status: expired", "resolved deadline: 2026-09-30T11:00:00+04:00",
                      oe.SOURCE_LIST, oe.SOURCE_DETAIL, "evaluation time: 2026-10-01T12:00:00+04:00",
                      "expired: true", "operator override (--allow-expired): false"):
            self.assertIn(label, text)
        self.assertEqual(hashlib.sha256(self.db_path.read_bytes()).hexdigest(), before)
        self.assert_zero_spend()

    def test_expired_with_confirm_but_without_allow_expired_makes_no_paid_call(self):
        url = self.add_deadline(1, PAST)
        code, text = self.run_one(url, confirm=True)
        self.assertEqual(code, one_shot.EXIT_OK)
        self.assertEqual(self.state(url)["state"], state_repo.STATE_SKIPPED_EXPIRED)
        self.assertIn("--allow-expired", text)
        self.assert_zero_spend()

    def test_allow_expired_without_confirm_makes_no_paid_call(self):
        url = self.add_deadline(1, PAST)
        code, text = self.run_one(url, allow_expired=True)
        self.assertEqual(code, 0)
        self.assertIn("operator override (--allow-expired): true", text)
        self.assertIn("DRY-RUN", text)
        self.assertIsNone(self.state(url))
        self.assert_zero_spend()

    def test_allow_expired_with_confirm_runs_existing_pipeline(self):
        url = self.add_deadline(1, PAST, estimated_value_amd="5000000")  # реальный Commercial Gate пропускает
        code, text = self.run_one(url, confirm=True, allow_expired=True)
        self.assertEqual(code, 0)
        self.assertEqual((self.triage.calls, self.deep.calls), ([url], [url]))
        self.assertEqual(self.state(url)["state"], state_repo.STATE_DEEP_COMPLETED)
        self.assertIn("API calls actually made: 2", text)

    def test_allow_expired_does_not_bypass_budget(self):
        url = self.add_deadline(1, PAST)
        tiny = BudgetSettings(monthly_budget_usd=Decimal("0.000001"), soft_limit_usd=Decimal("0.0000005"))
        self.run_one(url, confirm=True, allow_expired=True, settings=tiny)
        self.assertEqual(self.state(url)["state"], state_repo.STATE_TRIAGE_DEFERRED_BUDGET)
        self.assert_zero_spend()

    def test_allow_expired_does_not_bypass_accounting_block(self):
        url = self.add_deadline(1, PAST)
        ai_usage_repository.reserve_call("triage", "m", Decimal("0.01"), reference="other", db_path=self.db_path)
        code, text = self.run_one(url, confirm=True, allow_expired=True)
        self.assertEqual(code, one_shot.EXIT_FAILED)
        self.assertIn("accounting_blocked", text)
        self.assertEqual(self.triage.calls, [])

    def test_existing_deep_result_not_overwritten_by_expiry_in_one_shot(self):
        url = self.add_deadline(1, "2026-10-05 11:00:00", estimated_value_amd="5000000")
        self.run_one(url, confirm=True)
        deep_before = analysis_repository.get_deep_analysis(url, db_path=self.db_path)
        self.now = datetime(2026, 10, 6, 12, 0, tzinfo=YEREVAN)
        code, text = self.run_one(url, confirm=True)
        self.assertEqual(code, 0)
        self.assertIn("status: expired", text)
        self.assertEqual(self.state(url)["state"], state_repo.STATE_DEEP_COMPLETED)
        self.assertEqual(analysis_repository.get_deep_analysis(url, db_path=self.db_path), deep_before)
        self.assertEqual((len(self.triage.calls), len(self.deep.calls)), (1, 1))


class CliTests(DeadlineCase):
    def test_allow_expired_requires_run_one(self):
        for argv in (["--dry-run", "--allow-expired"], ["--dry-run", "--allow-expired", "--confirm-paid-call"]):
            with self.assertRaises(SystemExit), mock.patch("sys.stderr", io.StringIO()):
                analysis_pipeline.main(argv)

    def test_allow_expired_is_passed_to_run_one_only_when_given(self):
        with mock.patch("dotenv.load_dotenv"), mock.patch.object(one_shot, "run_one", return_value=0) as run_one:
            analysis_pipeline.main(["--run-one", "u", "--db-path", str(self.db_path)])
            self.assertFalse(run_one.call_args.kwargs["allow_expired"])
            analysis_pipeline.main(["--run-one", "u", "--allow-expired", "--db-path", str(self.db_path)])
            self.assertTrue(run_one.call_args.kwargs["allow_expired"])


class MonitorTests(unittest.TestCase):
    def test_ai_disabled_behaviour_unchanged(self):
        env = {k: v for k, v in os.environ.items() if k != "AI_ANALYSIS_ENABLED"}
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch.object(analysis_pipeline, "run_ai_analysis", side_effect=AssertionError("must not run")):
            self.assertIsNone(monitor._run_ai_analysis_if_enabled())


if __name__ == "__main__":
    unittest.main()
