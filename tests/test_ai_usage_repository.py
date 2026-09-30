"""
Тесты src.database.ai_usage_repository на временной SQLite БД (production data/tenders.db
недоступна: tests/safety_guards.py). Append-only, месячные агрегаты, rollover по UTC.

Запуск: python -m unittest tests.test_ai_usage_repository -v
"""

import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from src.ai import pricing
from src.database import ai_usage_repository as ledger
from tests import safety_guards

USAGE = {"input_tokens": 1000, "output_tokens": 500, "cached_tokens": 200, "reasoning_tokens": 300, "total_tokens": 1500}


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.db = Path(self.tempdir.name) / "usage.db"
        ledger.init_db(self.db)

    def record(self, **kwargs):
        defaults = dict(reference="https://example.test/1", now=utc(2026, 9, 10, 12), db_path=self.db)
        defaults.update(kwargs)
        analysis_type = defaults.pop("analysis_type", "triage")
        model = defaults.pop("model", "gpt-5.6-luna")
        usage = defaults.pop("usage", USAGE)
        return ledger.record_usage(analysis_type, model, usage, **defaults)

    def test_append_event_stores_cost_as_exact_decimal(self):
        cost = self.record(response_id="resp_1", prompt_version="procurement-v2")
        # 800*0.20 + 200*0.02 + 500*1.20 = 160 + 4 + 600 = 764 / 1e6
        self.assertEqual(cost, Decimal("0.000764"))
        with closing(sqlite3.connect(self.db)) as conn:
            row = conn.execute(
                "SELECT analysis_type, model, reference, response_id, input_tokens, cached_input_tokens,"
                " output_tokens, reasoning_tokens, estimated_cost_usd, pricing_version, prompt_version, success,"
                " error_kind, created_at FROM ai_usage_events"
            ).fetchone()
        self.assertEqual(row, (
            "triage", "gpt-5.6-luna", "https://example.test/1", "resp_1", 1000, 200, 500, 300, "0.000764",
            pricing.PRICING_VERSION, "procurement-v2", 1, None, "2026-09-10T12:00:00+00:00",
        ))

    def test_response_id_optional(self):
        self.record()
        with closing(sqlite3.connect(self.db)) as conn:
            self.assertIsNone(conn.execute("SELECT response_id FROM ai_usage_events").fetchone()[0])

    def test_failed_attempt_with_usage_is_recorded_as_unsuccessful(self):
        self.record(success=False, error_kind="validation")
        with closing(sqlite3.connect(self.db)) as conn:
            self.assertEqual(conn.execute("SELECT success, error_kind FROM ai_usage_events").fetchone(), (0, "validation"))

    def test_no_usage_means_no_event_and_no_fake_cost(self):
        self.assertIsNone(self.record(usage=None))
        self.assertIsNone(self.record(usage={"input_tokens": None, "output_tokens": None}))
        self.assertEqual(ledger.monthly_usage(2026, 9, self.db)["events"], 0)

    def test_unknown_model_not_recorded_with_invented_price(self):
        with self.assertRaises(pricing.CostEstimationError):
            self.record(model="gpt-unknown")
        self.assertEqual(ledger.monthly_usage(2026, 9, self.db)["events"], 0)

    def test_naive_timestamp_rejected(self):
        with self.assertRaises(ValueError):
            self.record(now=datetime(2026, 9, 1))

    def test_monthly_cost_and_usage_totals(self):
        self.record()
        self.record(model="gpt-5.6-terra", analysis_type="deep")
        luna, terra = Decimal("0.000764"), Decimal("0.007640")
        self.assertEqual(ledger.monthly_cost(2026, 9, self.db), luna + terra)
        usage = ledger.monthly_usage(2026, 9, self.db)
        self.assertEqual(usage["events"], 2)
        self.assertEqual(usage["input_tokens"], 2000)
        self.assertEqual(usage["cached_input_tokens"], 400)
        self.assertEqual(usage["output_tokens"], 1000)
        self.assertEqual(usage["reasoning_tokens"], 600)
        self.assertEqual(usage["cost_usd"], luna + terra)

    def test_month_rollover_by_utc(self):
        self.record(now=utc(2026, 9, 30, 23, 59, 59))
        self.record(now=utc(2026, 10, 1, 0, 0, 0))
        self.record(now=utc(2026, 8, 31, 23, 59, 59))
        self.assertEqual(ledger.monthly_usage(2026, 9, self.db)["events"], 1)
        self.assertEqual(ledger.monthly_usage(2026, 10, self.db)["events"], 1)
        self.assertEqual(ledger.monthly_usage(2026, 8, self.db)["events"], 1)

    def test_non_utc_timestamp_is_normalized_to_utc_month(self):
        tz = timezone(timedelta(hours=4))  # 2026-10-01 02:00 +04:00 == 2026-09-30 22:00 UTC
        self.record(now=datetime(2026, 10, 1, 2, tzinfo=tz))
        self.assertEqual(ledger.monthly_usage(2026, 9, self.db)["events"], 1)
        self.assertEqual(ledger.monthly_usage(2026, 10, self.db)["events"], 0)

    def test_december_rollover_to_january(self):
        self.record(now=utc(2026, 12, 31, 23))
        self.assertEqual(ledger.monthly_usage(2026, 12, self.db)["events"], 1)
        self.assertEqual(ledger.monthly_usage(2027, 1, self.db)["events"], 0)

    def test_current_month_cost_uses_given_now(self):
        self.record(now=utc(2026, 9, 5))
        self.assertEqual(ledger.current_month_cost(utc(2026, 9, 29), self.db), Decimal("0.000764"))
        self.assertEqual(ledger.current_month_cost(utc(2026, 10, 1), self.db), Decimal(0))

    def test_cost_by_model_and_analysis_type(self):
        self.record()
        self.record()
        self.record(model="gpt-5.6-terra", analysis_type="deep")
        self.assertEqual(ledger.cost_by_model(2026, 9, self.db), {
            "gpt-5.6-luna": Decimal("0.001528"), "gpt-5.6-terra": Decimal("0.007640"),
        })
        self.assertEqual(ledger.cost_by_analysis_type(2026, 9, self.db), {
            "triage": Decimal("0.001528"), "deep": Decimal("0.007640"),
        })

    def test_ledger_is_append_only(self):
        self.record()
        with closing(sqlite3.connect(self.db)) as conn:
            with self.assertRaises(sqlite3.DatabaseError):
                conn.execute("UPDATE ai_usage_events SET estimated_cost_usd = '0'")
            with self.assertRaises(sqlite3.DatabaseError):
                conn.execute("DELETE FROM ai_usage_events")
        self.assertEqual(ledger.monthly_usage(2026, 9, self.db)["events"], 1)

    def test_no_api_key_column_or_secret_storage(self):
        with closing(sqlite3.connect(self.db)) as conn:
            columns = [row[1] for row in conn.execute("PRAGMA table_info(ai_usage_events)")]
        self.assertFalse([name for name in columns if "key" in name or "secret" in name or "token" == name])

    def test_init_db_is_idempotent(self):
        self.record()
        ledger.init_db(self.db)
        self.assertEqual(ledger.monthly_usage(2026, 9, self.db)["events"], 1)

    def test_default_path_is_not_touched_by_tests(self):
        # tests/safety_guards.py блокирует production data/tenders.db; явный temp path обязателен.
        with safety_guards.expect_violation(self, safety_guards.DATABASE_MESSAGE):
            ledger.init_db()


if __name__ == "__main__":
    unittest.main()
