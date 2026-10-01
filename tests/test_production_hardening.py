"""
Тесты production-hardening: семантика барьеров Deep prompt v5, primary/fallback конфиг моделей,
cache_write_tokens (extraction, ledger + миграция, симулятор, artifacts), причины эскалации.
Реальных OpenAI-вызовов нет: только fake-объекты и временные SQLite БД.

Запуск: python -m unittest tests.test_production_hardening -v
"""

import io
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing, redirect_stdout
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src.ai import budget_settings, cost_simulator, deep_analysis_evaluator, deep_prompt, escalation
from src.ai import evidence_catalog, openai_deep_analysis, openai_triage, pricing, relevance_schema
from src.database import ai_usage_repository as ledger
from tests.test_evidence_catalog import build, by_text, raw_output

LUNA, TERRA = "gpt-5.6-luna", "gpt-5.6-terra"
SRC_DIR = Path(deep_prompt.__file__).parent


# --------------------------------------------------------------------------
# 1. barrier semantics (prompt v5)
# --------------------------------------------------------------------------

QUALIFICATION_TEXT = "Обеспечение квалификации участника: 15% от суммы"
PERFORMANCE_TEXT = "Обеспечение исполнения договора: 10% от цены договора"


class BarrierSemanticsPromptTests(unittest.TestCase):
    def test_prompt_version_is_v6(self):
        self.assertEqual(deep_prompt.DEEP_PROMPT_VERSION, "procurement-deep-v6")
        self.assertEqual(openai_deep_analysis.OpenAIDeepAnalysisAnalyzer(environ={}).prompt_version, "procurement-deep-v6")

    def test_prompt_defines_the_three_money_barrier_types_by_purpose(self):
        prompt = " ".join(deep_prompt.SYSTEM_PROMPT.split())
        for phrase in (
            '"bid_security": security for the bid/application itself',
            '"contract_security": security for performance of the concluded contract',
            '"financial_requirement": qualification security / qualification guarantee',
            "NOT security for the bid and NOT security for contract performance",
            "give each distinct requirement exactly ONE type",
        ):
            self.assertIn(phrase, prompt)

    def test_prompt_examples_map_qualification_and_contract_security(self):
        prompt = deep_prompt.SYSTEM_PROMPT
        self.assertIn('securing the participant\'s qualification ->\n  "financial_requirement"', prompt)
        self.assertIn('securing contract performance ->\n  "contract_security"', prompt)

    def test_schema_keeps_barrier_enum_and_evidence_id_architecture(self):
        schema = deep_prompt.build_deep_output_schema()
        barrier = schema["properties"]["participation_barriers"]["items"]
        self.assertEqual(barrier["properties"]["type"]["enum"], list(relevance_schema.PARTICIPATION_BARRIER_TYPES))
        self.assertIn("evidence_ids", barrier["properties"])
        for name in ("bid_security", "contract_security", "financial_requirement"):
            self.assertIn(name, relevance_schema.PARTICIPATION_BARRIER_TYPES)

    def test_no_keyword_postprocessing_of_barrier_types_in_code(self):
        for name in ("evidence_catalog.py", "openai_deep_analysis.py", "relevance_schema.py"):
            source = (SRC_DIR / name).read_text(encoding="utf-8").lower()
            self.assertNotIn("qualification", source, name)
            self.assertNotIn("performance security", source, name)


class BarrierRegressionFixtureTests(unittest.TestCase):
    """Medical benchmark: 15% qualification security и 10% contract security — разные барьеры."""

    def setUp(self):
        self.context = build([], enrichment={"description": f"{QUALIFICATION_TEXT}\n{PERFORMANCE_TEXT}"})
        self.qualification_id = by_text(self.context, QUALIFICATION_TEXT)["evidence_id"]
        self.performance_id = by_text(self.context, PERFORMANCE_TEXT)["evidence_id"]

    def barrier(self, barrier_type, evidence_id, description):
        return {"type": barrier_type, "description": description, "severity": "high", "evidence_ids": [evidence_id]}

    def validate(self, barriers):
        raw = raw_output(
            evidence_ids=["ev_ann_title_0"], participation_barriers=barriers, opportunity_type="unclear", category=None,
        )
        materialized = evidence_catalog.materialize_deep_model_output(raw, self.context["evidence_catalog"])
        return relevance_schema.validate_deep_analysis_result(materialized)

    def test_expected_classification_passes_validation_unchanged(self):
        validated = self.validate([
            self.barrier("financial_requirement", self.qualification_id, "Обеспечение квалификации 15%"),
            self.barrier("contract_security", self.performance_id, "Обеспечение исполнения договора 10%"),
        ])
        by_type = {b["type"]: b for b in validated["participation_barriers"]}
        self.assertEqual(set(by_type), {"financial_requirement", "contract_security"})
        self.assertEqual(by_type["financial_requirement"]["evidence"][0]["text"], QUALIFICATION_TEXT)
        self.assertEqual(by_type["contract_security"]["evidence"][0]["text"], PERFORMANCE_TEXT)

    def test_bid_security_remains_a_valid_distinct_type(self):
        validated = self.validate([self.barrier("bid_security", self.qualification_id, "Обеспечение заявки")])
        self.assertEqual(validated["participation_barriers"][0]["type"], "bid_security")

    def test_code_does_not_rewrite_model_barrier_types(self):
        # даже «неправильная» классификация не переписывается кодом: семантика — в prompt, не в постобработке
        validated = self.validate([
            self.barrier("contract_security", self.qualification_id, "d1"),
            self.barrier("contract_security", self.performance_id, "d2"),
        ])
        self.assertEqual([b["type"] for b in validated["participation_barriers"]], ["contract_security"] * 2)


# --------------------------------------------------------------------------
# 2. model config
# --------------------------------------------------------------------------

class ModelConfigTests(unittest.TestCase):
    def test_defaults_luna_primary_terra_fallback(self):
        settings = budget_settings.load_budget_settings({})
        self.assertEqual((settings.triage_model, settings.deep_primary_model, settings.deep_fallback_model),
                         (LUNA, LUNA, TERRA))
        self.assertEqual(openai_deep_analysis.load_settings({})["model"], LUNA)
        self.assertEqual(openai_triage.load_settings({})["model"], LUNA)
        self.assertEqual(
            (settings.triage_reasoning_effort, settings.deep_primary_reasoning_effort,
             settings.deep_fallback_reasoning_effort),
            ("medium", "high", "high"),
        )

    def test_deep_evaluator_precedence_chain(self):
        load = openai_deep_analysis.load_settings
        env = {"OPENAI_MODEL": "m-openai", "DEEP_MODEL": "m-legacy", "DEEP_PRIMARY_MODEL": "m-primary"}
        self.assertEqual(load(env)["model"], "m-primary")
        del env["DEEP_PRIMARY_MODEL"]
        self.assertEqual(load(env)["model"], "m-legacy")
        del env["DEEP_MODEL"]
        self.assertEqual(load(env)["model"], "m-openai")
        self.assertEqual(load(env, model="m-cli")["model"], "m-cli")
        full = {"DEEP_PRIMARY_MODEL": "m-primary", "DEEP_MODEL": "m-legacy"}
        self.assertEqual(load(full, model="m-cli")["model"], "m-cli")

    def test_deep_evaluator_effort_chain(self):
        load = openai_deep_analysis.load_settings
        env = {"OPENAI_DEEP_REASONING_EFFORT": "low"}
        self.assertEqual(load(env)["reasoning_effort"], "low")
        env["DEEP_REASONING_EFFORT"] = "medium"
        self.assertEqual(load(env)["reasoning_effort"], "medium")
        env["DEEP_PRIMARY_REASONING_EFFORT"] = "xhigh"
        self.assertEqual(load(env)["reasoning_effort"], "xhigh")
        self.assertEqual(load(env, reasoning_effort="low")["reasoning_effort"], "low")

    def test_fallback_model_env_does_not_change_deep_evaluator_model(self):
        self.assertEqual(openai_deep_analysis.load_settings({"DEEP_FALLBACK_MODEL": TERRA})["model"], LUNA)

    def test_triage_evaluator_precedence_chain(self):
        load = openai_triage.load_settings
        env = {"OPENAI_MODEL": "m-openai", "TRIAGE_MODEL": "m-triage"}
        self.assertEqual(load(env)["model"], "m-triage")
        self.assertEqual(load({"OPENAI_MODEL": "m-openai"})["model"], "m-openai")
        self.assertEqual(load(env, model="m-cli")["model"], "m-cli")
        self.assertEqual(load({"TRIAGE_REASONING_EFFORT": "high", "OPENAI_TRIAGE_REASONING_EFFORT": "low"})["reasoning_effort"], "high")

    def test_cli_model_flag_overrides_env_and_default(self):
        parser = deep_analysis_evaluator._build_arg_parser()
        env = {"DEEP_PRIMARY_MODEL": "m-env"}
        with mock.patch("dotenv.load_dotenv"), mock.patch.dict(os.environ, env, clear=True):
            cli = deep_analysis_evaluator._default_analyzer(parser.parse_args(["--dry-run", "--model", "m-cli"]))
            plain = deep_analysis_evaluator._default_analyzer(parser.parse_args(["--dry-run"]))
        self.assertEqual(cli.model, "m-cli")
        self.assertEqual(plain.model, "m-env")
        with mock.patch("dotenv.load_dotenv"), mock.patch.dict(os.environ, {}, clear=True):
            default = deep_analysis_evaluator._default_analyzer(parser.parse_args(["--dry-run", "--model", LUNA]))
        self.assertEqual(default.model, LUNA)


# --------------------------------------------------------------------------
# 3. usage extraction
# --------------------------------------------------------------------------

def response_with_details(input_details):
    return SimpleNamespace(usage=SimpleNamespace(
        input_tokens=1000, output_tokens=500, total_tokens=1500,
        input_tokens_details=input_details,
        output_tokens_details=SimpleNamespace(reasoning_tokens=300),
    ))


class UsageExtractionTests(unittest.TestCase):
    def test_cache_write_tokens_extracted(self):
        details = SimpleNamespace(cached_tokens=100, cache_write_tokens=400)
        usage = openai_triage.usage_dict(response_with_details(details))
        self.assertEqual(usage, {
            "input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500, "cached_tokens": 100,
            "cache_write_tokens": 400, "reasoning_tokens": 300,
        })

    def test_missing_cache_write_tokens_is_zero(self):
        usage = openai_triage.usage_dict(response_with_details(SimpleNamespace(cached_tokens=100)))
        self.assertEqual(usage["cache_write_tokens"], 0)
        self.assertEqual(openai_triage.usage_dict(response_with_details(None))["cache_write_tokens"], 0)
        none_value = SimpleNamespace(cached_tokens=0, cache_write_tokens=None)
        self.assertEqual(openai_triage.usage_dict(response_with_details(none_value))["cache_write_tokens"], 0)

    def test_cost_from_extracted_usage_includes_cache_write(self):
        usage = openai_triage.usage_dict(
            response_with_details(SimpleNamespace(cached_tokens=100, cache_write_tokens=400))
        )
        expected = (Decimal(500) * Decimal("0.20") + Decimal(100) * Decimal("0.02")
                    + Decimal(400) * Decimal("0.25") + Decimal(500) * Decimal("1.20")) / 1_000_000
        self.assertEqual(pricing.cost_from_usage(LUNA, usage), expected)


# --------------------------------------------------------------------------
# 8. ledger + migration
# --------------------------------------------------------------------------

OLD_SCHEMA = """
CREATE TABLE ai_usage_events (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT NOT NULL,
    analysis_type       TEXT NOT NULL,
    model               TEXT NOT NULL,
    reference           TEXT,
    response_id         TEXT,
    input_tokens        INTEGER NOT NULL,
    cached_input_tokens INTEGER NOT NULL,
    output_tokens       INTEGER NOT NULL,
    reasoning_tokens    INTEGER NOT NULL,
    estimated_cost_usd  TEXT NOT NULL,
    pricing_version     TEXT NOT NULL,
    prompt_version      TEXT,
    success             INTEGER NOT NULL,
    error_kind          TEXT
)
"""
OLD_ROW = (
    "2026-09-05T10:00:00+00:00", "triage", LUNA, "https://old.example/1", "resp_old",
    1000, 200, 500, 300, "0.000764", "2026-09-29", "procurement-v2", 1, None,
)
INSERT_OLD = (
    "INSERT INTO ai_usage_events (created_at, analysis_type, model, reference, response_id, input_tokens,"
    " cached_input_tokens, output_tokens, reasoning_tokens, estimated_cost_usd, pricing_version,"
    " prompt_version, success, error_kind) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


def columns(db):
    with closing(sqlite3.connect(db)) as conn:
        return [row[1] for row in conn.execute("PRAGMA table_info(ai_usage_events)")]


class LedgerMigrationTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.db = Path(self.tempdir.name) / "usage.db"

    def make_old_db(self):
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute(OLD_SCHEMA)
            conn.execute(
                "CREATE TRIGGER ai_usage_events_no_update BEFORE UPDATE ON ai_usage_events"
                " BEGIN SELECT RAISE(ABORT, 'ai_usage_events is append-only'); END"
            )
            conn.execute(INSERT_OLD, OLD_ROW)

    def test_new_table_has_cache_write_column(self):
        ledger.init_db(self.db)
        self.assertIn("cache_write_tokens", columns(self.db))

    def test_old_schema_is_migrated_and_old_rows_default_to_zero(self):
        self.make_old_db()
        self.assertNotIn("cache_write_tokens", columns(self.db))
        ledger.init_db(self.db)
        self.assertIn("cache_write_tokens", columns(self.db))
        with closing(sqlite3.connect(self.db)) as conn:
            self.assertEqual(conn.execute("SELECT cache_write_tokens, estimated_cost_usd FROM ai_usage_events").fetchall(),
                             [(0, "0.000764")])

    def test_migration_is_idempotent(self):
        self.make_old_db()
        for _ in range(3):
            ledger.init_db(self.db)
        self.assertEqual(columns(self.db).count("cache_write_tokens"), 1)
        with closing(sqlite3.connect(self.db)) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM ai_usage_events").fetchone()[0], 1)

    def test_old_rows_remain_valid_in_month_aggregates(self):
        self.make_old_db()
        ledger.init_db(self.db)
        self.assertEqual(ledger.monthly_cost(2026, 9, self.db), Decimal("0.000764"))
        usage = ledger.monthly_usage(2026, 9, self.db)
        self.assertEqual((usage["events"], usage["input_tokens"], usage["cache_write_tokens"]), (1, 1000, 0))

    def test_reads_and_writes_migrate_lazily_without_init_db(self):
        self.make_old_db()
        self.assertEqual(ledger.monthly_cost(2026, 9, self.db), Decimal("0.000764"))
        self.assertIn("cache_write_tokens", columns(self.db))

    def test_record_usage_on_migrated_db_stores_cache_write(self):
        self.make_old_db()
        ledger.init_db(self.db)
        usage = {"input_tokens": 100_000, "output_tokens": 1000, "cached_tokens": 10_000,
                 "cache_write_tokens": 40_000, "reasoning_tokens": 200}
        cost = ledger.record_usage(
            "deep", LUNA, usage, reference="https://new.example/1", response_id="resp_new",
            prompt_version="procurement-deep-v6", now=self._now(), db_path=self.db,
        )
        expected = (Decimal(50_000) * Decimal("0.20") + Decimal(10_000) * Decimal("0.02")
                    + Decimal(40_000) * Decimal("0.25") + Decimal(1000) * Decimal("1.20")) / 1_000_000
        self.assertEqual(cost, expected)
        with closing(sqlite3.connect(self.db)) as conn:
            row = conn.execute(
                "SELECT cache_write_tokens, cached_input_tokens, reasoning_tokens, response_id, prompt_version,"
                " success FROM ai_usage_events WHERE analysis_type = 'deep'"
            ).fetchone()
        self.assertEqual(row, (40_000, 10_000, 200, "resp_new", "procurement-deep-v6", 1))
        totals = ledger.monthly_usage(2026, 9, self.db)
        self.assertEqual(totals["cache_write_tokens"], 40_000)
        self.assertEqual(totals["cost_usd"], expected + Decimal("0.000764"))

    def test_legacy_usage_without_cache_write_records_zero(self):
        ledger.init_db(self.db)
        ledger.record_usage("triage", LUNA, {"input_tokens": 10, "output_tokens": 5}, now=self._now(), db_path=self.db)
        with closing(sqlite3.connect(self.db)) as conn:
            self.assertEqual(conn.execute("SELECT cache_write_tokens FROM ai_usage_events").fetchone()[0], 0)

    def test_inconsistent_usage_is_not_recorded(self):
        ledger.init_db(self.db)
        usage = {"input_tokens": 100, "output_tokens": 5, "cached_tokens": 80, "cache_write_tokens": 50}
        with self.assertRaises(pricing.CostEstimationError):
            ledger.record_usage("triage", LUNA, usage, now=self._now(), db_path=self.db)
        with closing(sqlite3.connect(self.db)) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM ai_usage_events").fetchone()[0], 0)

    def test_append_only_still_enforced_after_migration(self):
        self.make_old_db()
        ledger.init_db(self.db)
        with closing(sqlite3.connect(self.db)) as conn, self.assertRaises(sqlite3.DatabaseError):
            conn.execute("UPDATE ai_usage_events SET model = 'x'")
        with closing(sqlite3.connect(self.db)) as conn, self.assertRaises(sqlite3.DatabaseError):
            conn.execute("DELETE FROM ai_usage_events")

    @staticmethod
    def _now():
        from datetime import datetime, timezone

        return datetime(2026, 9, 10, 12, tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# 10. cost simulator
# --------------------------------------------------------------------------

class SimulatorCacheWriteTests(unittest.TestCase):
    def simulate(self, **kwargs):
        params = dict(announcements_per_day=30, days_per_month=30, deep_per_month=30)
        params.update(kwargs)
        return cost_simulator.simulate(**params)

    def test_defaults_are_backward_compatible_and_use_luna_primary(self):
        result = self.simulate()
        self.assertEqual(result["deep_model"], LUNA)
        self.assertEqual(result["triage_cost_usd"], Decimal("1.062"))
        self.assertEqual(result["deep_tokens_per_call"]["cache_write_input"], 0)
        self.assertEqual(result["triage_tokens_per_call"]["cache_write_input"], 0)

    def test_deep_cache_write_scenario(self):
        base = self.simulate()
        result = self.simulate(deep_cache_write_tokens=70_000)
        # 5473 ordinary*0.20 + 70000 write*0.25 + 4350 out*1.20 (/1e6)
        expected_unit = (Decimal(5473) * Decimal("0.20") + Decimal(70_000) * Decimal("0.25")
                         + Decimal(4350) * Decimal("1.20")) / 1_000_000
        self.assertEqual(result["deep_cost_per_call_usd"], expected_unit)
        self.assertEqual(result["deep_cost_usd"], expected_unit * 30)
        self.assertGreater(result["deep_cost_usd"], base["deep_cost_usd"])
        self.assertEqual(result["deep_tokens_per_call"],
                         {"ordinary_input": 5473, "cached_input": 0, "cache_write_input": 70_000, "output": 4350})

    def test_triage_cache_write_and_cached_are_distinguished(self):
        result = self.simulate(triage_cached_input_tokens=1000, triage_cache_write_tokens=2000)
        self.assertEqual(result["triage_tokens_per_call"],
                         {"ordinary_input": 1100, "cached_input": 1000, "cache_write_input": 2000, "output": 300})
        unit = (Decimal(1100) * Decimal("0.20") + Decimal(1000) * Decimal("0.02")
                + Decimal(2000) * Decimal("0.25") + Decimal(300) * Decimal("1.20")) / 1_000_000
        self.assertEqual(result["triage_cost_per_call_usd"], unit)

    def test_totals_by_component_and_overall(self):
        result = self.simulate(deep_cache_write_tokens=1000)
        self.assertEqual(result["total_usd"], result["triage_cost_usd"] + result["deep_cost_usd"])
        report = cost_simulator.format_report(result)
        self.assertIn("monthly totals: triage", report)
        self.assertIn("deep primary", report)
        self.assertIn("overall", report)
        self.assertIn("cache-write input 1000", report)

    def test_inconsistent_split_is_rejected(self):
        with self.assertRaises(pricing.CostEstimationError):
            self.simulate(deep_input_tokens=1000, deep_cached_input_tokens=600, deep_cache_write_tokens=600)

    def test_cli_accepts_new_flags_and_old_flags(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = cost_simulator.main([
                "--announcements-per-day", "30", "--deep-per-month", "30",
                "--triage-cache-write-tokens", "500", "--deep-cache-write-tokens", "70000",
                "--deep-cached-input-tokens", "1000", "--triage-cached-input-tokens", "100",
            ])
        self.assertEqual(code, 0)
        self.assertIn("cache-write input 70000", out.getvalue())
        with redirect_stdout(io.StringIO()):
            self.assertEqual(cost_simulator.main([
                "--announcements-per-day", "1", "--deep-per-month", "1", "--deep-input-tokens", "10",
                "--deep-cache-write-tokens", "20",
            ]), 2)

    def test_terra_can_still_be_simulated_explicitly(self):
        luna = self.simulate()["deep_cost_usd"]
        self.assertEqual(self.simulate(deep_model=TERRA)["deep_cost_usd"], luna * 10)


# --------------------------------------------------------------------------
# 9. evaluator artifacts
# --------------------------------------------------------------------------

class EvaluatorUsageOutputTests(unittest.TestCase):
    def test_deep_artifact_usage_and_cost_include_cache_write(self):
        from datetime import datetime, timezone

        analyzer = SimpleNamespace(model=LUNA, reasoning_effort="high", prompt_version="p", provider="openai")
        legacy = {"usage": {"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500,
                            "cached_tokens": 0, "reasoning_tokens": 200}}
        started = datetime(2026, 9, 30, tzinfo=timezone.utc)
        artifact = deep_analysis_evaluator.build_artifact(legacy, analyzer, "golden.json", started)
        self.assertEqual(artifact["usage"], {
            "input_tokens": 1000, "cached_tokens": 0, "cache_write_tokens": 0, "output_tokens": 500,
            "reasoning_tokens": 200, "total_tokens": 1500,
        })
        self.assertEqual(artifact["cost"]["estimated_cost_usd"], str(pricing.calculate_cost(LUNA, 1000, 500)))

        written = {"usage": dict(legacy["usage"], cached_tokens=100, cache_write_tokens=300)}
        artifact = deep_analysis_evaluator.build_artifact(written, analyzer, "golden.json", started)
        self.assertEqual(artifact["usage"]["cache_write_tokens"], 300)
        expected = pricing.calculate_cost(LUNA, 1000, 500, cached_input_tokens=100, cache_write_tokens=300)
        self.assertEqual(Decimal(artifact["cost"]["estimated_cost_usd"]), expected)
        json.dumps(artifact)  # сериализуется

    def test_missing_usage_gives_zeroed_usage(self):
        from datetime import datetime, timezone

        analyzer = SimpleNamespace(model=LUNA)
        artifact = deep_analysis_evaluator.build_artifact(
            {"usage": None}, analyzer, "g.json", datetime(2026, 9, 30, tzinfo=timezone.utc),
        )
        self.assertEqual(artifact["usage"], {name: 0 for name in pricing.USAGE_FIELDS})


# --------------------------------------------------------------------------
# 11. escalation reasons
# --------------------------------------------------------------------------

class EscalationReasonTests(unittest.TestCase):
    def test_reason_vocabulary(self):
        self.assertEqual(escalation.ALL_REASONS, {
            "validation_failure", "api_failure", "document_coverage_incomplete", "critical_source_conflict",
            "high_value_manual_review", "deep_input_limit_exceeded", "budget_deferred",
        })

    def test_deferrals_are_not_escalations(self):
        self.assertEqual(escalation.ESCALATION_REASONS & escalation.DEFERRAL_REASONS, set())
        self.assertFalse(escalation.is_escalation_reason(escalation.BUDGET_DEFERRED))
        self.assertFalse(escalation.is_escalation_reason(escalation.DEEP_INPUT_LIMIT_EXCEEDED))
        self.assertTrue(escalation.is_escalation_reason(escalation.VALIDATION_FAILURE))

    def test_manual_review_and_medium_confidence_alone_never_escalate(self):
        for value in ("manual_review_required", "manual_review_required=true", "confidence=medium", "medium", None, ""):
            self.assertFalse(escalation.is_escalation_reason(value), value)
            self.assertNotIn(value, escalation.ALL_REASONS)


if __name__ == "__main__":
    unittest.main()
