"""
Тесты операторского batch (src.ai.operator_batch): только fake analyzers, временная БД, ZERO OpenAI.

    python -m unittest tests.test_operator_batch -v
"""

import hashlib
import io
import os
import unittest
from decimal import Decimal
from unittest import mock

from src.ai import analysis_pipeline, one_shot, openai_triage, operator_batch
from src.ai.budget_settings import BudgetSettings
from src.database import ai_usage_repository, pipeline_state_repository as state_repo
from tests.test_analysis_pipeline import USAGE, gate_pass, triage_result
from tests.test_operational_eligibility import FUTURE, PAST, DeadlineCase


class BatchCase(DeadlineCase):
    def batch(self, limit=5, confirm=False, settings=None, gate=gate_pass):
        lines = []
        code = operator_batch.run_batch(
            limit, self.triage, self.deep, settings or self.settings, db_path=self.db_path, confirm=confirm,
            out=lines.append, clock=self.clock, gate=gate,
        )
        return code, "\n".join(lines)

    def digest(self):
        return hashlib.sha256(self.db_path.read_bytes()).hexdigest()

    def calls(self):
        return len(self.triage.calls) + len(self.deep.calls)


class CliTests(BatchCase):
    def cli(self, argv):
        with mock.patch("sys.stderr", io.StringIO()) as err, self.assertRaises(SystemExit):
            analysis_pipeline.main(argv)
        return err.getvalue()

    def test_requires_limit(self):  # 1
        self.assertIn("--limit", self.cli(["--run-batch", "--db-path", str(self.db_path)]))

    def test_limit_must_be_positive(self):  # 2
        for bad in ("0", "-3"):
            self.assertIn("> 0", self.cli(["--run-batch", "--limit", bad, "--db-path", str(self.db_path)]))
        code, _ = self.batch(limit=0)
        self.assertEqual(code, one_shot.EXIT_USAGE)

    def test_mutually_exclusive_with_dry_run_and_run_one(self):
        self.cli(["--run-batch", "--dry-run", "--limit", "1"])
        self.cli(["--run-batch", "--run-one", "u", "--limit", "1"])

    def test_allow_expired_rejected_with_batch(self):
        self.cli(["--run-batch", "--limit", "1", "--allow-expired"])

    def test_cli_passes_args_and_preview_by_default(self):
        with mock.patch("dotenv.load_dotenv"), mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-fake"}),              mock.patch.object(operator_batch, "run_batch", return_value=0) as rb:
            analysis_pipeline.main(["--run-batch", "--limit", "3", "--db-path", str(self.db_path)])
            self.assertEqual(rb.call_args.args[0], 3)
            self.assertFalse(rb.call_args.kwargs["confirm"])
            analysis_pipeline.main(
                ["--run-batch", "--limit", "3", "--confirm-paid-call", "--db-path", str(self.db_path)])
            self.assertTrue(rb.call_args.kwargs["confirm"])

    def test_confirm_without_api_key_makes_no_calls(self):
        with mock.patch("dotenv.load_dotenv"), mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""}), \
             mock.patch.object(operator_batch, "run_batch") as rb, mock.patch("sys.stdout", io.StringIO()):
            code = analysis_pipeline.main(["--run-batch", "--limit", "1", "--confirm-paid-call"])
        self.assertEqual(code, one_shot.EXIT_USAGE)
        rb.assert_not_called()

    def test_run_one_and_dry_run_unchanged(self):  # 19, 20
        with mock.patch("dotenv.load_dotenv"), mock.patch.object(one_shot, "run_one", return_value=0) as ro:
            analysis_pipeline.main(["--run-one", "u", "--db-path", str(self.db_path)])
            self.assertFalse(ro.call_args.kwargs["confirm"])
        self.add_deadline(1, FUTURE)
        with mock.patch("sys.stdout", io.StringIO()) as out:
            self.assertEqual(analysis_pipeline.main(["--dry-run", "--db-path", str(self.db_path)]), 0)
        self.assertIn("DRY RUN", out.getvalue())
        self.cli(["--dry-run", "--confirm-paid-call"])


class PreviewTests(BatchCase):
    def test_preview_zero_client_calls_and_writes(self):  # 3, 4
        for n in range(1, 4):
            self.add_deadline(n, FUTURE)
        before = self.digest()
        with mock.patch.object(openai_triage.OpenAITriageAnalyzer, "ensure_client", side_effect=AssertionError("client")):
            code, text = self.batch(limit=2)
        self.assertEqual(code, 0)
        self.assertEqual(self.digest(), before)
        self.assert_zero_spend()
        self.assertIn("PREVIEW ONLY", text)

    def test_preview_shows_exact_urls_and_totals(self):  # 5
        urls = [self.add_deadline(n, FUTURE) for n in range(1, 4)]
        _, text = self.batch(limit=2)
        self.assertIn(urls[0], text)
        self.assertIn(urls[1], text)
        self.assertNotIn(urls[2], text)
        for label in (
            "selected tenders: 2", "new triage calls potentially required: 2", "known reusable triage: 0",
            "expired excluded", "deadline_unknown", "deadline_conflict", "monthly actual spend",
            "outstanding reservations", "effective committed spend", "conservative triage max cost total",
            "не прогноз", "hard monthly budget", "soft limit", "estimated triage input tokens",
            "estimated triage max cost", "Тендер 1",
        ):
            self.assertIn(label, text)

    def test_preview_equals_live_selection(self):  # 6
        for n in range(1, 9):
            self.add_deadline(n, FUTURE)
        self.add_deadline(20, PAST)
        _, preview = self.batch(limit=3)
        shown = [u for u in (f"https://example.test/resource/{n}" for n in range(1, 21)) if f"{u}\n" in preview + "\n"]
        self.assertEqual(len(shown), 3)
        self.assertEqual(self.calls(), 0)
        code, _ = self.batch(limit=3, confirm=True)
        self.assertEqual(code, 0)
        self.assertEqual(sorted(self.triage.calls), sorted(shown))

    def test_reusable_triage_labelled_and_not_called(self):  # 11
        url = self.add_deadline(1, FUTURE)
        self.make().process_announcement(url)
        self.deep.prompt_version = "v-new"  # deep устарел -> retry_deep, triage reusable
        _, text = self.batch(limit=1)
        self.assertIn("reusable triage", text)
        self.assertIn("known reusable triage: 1", text)
        before = len(self.triage.calls)
        self.batch(limit=1, confirm=True)
        self.assertEqual(len(self.triage.calls), before)


class LiveTests(BatchCase):
    def test_limit_five_among_fifty_with_expired(self):  # 7, 8
        expired = [self.add_deadline(n, PAST) for n in range(1, 6)]
        active = [self.add_deadline(n, FUTURE) for n in range(100, 145)]
        code, text = self.batch(limit=5, confirm=True)
        self.assertEqual(code, 0)
        self.assertEqual(self.triage.calls, active[:5])
        self.assertEqual(len(self.deep.calls), 5)
        for url in expired + active[5:]:
            self.assertIsNone(self.state(url))  # не тронуты (expired в live не обрабатываются)
        self.assertIn("selected tenders: 5", text)
        self.assertIn("expired excluded: 5", text)
        self.assertIn("BATCH OPERATOR REPORT", text)

    def test_unknown_and_conflict_can_be_selected(self):  # 9, 10
        unknown = self.add_deadline(1, None)
        conflict = self.add_deadline(2, PAST, FUTURE)
        _, text = self.batch(limit=5, confirm=True)
        self.assertEqual(sorted(self.triage.calls), sorted([unknown, conflict]))
        self.assertIn("deadline_unknown: 1", text)
        self.assertIn("deadline_conflict: 1", text)

    def test_not_relevant_continues(self):  # 12
        urls = [self.add_deadline(n, FUTURE) for n in range(1, 4)]
        self.triage.outcomes[urls[0]] = triage_result("not_relevant")
        _, text = self.batch(limit=3, confirm=True)
        self.assertEqual(self.triage.calls, urls)
        self.assertIn("not_relevant: 1", text)
        self.assertIn("deep_completed: 2", text)

    def test_gate_skip_continues(self):  # 13
        urls = [self.add_deadline(n, FUTURE) for n in range(1, 4)]

        def gate(triage, ctx, min_value):
            if ctx["resource_url"] == urls[0]:
                return {"gate_decision": "skip_low_value", "gate_reason": "t", "facts_used": {}}
            return gate_pass(triage, ctx, min_value)

        _, text = self.batch(limit=3, confirm=True, gate=gate)
        self.assertEqual(self.triage.calls, urls)
        self.assertEqual(self.deep.calls, urls[1:])
        self.assertIn("commercial gate skips: 1", text)

    def test_validation_failure_continues(self):  # 14
        urls = [self.add_deadline(n, FUTURE) for n in range(1, 4)]
        self.triage.outcomes[urls[0]] = openai_triage.TriageError(
            openai_triage.KIND_VALIDATION, "bad", usage=USAGE, response_id="resp-bad")
        code, _ = self.batch(limit=3, confirm=True)
        self.assertEqual(code, one_shot.EXIT_FAILED)
        self.assertEqual(self.triage.calls, urls)
        self.assertEqual(self.deep.calls, urls[1:])

    def test_accounting_unresolved_stops_further_calls(self):  # 15
        urls = [self.add_deadline(n, FUTURE) for n in range(1, 5)]
        self.triage.outcomes[urls[1]] = openai_triage.TriageError(openai_triage.KIND_TIMEOUT, "ambiguous")
        code, text = self.batch(limit=4, confirm=True)
        self.assertEqual(code, one_shot.EXIT_FAILED)
        self.assertEqual(self.triage.calls, urls[:2])  # третий и четвёртый платный вызов не начат
        self.assertEqual(self.deep.calls, urls[:1])
        self.assertIn("ACCOUNTING FAIL-CLOSED", text)
        self.assertIn("not started: 2", text)

    def test_preexisting_outstanding_reservation_blocks(self):
        for n in range(1, 4):
            self.add_deadline(n, FUTURE)
        ai_usage_repository.reserve_call("triage", "m", Decimal("0.01"), reference="other", db_path=self.db_path)
        code, text = self.batch(limit=3, confirm=True)
        self.assertEqual(self.calls(), 0)
        self.assertEqual(code, one_shot.EXIT_FAILED)
        self.assertIn("not started: 2", text)

    def test_budget_exhausted_makes_no_calls(self):  # 16
        urls = [self.add_deadline(n, FUTURE) for n in range(1, 3)]
        tiny = BudgetSettings(monthly_budget_usd=Decimal("0.000001"), soft_limit_usd=Decimal("0.0000005"))
        _, text = self.batch(limit=2, confirm=True, settings=tiny)
        self.assertEqual(self.calls(), 0)
        self.assertEqual(self.state(urls[0])["state"], state_repo.STATE_TRIAGE_DEFERRED_BUDGET)
        self.assertIn("budget deferred: 2", text)

    def test_no_terra_and_report_usage(self):  # 17
        self.add_deadline(1, FUTURE)
        _, text = self.batch(limit=1, confirm=True)
        self.assertNotIn("terra", text.lower())
        self.assertNotIn("terra", self.deep.model.lower())
        self.assertIn("input_tokens: 2000", text)
        self.assertIn("output_tokens: 1000", text)
        self.assertIn("outstanding reservations after batch: 0", text)

    def test_disabled_flag_does_not_prevent_batch(self):  # 18
        self.add_deadline(1, FUTURE)
        with mock.patch.dict(os.environ, {"AI_ANALYSIS_ENABLED": "false"}):
            self.batch(limit=1, confirm=True)
        self.assertEqual(len(self.triage.calls), 1)

    def test_api_key_not_printed(self):  # 21
        self.add_deadline(1, FUTURE)
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-secret-test-key"}):
            _, preview = self.batch(limit=1)
            _, live = self.batch(limit=1, confirm=True)
        self.assertNotIn("sk-secret-test-key", preview + live)


if __name__ == "__main__":
    unittest.main()
