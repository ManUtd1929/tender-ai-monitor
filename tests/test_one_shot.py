"""
Тесты операторского one-shot (src.ai.one_shot): только fake analyzers, без OpenAI/сети/production БД.

    python -m unittest tests.test_one_shot -v
"""

import io
import os
import unittest
from decimal import Decimal
from unittest import mock

from src.ai import analysis_pipeline, one_shot
from src.ai.budget_settings import BudgetSettings
from src.database import ai_usage_repository, document_repository, pipeline_state_repository as state_repo
from tests.test_analysis_pipeline import PipelineTestCase, triage_result
from tests.test_tender_context import make_docx_extraction, make_download_result


class OneShotTestCase(PipelineTestCase):
    def add_with_document(self, number, **enrichment) -> str:
        """Тендер БЕЗ estimated_value_amd, но с успешно извлечённым документом (реальный Commercial Gate)."""
        url = self.add(number, **enrichment)
        download = document_repository.save_download(
            url, "armeps_document", "1", make_download_result(), db_path=self.db_path,
        )
        document_repository.save_docx_extraction(
            download["download_id"], make_docx_extraction("Поставка товаров, технические требования"), self.db_path,
        )
        return url

    def run_one(self, url, confirm=True, settings=None):
        lines = []
        code = one_shot.run_one(
            url, self.triage, self.deep, settings or self.settings, db_path=self.db_path,
            confirm=confirm, out=lines.append,
        )
        return code, "\n".join(lines)

    def total_calls(self):
        return len(self.triage.calls) + len(self.deep.calls)


class SafetyTests(OneShotTestCase):
    def test_without_confirm_makes_no_calls_and_shows_preflight(self):
        url = self.add(1)
        code, text = self.run_one(url, confirm=False)
        self.assertEqual(code, 0)
        self.assertEqual(self.total_calls(), 0)
        self.openai_client.assert_not_called()
        self.assertIsNone(self.state(url))
        for label in (
            "resource_url", "triage model", "triage reasoning effort", "deep primary model", "deep reasoning effort",
            "monthly actual spend", "outstanding reservations", "effective committed spend",
            "estimated triage input tokens", "estimated triage max cost", "monthly hard limit", "soft limit",
        ):
            self.assertIn(label, text)
        self.assertIn("DRY-RUN", text)

    def test_unknown_url_makes_no_calls(self):
        self.add(1)
        code, text = self.run_one("https://example.test/resource/unknown")
        self.assertEqual(code, one_shot.EXIT_USAGE)
        self.assertIn("отсутствует в БД", text)
        self.assertEqual(self.total_calls(), 0)

    def test_budget_blocked_makes_no_calls(self):
        url = self.add(1)
        code, text = self.run_one(url, settings=BudgetSettings(monthly_budget_usd=Decimal("0.000001"),
                                                               soft_limit_usd=Decimal("0.0000005")))
        self.assertEqual(self.total_calls(), 0)
        self.assertEqual(self.state(url)["state"], state_repo.STATE_TRIAGE_DEFERRED_BUDGET)
        self.assertEqual(code, 0)

    def test_outstanding_reservation_blocks_calls(self):
        url = self.add(1)
        ai_usage_repository.reserve_call("triage", "m", Decimal("0.01"), reference="other", db_path=self.db_path)
        code, text = self.run_one(url)
        self.assertEqual(self.total_calls(), 0)
        self.assertEqual(code, one_shot.EXIT_FAILED)
        self.assertIn("accounting_blocked", text)

    def test_no_api_key_printed(self):
        url = self.add(1)
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-secret-test-key"}):
            _, text = self.run_one(url)
        self.assertNotIn("sk-secret-test-key", text)


class OneTenderOnlyTests(OneShotTestCase):
    def test_only_requested_url_among_many(self):
        urls = [self.add_with_document(n) for n in range(55)]
        target = urls[27]
        with mock.patch.object(analysis_pipeline.AnalysisPipeline, "process_batch",
                               side_effect=AssertionError("no batch")), \
             mock.patch.object(analysis_pipeline.AnalysisPipeline, "select_candidates",
                               side_effect=AssertionError("no backlog selection")):
            code, _ = self.run_one(target)
        self.assertEqual(code, 0)
        self.assertEqual(self.triage.calls, [target])
        self.assertEqual(self.deep.calls, [target])
        self.assertIsNone(self.state(urls[0]))
        self.assertIsNone(self.state(urls[28]))


class FlowTests(OneShotTestCase):
    def test_relevant_goes_triage_gate_deep_with_reports(self):
        url = self.add_with_document(1)
        code, text = self.run_one(url)
        self.assertEqual(code, 0)
        self.assertEqual((self.triage.calls, self.deep.calls), ([url], [url]))
        self.assertIn("DEEP PREFLIGHT: Commercial Gate result", text)
        self.assertIn("Deep Admission result", text)
        self.assertIn("estimated Deep input tokens", text)
        self.assertIn("estimated Deep max cost", text)
        self.assertIn("API calls actually made: 2", text)
        self.assertIn("input_tokens: 2000", text)
        self.assertIn("cached_tokens: 400", text)
        self.assertIn("cache_write_tokens: 200", text)
        self.assertIn("output_tokens: 1000", text)
        self.assertIn("deep_completed", text)
        self.assertIn("outstanding reservations after run: 0", text)

    def test_unknown_value_without_documents_is_insufficient_information_and_no_deep(self):
        url = self.add(1)
        code, text = self.run_one(url)
        self.assertEqual(code, 0)
        self.assertEqual((len(self.triage.calls), self.deep.calls), (1, []))
        self.assertEqual(self.state(url)["state"], state_repo.STATE_COMMERCIAL_GATE_SKIP)
        self.assertEqual(self.state(url)["reason_code"], "insufficient_information")

    def test_not_relevant_only_triage(self):
        url = self.add(1)
        self.triage.default_outcome = triage_result("not_relevant")
        code, text = self.run_one(url)
        self.assertEqual(code, 0)
        self.assertEqual((len(self.triage.calls), self.deep.calls), (1, []))
        self.assertNotIn("DEEP PREFLIGHT", text)
        self.assertIn("API calls actually made: 1", text)
        self.assertIn("stopped_not_relevant", text)

    def test_existing_result_is_idempotent(self):
        url = self.add_with_document(1)
        self.run_one(url)
        code, text = self.run_one(url)
        self.assertEqual(code, 0)
        self.assertEqual((len(self.triage.calls), len(self.deep.calls)), (1, 1))
        self.assertIn("API calls actually made: 0", text)
        self.assertIn("reused", text)

    def test_terra_fallback_not_used(self):
        url = self.add_with_document(1)
        self.run_one(url)
        self.assertEqual({self.deep.model}, {self.settings.deep_primary_model})
        self.assertNotEqual(self.deep.model, self.settings.deep_fallback_model)

    def test_ai_analysis_enabled_false_does_not_block(self):
        url = self.add(1)
        with mock.patch.dict(os.environ, {"AI_ANALYSIS_ENABLED": "false"}):
            code, _ = self.run_one(url)
        self.assertEqual(code, 0)
        self.assertEqual(self.triage.calls, [url])


class CliTests(OneShotTestCase):
    def test_confirm_requires_run_one(self):
        with self.assertRaises(SystemExit), mock.patch("sys.stderr", io.StringIO()):
            analysis_pipeline.main(["--dry-run", "--confirm-paid-call"])

    def test_run_one_rejects_limit(self):
        with self.assertRaises(SystemExit), mock.patch("sys.stderr", io.StringIO()):
            analysis_pipeline.main(["--run-one", "u", "--limit", "5", "--db-path", str(self.db_path)])

    def test_regular_dry_run_unchanged(self):
        self.add(1)
        with mock.patch("sys.stdout", io.StringIO()) as stdout:
            code = analysis_pipeline.main(["--dry-run", "--limit", "5", "--db-path", str(self.db_path)])
        self.assertEqual(code, 0)
        self.assertIn("DRY RUN", stdout.getvalue())
        self.assertIsNone(self.state(self.add(1)))
        self.openai_client.assert_not_called()


class DbPathResolutionTests(OneShotTestCase):
    """Регрессия: --run-one без --db-path падал на sqlite3.connect(None)."""

    def run_main(self, argv, env):
        """main() с подменённым окружением; .env не читается, run_one перехвачен."""
        with mock.patch.dict(os.environ, env, clear=False), \
             mock.patch("dotenv.load_dotenv"), \
             mock.patch.object(one_shot, "run_one", return_value=0) as run_one:
            if "DATABASE_PATH" not in env:
                os.environ.pop("DATABASE_PATH", None)
            analysis_pipeline.main(argv)
        return run_one

    def resolved_by_run_one(self, db_path_arg, env, confirm=False):
        """Реальный run_one; возвращает db_path, полученный каждым helper-ом."""
        seen = []
        real_max, real_rows = one_shot._max_event_id, one_shot._ledger_rows

        def spy_max(path):
            seen.append(path)
            return real_max(path)

        def spy_rows(path, url, after):
            seen.append(path)
            return real_rows(path, url, after)

        url = self.add(1)
        with mock.patch.dict(os.environ, env), \
             mock.patch.object(one_shot, "_max_event_id", spy_max), \
             mock.patch.object(one_shot, "_ledger_rows", spy_rows):
            one_shot.run_one(url, self.triage, self.deep, self.settings, db_path=db_path_arg,
                             confirm=confirm, out=lambda _: None)
        return seen

    def test_env_path_used_without_cli_arg(self):
        from src.database import tender_repository as repo
        run_one = self.run_main(["--run-one", "u"], {"DATABASE_PATH": "data/custom.db"})
        self.assertEqual(run_one.call_args.kwargs["db_path"], repo.PROJECT_ROOT / "data" / "custom.db")

    def test_main_default_and_explicit(self):
        from src.database import tender_repository as repo
        run_one = self.run_main(["--run-one", "u"], {})
        self.assertEqual(run_one.call_args.kwargs["db_path"], repo.DEFAULT_DB_PATH)
        run_one = self.run_main(["--run-one", "u", "--db-path", "cli.db"], {"DATABASE_PATH": "env.db"})
        self.assertEqual(str(run_one.call_args.kwargs["db_path"]), "cli.db")

    def test_resolver_env_relative_and_absolute(self):
        from pathlib import Path
        from src.database import tender_repository as repo
        self.assertEqual(repo.resolve_db_path(None, {"DATABASE_PATH": "data/x.db"}), repo.PROJECT_ROOT / "data" / "x.db")
        absolute = Path(self.db_path).resolve()
        self.assertEqual(repo.resolve_db_path(None, {"DATABASE_PATH": str(absolute)}), absolute)

    def test_project_default_without_env(self):
        from src.database import tender_repository as repo
        self.assertEqual(repo.resolve_db_path(None, {}), repo.DEFAULT_DB_PATH)
        self.assertEqual(repo.resolve_db_path(None, {"DATABASE_PATH": "  "}), repo.DEFAULT_DB_PATH)

    def test_explicit_cli_path_beats_env(self):
        from src.database import tender_repository as repo
        self.assertEqual(repo.resolve_db_path("cli.db", {"DATABASE_PATH": "env.db"}).name, "cli.db")
        seen = self.resolved_by_run_one(self.db_path, {"DATABASE_PATH": "should/not/be/used.db"}, confirm=True)
        self.assertTrue(seen)
        self.assertTrue(all(str(p) == str(self.db_path) for p in seen))

    def test_env_path_reaches_pipeline_when_no_cli_arg(self):
        seen = self.resolved_by_run_one(None, {"DATABASE_PATH": str(self.db_path)}, confirm=True)
        self.assertTrue(seen)
        self.assertTrue(all(p is not None and str(p) == str(self.db_path) for p in seen))

    def test_default_path_reaches_pipeline_when_no_cli_arg_and_no_env(self):
        from src.database import tender_repository as repo
        env = dict(os.environ)
        env.pop("DATABASE_PATH", None)
        with mock.patch.object(repo, "DEFAULT_DB_PATH", self.db_path), \
             mock.patch.dict(os.environ, env, clear=True):
            seen = self.resolved_by_run_one(None, {}, confirm=True)
        self.assertTrue(seen)
        self.assertTrue(all(p is not None and str(p) == str(self.db_path) for p in seen))

    def test_no_confirm_without_db_path_no_client_no_writes(self):
        url = self.add(1)
        with mock.patch.dict(os.environ, {"DATABASE_PATH": str(self.db_path)}):
            before = self.state(url)
            code = one_shot.run_one(url, self.triage, self.deep, self.settings, db_path=None,
                                    confirm=False, out=lambda _: None)
        self.assertEqual(code, 0)
        self.assertEqual(self.total_calls(), 0)
        self.openai_client.assert_not_called()
        self.assertEqual(self.state(url), before)


if __name__ == "__main__":
    unittest.main()
