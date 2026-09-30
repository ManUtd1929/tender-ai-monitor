"""
Тесты monitor.run_monitor без сети и без базы: все внешние вызовы замоканы.

Запуск из корня проекта:
    python -m unittest tests.test_monitor -v
"""

import copy
import logging
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import enrichment_pipeline, monitor
from src.database import announcement_repository, document_repository, enrichment_repository


def make_announcement(n: int = 1, **overrides) -> dict:
    announcement = {
        "title": f"Тестовое объявление {n}",
        "source_section": "open_competition",
        "source_section_name": "Открытый конкурс",
        "source_page_url": "https://gnumner.minfin.am/ru/page/test/",
        "resource_url": f"https://example.test/resource/{n}",
        "resource_type": "direct_file",
        "published_at": "2026-09-25 18:59:34",
        "deadline_at": "2026-10-02 11:10:00",
        "tender_time_raw": "(опубликован 2026-09-25 18:59:34-от До 2026-10-02 11:10:00 час включен)",
    }
    announcement.update(overrides)
    return announcement


def make_save_result(**overrides) -> dict:
    result = {
        "new_count": 0,
        "updated_count": 0,
        "existing_count": 0,
        "failed_count": 0,
        "new_announcements": [],
        "updated_announcements": [],
    }
    result.update(overrides)
    return result


def make_document_result(**overrides) -> dict:
    result = {
        "processed_count": 0,
        "success_count": 0,
        "no_supported_document_count": 0,
        "failed_count": 0,
        "download_new_count": 0,
        "download_existing_count": 0,
        "extraction_new_count": 0,
        "extraction_updated_count": 0,
        "extraction_existing_count": 0,
        "results": [],
        "failures": [],
    }
    result.update(overrides)
    return result


def make_combined(n: int = 1, **overrides) -> dict:
    """Объединённый announcement + enrichment, как возвращает get_announcement_with_enrichment."""
    combined = make_announcement(n)
    combined.update({"enrichment_status": "success", "cpv_codes": [], "documents": []})
    combined.update(overrides)
    return combined


def make_enrichment_result(**overrides) -> dict:
    result = {
        "processed_count": 0,
        "success_count": 0,
        "failed_count": 0,
        "storage_new_count": 0,
        "storage_updated_count": 0,
        "storage_existing_count": 0,
        "results": [],
        "failures": [],
    }
    result.update(overrides)
    return result


class MergeEnrichmentCandidatesTest(unittest.TestCase):
    def test_current_goes_before_pending(self):
        current = [make_announcement(1), make_announcement(2)]
        pending = [make_announcement(3), make_announcement(4)]

        merged = monitor.merge_enrichment_candidates(current, pending)

        self.assertEqual(merged, current + pending)

    def test_same_url_in_current_and_pending_is_processed_once(self):
        current = [make_announcement(1, title="из текущего запуска")]
        pending = [make_announcement(1, title="из pending"), make_announcement(2)]

        merged = monitor.merge_enrichment_candidates(current, pending)

        self.assertEqual([a["resource_url"] for a in merged], [
            "https://example.test/resource/1",
            "https://example.test/resource/2",
        ])
        self.assertEqual(merged[0]["title"], "из текущего запуска")

    def test_duplicate_urls_inside_current_give_one_record(self):
        current = [make_announcement(1, title="первая"), make_announcement(1, title="вторая")]

        merged = monitor.merge_enrichment_candidates(current, [])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["title"], "первая")

    def test_source_lists_are_not_modified(self):
        current = [make_announcement(1), make_announcement(1)]
        pending = [make_announcement(1), make_announcement(2)]
        current_before = copy.deepcopy(current)
        pending_before = copy.deepcopy(pending)

        merged = monitor.merge_enrichment_candidates(current, pending)

        self.assertEqual(current, current_before)
        self.assertEqual(pending, pending_before)
        self.assertIsNot(merged, current)
        self.assertIsNot(merged, pending)

    def test_empty_inputs(self):
        self.assertEqual(monitor.merge_enrichment_candidates([], []), [])


class MergeDocumentCandidatesTest(unittest.TestCase):
    def test_current_goes_before_pending(self):
        current = [make_combined(1), make_combined(2)]
        pending = [make_combined(3)]

        self.assertEqual(monitor.merge_document_candidates(current, pending), current + pending)

    def test_dedupes_by_resource_url_first_wins(self):
        current = [make_combined(1, title="из текущего запуска")]
        pending = [make_combined(1, title="из pending"), make_combined(2)]

        merged = monitor.merge_document_candidates(current, pending)

        self.assertEqual([a["resource_url"] for a in merged], [
            "https://example.test/resource/1",
            "https://example.test/resource/2",
        ])
        self.assertEqual(merged[0]["title"], "из текущего запуска")

    def test_source_lists_are_not_modified(self):
        current = [make_combined(1), make_combined(1)]
        pending = [make_combined(1), make_combined(2)]
        current_before = copy.deepcopy(current)
        pending_before = copy.deepcopy(pending)

        merged = monitor.merge_document_candidates(current, pending)

        self.assertEqual(current, current_before)
        self.assertEqual(pending, pending_before)
        self.assertIsNot(merged, current)
        self.assertIsNot(merged, pending)

    def test_empty_inputs(self):
        self.assertEqual(monitor.merge_document_candidates([], []), [])


class RunMonitorTest(unittest.TestCase):
    def setUp(self):
        patches = {
            "configure_tls": mock.patch.object(monitor, "configure_tls"),
            "init_db": mock.patch.object(monitor, "init_db"),
            "fetch_all_sections": mock.patch.object(monitor, "fetch_all_sections"),
            "save_announcements": mock.patch.object(monitor, "save_announcements"),
            "count_announcements": mock.patch.object(monitor, "count_announcements"),
            "enrichment_init_db": mock.patch.object(monitor.enrichment_repository, "init_db"),
            "get_pending": mock.patch.object(
                monitor.enrichment_repository, "get_enrichment_processing_candidates"
            ),
            "count_pending": mock.patch.object(
                monitor.enrichment_repository, "count_enrichment_processing_candidates"
            ),
            "process_announcements": mock.patch.object(monitor, "process_announcements"),
            "document_init_db": mock.patch.object(monitor.document_repository, "init_db"),
            "count_document_pending": mock.patch.object(
                monitor.document_repository, "count_document_processing_candidates"
            ),
            "get_document_pending": mock.patch.object(
                monitor.document_repository, "get_document_processing_candidates"
            ),
            "get_combined": mock.patch.object(
                monitor.enrichment_repository, "get_announcement_with_enrichment"
            ),
            "process_documents": mock.patch.object(monitor, "process_enriched_announcements"),
        }
        for name, patcher in patches.items():
            setattr(self, name, patcher.start())
            self.addCleanup(patcher.stop)

        self.fetch_all_sections.return_value = []
        self.save_announcements.return_value = make_save_result()
        self.count_announcements.return_value = 0
        self.get_pending.return_value = []
        self.count_pending.return_value = 0
        self.process_announcements.return_value = make_enrichment_result()
        self.count_document_pending.return_value = 0
        self.get_document_pending.return_value = []
        self.get_combined.side_effect = lambda url: make_combined(resource_url=url)
        self.process_documents.return_value = make_document_result()

    def enrichment_success(self, *numbers) -> dict:
        return make_enrichment_result(
            success_count=len(numbers),
            results=[
                {"resource_url": f"https://example.test/resource/{n}", "enrichment_status": "success"}
                for n in numbers
            ],
        )

    def document_urls(self) -> list[str]:
        candidates = self.process_documents.call_args.args[0]
        return [a["resource_url"] for a in candidates]

    def processed_urls(self) -> list[str]:
        candidates = self.process_announcements.call_args.args[0]
        return [a["resource_url"] for a in candidates]

    def run_with_ai_env(self, value):
        env = {} if value is None else {"AI_ANALYSIS_ENABLED": value}
        return mock.patch.dict("os.environ", env, clear=True)

    def test_ai_analysis_disabled_by_default_makes_no_ai_calls(self):
        from src.ai import analysis_pipeline

        for value in (None, "", "false", "0"):
            with self.subTest(value=value), self.run_with_ai_env(value),                  mock.patch.object(analysis_pipeline, "run_ai_analysis") as run_ai,                  mock.patch.object(analysis_pipeline, "build_analyzers") as build,                  mock.patch("openai.OpenAI") as client:
                result = monitor.run_monitor()

            self.assertIsNone(result["ai_analysis"])
            run_ai.assert_not_called()
            build.assert_not_called()
            client.assert_not_called()

    def test_ai_analysis_enabled_runs_pipeline_after_documents(self):
        from src.ai import analysis_pipeline

        order = []
        self.process_documents.side_effect = lambda candidates: order.append("documents") or make_document_result()
        with self.run_with_ai_env("true"), mock.patch.object(
            analysis_pipeline, "run_ai_analysis", side_effect=lambda: order.append("ai") or {"processed_count": 0},
        ):
            result = monitor.run_monitor()

        self.assertEqual(order, ["documents", "ai"])
        self.assertEqual(result["ai_analysis"], {"processed_count": 0})

    def test_ai_analysis_error_does_not_abort_monitor(self):
        from src.ai import analysis_pipeline

        with self.run_with_ai_env("true"), mock.patch.object(
            analysis_pipeline, "run_ai_analysis", side_effect=RuntimeError("boom"),
        ):
            result = monitor.run_monitor()

        self.assertEqual(result["ai_analysis"]["status"], "error")
        self.assertEqual(result["ai_analysis"]["error_type"], "RuntimeError")
        self.assertEqual(result["fetched_count"], 0)

    def test_counts_fetched_unique_and_duplicates(self):
        self.fetch_all_sections.return_value = [make_announcement(1), make_announcement(2), make_announcement(3)]

        result = monitor.run_monitor()

        self.assertEqual(result["fetched_count"], 3)
        self.assertEqual(result["unique_resource_count"], 3)
        self.assertEqual(result["duplicate_count"], 0)

    def test_same_resource_url_twice_is_one_unique_and_one_duplicate(self):
        self.fetch_all_sections.return_value = [
            make_announcement(1),
            make_announcement(1, source_section="other_section"),
        ]

        result = monitor.run_monitor()

        self.assertEqual(result["fetched_count"], 2)
        self.assertEqual(result["unique_resource_count"], 1)
        self.assertEqual(result["duplicate_count"], 1)

    def test_passes_all_announcements_to_repository_including_duplicates(self):
        announcements = [make_announcement(1), make_announcement(1), make_announcement(2)]
        self.fetch_all_sections.return_value = announcements

        monitor.run_monitor()

        self.save_announcements.assert_called_once_with(announcements)

    def test_empty_resource_url_is_not_counted_as_unique(self):
        self.fetch_all_sections.return_value = [
            make_announcement(1),
            make_announcement(2, resource_url=""),
        ]

        result = monitor.run_monitor()

        self.assertEqual(result["fetched_count"], 2)
        self.assertEqual(result["unique_resource_count"], 1)
        self.assertEqual(result["duplicate_count"], 1)

    def test_repository_counters_are_carried_over(self):
        self.save_announcements.return_value = make_save_result(
            new_count=5, updated_count=3, existing_count=30, failed_count=2,
        )

        result = monitor.run_monitor()

        self.assertEqual(result["new_count"], 5)
        self.assertEqual(result["updated_count"], 3)
        self.assertEqual(result["existing_count"], 30)
        self.assertEqual(result["failed_count"], 2)

    def test_total_count_comes_from_count_announcements(self):
        self.count_announcements.return_value = 123

        result = monitor.run_monitor()

        self.assertEqual(result["total_count"], 123)
        self.count_announcements.assert_called_once_with()

    def test_new_and_updated_announcements_returned_without_loss(self):
        new = [make_announcement(1), make_announcement(2)]
        updated = [make_announcement(3, deadline_at="2026-11-01 10:00:00")]
        self.save_announcements.return_value = make_save_result(
            new_count=2, updated_count=1, new_announcements=new, updated_announcements=updated,
        )

        result = monitor.run_monitor()

        self.assertEqual(result["new_announcements"], new)
        self.assertEqual(result["updated_announcements"], updated)

    def test_page_is_passed_to_fetch_all_sections(self):
        monitor.run_monitor(page=2)

        self.fetch_all_sections.assert_called_once_with(page=2)

    def test_init_db_is_called_before_fetch(self):
        calls = mock.Mock()
        calls.attach_mock(self.init_db, "init_db")
        calls.attach_mock(self.fetch_all_sections, "fetch_all_sections")

        monitor.run_monitor()

        self.assertEqual(
            [c[0] for c in calls.mock_calls], ["init_db", "fetch_all_sections"],
        )

    def test_fetch_error_is_not_swallowed(self):
        self.fetch_all_sections.side_effect = RuntimeError("сеть недоступна")

        with self.assertRaises(RuntimeError):
            monitor.run_monitor()

        self.save_announcements.assert_not_called()

    def test_database_error_is_not_swallowed(self):
        self.fetch_all_sections.return_value = [make_announcement(1)]
        self.save_announcements.side_effect = RuntimeError("база недоступна")

        with self.assertRaises(RuntimeError):
            monitor.run_monitor()

    def test_enrichment_fatal_error_is_not_swallowed(self):
        self.process_announcements.side_effect = RuntimeError("enrichment упал")

        with self.assertRaises(RuntimeError):
            monitor.run_monitor()

    # --- enrichment ---

    def test_new_announcement_is_sent_to_enrichment(self):
        new = make_announcement(1)
        self.save_announcements.return_value = make_save_result(new_count=1, new_announcements=[new])

        monitor.run_monitor()

        self.assertEqual(self.process_announcements.call_args.args[0], [new])

    def test_updated_announcement_is_sent_to_enrichment(self):
        updated = make_announcement(2, deadline_at="2026-11-01 10:00:00")
        self.save_announcements.return_value = make_save_result(
            updated_count=1, updated_announcements=[updated],
        )

        monitor.run_monitor()

        self.assertEqual(self.process_announcements.call_args.args[0], [updated])

    def test_existing_announcement_is_not_sent_by_itself(self):
        self.fetch_all_sections.return_value = [make_announcement(1)]
        self.save_announcements.return_value = make_save_result(existing_count=1)
        # У существующего объявления enrichment уже есть — в pending его нет.
        self.get_pending.return_value = []

        monitor.run_monitor()

        self.assertEqual(self.process_announcements.call_args.args[0], [])

    def test_pending_retry_is_requested_with_limit_3(self):
        monitor.run_monitor()

        self.get_pending.assert_called_once_with(limit=3)
        self.assertEqual(monitor.PENDING_ENRICHMENT_RETRY_LIMIT, 3)

    def test_without_new_or_updated_process_gets_pending_retry(self):
        pending = [make_announcement(7), make_announcement(8), make_announcement(9)]
        self.get_pending.return_value = pending

        monitor.run_monitor()

        self.assertEqual(self.process_announcements.call_args.args[0], pending)

    def test_new_present_in_pending_retry_is_processed_once(self):
        new = make_announcement(1)
        self.save_announcements.return_value = make_save_result(new_count=1, new_announcements=[new])
        self.get_pending.return_value = [make_announcement(1), make_announcement(5)]

        result = monitor.run_monitor()

        self.assertEqual(self.processed_urls(), [
            "https://example.test/resource/1",
            "https://example.test/resource/5",
        ])
        self.assertEqual(result["enrichment_candidate_count"], 2)

    def test_updated_with_existing_enrichment_is_processed_even_if_not_pending(self):
        updated = make_announcement(4)
        self.save_announcements.return_value = make_save_result(
            updated_count=1, updated_announcements=[updated],
        )
        self.get_pending.return_value = [make_announcement(8)]

        monitor.run_monitor()

        self.assertEqual(self.processed_urls(), [
            "https://example.test/resource/4",
            "https://example.test/resource/8",
        ])

    def test_current_announcements_go_before_pending_retry(self):
        new = make_announcement(1)
        updated = make_announcement(2)
        self.save_announcements.return_value = make_save_result(
            new_count=1, updated_count=1, new_announcements=[new], updated_announcements=[updated],
        )
        self.get_pending.return_value = [make_announcement(3)]

        monitor.run_monitor()

        self.assertEqual(self.processed_urls(), [
            "https://example.test/resource/1",
            "https://example.test/resource/2",
            "https://example.test/resource/3",
        ])

    def test_pending_before_is_counted_before_processing(self):
        calls = mock.Mock()
        calls.attach_mock(self.enrichment_init_db, "enrichment_init_db")
        calls.attach_mock(self.count_pending, "count_pending")
        calls.attach_mock(self.process_announcements, "process_announcements")
        self.count_pending.side_effect = [36, 33]

        result = monitor.run_monitor()

        names = [c[0] for c in calls.mock_calls]
        self.assertEqual(names, [
            "enrichment_init_db", "count_pending", "process_announcements", "count_pending",
        ])
        self.assertEqual(result["pending_enrichment_before"], 36)

    def test_pending_after_is_counted_after_processing(self):
        self.count_pending.side_effect = [36, 33]

        result = monitor.run_monitor()

        self.assertEqual(result["pending_enrichment_after"], 33)
        self.assertEqual(self.count_pending.call_count, 2)

    def test_enrichment_result_is_returned_without_loss(self):
        enrichment_result = make_enrichment_result(
            processed_count=3, success_count=2, failed_count=1, storage_new_count=2,
            results=[{"resource_url": "a"}],
            failures=[{"resource_url": "b", "error_type": "E", "error_message": "m"}],
        )
        self.process_announcements.return_value = enrichment_result

        result = monitor.run_monitor()

        self.assertIs(result["enrichment_result"], enrichment_result)

    def test_enrichment_candidate_count(self):
        self.save_announcements.return_value = make_save_result(
            new_count=2, updated_count=1,
            new_announcements=[make_announcement(1), make_announcement(2)],
            updated_announcements=[make_announcement(3)],
        )
        self.get_pending.return_value = [make_announcement(2), make_announcement(4)]

        result = monitor.run_monitor()

        self.assertEqual(result["enrichment_candidate_count"], 4)

    def test_pending_retry_selected_count(self):
        self.get_pending.return_value = [make_announcement(7), make_announcement(8)]

        result = monitor.run_monitor()

        self.assertEqual(result["pending_retry_selected_count"], 2)

    def test_enrichment_runs_after_save(self):
        calls = mock.Mock()
        calls.attach_mock(self.save_announcements, "save_announcements")
        calls.attach_mock(self.process_announcements, "process_announcements")

        monitor.run_monitor()

        self.assertEqual(
            [c[0] for c in calls.mock_calls], ["save_announcements", "process_announcements"],
        )

    def test_legacy_return_fields_are_kept(self):
        result = monitor.run_monitor()

        for key in (
            "fetched_count", "unique_resource_count", "duplicate_count", "new_count",
            "updated_count", "existing_count", "failed_count", "total_count",
            "new_announcements", "updated_announcements",
        ):
            self.assertIn(key, result)

    # --- документы ---

    def test_retry_limit_constant_is_3(self):
        self.assertEqual(monitor.PENDING_DOCUMENT_RETRY_LIMIT, 3)

    def test_pending_documents_requested_with_limit_3(self):
        monitor.run_monitor()

        self.get_document_pending.assert_called_once_with(limit=3)

    def test_monitor_takes_at_most_limit_old_candidates(self):
        # Репозиторий уже отдаёт не больше limit; monitor не добавляет сверху.
        self.get_document_pending.side_effect = lambda limit: [
            make_combined(n) for n in range(10, 20)
        ][:limit]

        result = monitor.run_monitor()

        self.assertEqual(
            self.document_urls(), [f"https://example.test/resource/{n}" for n in (10, 11, 12)]
        )
        self.assertEqual(result["pending_document_retry_selected_count"], 3)

    def test_documents_of_successfully_enriched_announcements_are_processed(self):
        self.process_announcements.return_value = self.enrichment_success(1, 2)

        monitor.run_monitor()

        self.assertEqual(self.document_urls(), [
            "https://example.test/resource/1", "https://example.test/resource/2",
        ])

    def test_partial_not_required_and_unsupported_enrichment_are_processed(self):
        self.process_announcements.return_value = make_enrichment_result(results=[
            {"resource_url": "https://example.test/resource/1", "enrichment_status": "partial"},
            {"resource_url": "https://example.test/resource/2", "enrichment_status": "not_required"},
            {"resource_url": "https://example.test/resource/3", "enrichment_status": "unsupported"},
        ])

        monitor.run_monitor()

        self.assertEqual(len(self.document_urls()), 3)

    def test_enrichment_failure_does_not_go_to_document_pipeline(self):
        self.process_announcements.return_value = make_enrichment_result(
            failed_count=1,
            results=[{"resource_url": "https://example.test/resource/1", "enrichment_status": "success"}],
            failures=[{
                "resource_url": "https://example.test/resource/2",
                "error_type": "E", "error_message": "m",
            }],
        )

        monitor.run_monitor()

        self.assertEqual(self.document_urls(), ["https://example.test/resource/1"])
        self.get_combined.assert_called_once_with("https://example.test/resource/1")

    def test_updated_with_failed_enrichment_is_not_processed(self):
        updated = make_announcement(2)
        self.save_announcements.return_value = make_save_result(
            updated_count=1, updated_announcements=[updated],
        )
        self.process_announcements.return_value = make_enrichment_result(
            failed_count=1,
            failures=[{"resource_url": updated["resource_url"], "error_type": "E", "error_message": "m"}],
        )

        monitor.run_monitor()

        self.assertEqual(self.document_urls(), [])

    def test_updated_announcement_is_processed_even_if_state_is_success(self):
        # Состояние success исключило бы его из pending, но updated идёт как current.
        updated = make_announcement(4)
        self.save_announcements.return_value = make_save_result(
            updated_count=1, updated_announcements=[updated],
        )
        self.process_announcements.return_value = make_enrichment_result(results=[])
        self.get_document_pending.return_value = []

        monitor.run_monitor()

        self.assertEqual(self.document_urls(), ["https://example.test/resource/4"])

    def test_current_and_pending_documents_are_deduplicated(self):
        self.process_announcements.return_value = self.enrichment_success(1, 2)
        self.get_document_pending.return_value = [make_combined(2), make_combined(9)]

        result = monitor.run_monitor()

        self.assertEqual(self.document_urls(), [
            "https://example.test/resource/1",
            "https://example.test/resource/2",
            "https://example.test/resource/9",
        ])
        self.assertEqual(result["document_candidate_count"], 3)

    def test_same_url_in_enrichment_and_updated_is_processed_once(self):
        updated = make_announcement(1)
        self.save_announcements.return_value = make_save_result(
            updated_count=1, updated_announcements=[updated],
        )
        self.process_announcements.return_value = self.enrichment_success(1)

        monitor.run_monitor()

        self.assertEqual(self.document_urls(), ["https://example.test/resource/1"])
        self.get_combined.assert_called_once_with("https://example.test/resource/1")

    def test_candidates_are_full_objects_from_join_helper(self):
        self.process_announcements.return_value = self.enrichment_success(1)
        combined = make_combined(1, resource_type="armeps_documents_page", cpv_codes=[{"code": "1"}])
        self.get_combined.side_effect = None
        self.get_combined.return_value = combined

        monitor.run_monitor()

        self.get_combined.assert_called_once_with("https://example.test/resource/1")
        self.assertIs(self.process_documents.call_args.args[0][0], combined)

    def test_candidate_without_enrichment_is_skipped(self):
        self.process_announcements.return_value = self.enrichment_success(1, 2)
        self.get_combined.side_effect = lambda url: (
            None if url.endswith("/1") else make_combined(2)
        )

        result = monitor.run_monitor()

        self.assertEqual(self.document_urls(), ["https://example.test/resource/2"])
        self.assertEqual(result["document_candidate_count"], 1)

    def test_documents_run_after_enrichment_and_after_document_init(self):
        calls = mock.Mock()
        calls.attach_mock(self.process_announcements, "process_announcements")
        calls.attach_mock(self.document_init_db, "document_init_db")
        calls.attach_mock(self.count_document_pending, "count_document_pending")
        calls.attach_mock(self.process_documents, "process_documents")
        self.count_document_pending.side_effect = [5, 4]

        result = monitor.run_monitor()

        self.assertEqual([c[0] for c in calls.mock_calls], [
            "process_announcements", "document_init_db", "count_document_pending",
            "process_documents", "count_document_pending",
        ])
        self.assertEqual(result["pending_documents_before"], 5)
        self.assertEqual(result["pending_documents_after"], 4)

    def test_document_result_is_returned_without_loss(self):
        document_result = make_document_result(
            processed_count=2, success_count=1, failed_count=1,
            failures=[{"resource_url": "u", "error_type": "E", "error_message": "m"}],
        )
        self.process_documents.return_value = document_result

        result = monitor.run_monitor()

        self.assertIs(result["document_result"], document_result)

    def test_document_failure_of_one_item_does_not_break_monitor(self):
        # process_enriched_announcements сам собирает ошибки отдельных объявлений в failures.
        self.process_announcements.return_value = self.enrichment_success(1, 2)
        self.process_documents.return_value = make_document_result(
            processed_count=2, success_count=1, failed_count=1,
            failures=[{"resource_url": "https://example.test/resource/1",
                       "error_type": "SSLError", "error_message": "handshake"}],
        )

        result = monitor.run_monitor()

        self.assertEqual(result["document_result"]["failed_count"], 1)
        self.assertEqual(result["document_candidate_count"], 2)
        self.assertIn("pending_documents_after", result)

    def test_document_fatal_error_is_not_swallowed(self):
        self.process_documents.side_effect = RuntimeError("init_db упал")

        with self.assertRaises(RuntimeError):
            monitor.run_monitor()

    def test_document_counters_in_result(self):
        self.get_document_pending.return_value = [make_combined(7), make_combined(8)]
        self.count_document_pending.side_effect = [12, 10]

        result = monitor.run_monitor()

        self.assertEqual(result["document_candidate_count"], 2)
        self.assertEqual(result["pending_documents_before"], 12)
        self.assertEqual(result["pending_document_retry_selected_count"], 2)
        self.assertEqual(result["pending_documents_after"], 10)

    def test_enrichment_and_legacy_counters_survive_document_stage(self):
        self.save_announcements.return_value = make_save_result(
            new_count=1, updated_count=1, existing_count=3, failed_count=0,
            new_announcements=[make_announcement(1)], updated_announcements=[make_announcement(2)],
        )
        self.count_pending.side_effect = [4, 2]
        self.get_pending.return_value = [make_announcement(5)]
        self.process_announcements.return_value = self.enrichment_success(1, 2, 5)

        result = monitor.run_monitor()

        self.assertEqual(result["new_count"], 1)
        self.assertEqual(result["updated_count"], 1)
        self.assertEqual(result["existing_count"], 3)
        self.assertEqual(result["enrichment_candidate_count"], 3)
        self.assertEqual(result["pending_enrichment_before"], 4)
        self.assertEqual(result["pending_retry_selected_count"], 1)
        self.assertEqual(result["pending_enrichment_after"], 2)
        self.assertEqual(result["enrichment_result"]["success_count"], 3)
        self.assertEqual(self.document_urls(), [
            "https://example.test/resource/1",
            "https://example.test/resource/2",
            "https://example.test/resource/5",
        ])


    # --- retry enrichment (state failed) и документы ---

    def test_monitor_uses_processing_candidates_not_legacy_pending_query(self):
        with mock.patch.object(
            monitor.enrichment_repository, "get_announcements_without_enrichment"
        ) as legacy_get, mock.patch.object(
            monitor.enrichment_repository, "count_announcements_without_enrichment"
        ) as legacy_count:
            monitor.run_monitor()

        legacy_get.assert_not_called()
        legacy_count.assert_not_called()
        self.get_pending.assert_called_once_with(limit=3)
        self.assertEqual(self.count_pending.call_count, 2)

    def test_old_retry_candidate_with_successful_refresh_goes_to_documents(self):
        # Старое объявление с failed refresh: retry прошёл — теперь можно скачивать документы.
        self.get_pending.return_value = [make_announcement(7)]
        self.process_announcements.return_value = self.enrichment_success(7)

        monitor.run_monitor()

        self.assertEqual(self.processed_urls(), ["https://example.test/resource/7"])
        self.assertEqual(self.document_urls(), ["https://example.test/resource/7"])
        self.get_combined.assert_called_once_with("https://example.test/resource/7")

    def test_old_retry_candidate_with_failed_refresh_does_not_go_to_documents(self):
        self.get_pending.return_value = [make_announcement(7), make_announcement(8)]
        self.process_announcements.return_value = make_enrichment_result(
            success_count=1, failed_count=1,
            results=[{"resource_url": "https://example.test/resource/8", "enrichment_status": "success"}],
            failures=[{
                "resource_url": "https://example.test/resource/7",
                "error_type": "RuntimeError", "error_message": "m",
            }],
        )

        monitor.run_monitor()

        self.assertEqual(self.document_urls(), ["https://example.test/resource/8"])

    def test_updated_with_failed_refresh_stays_out_of_documents_even_if_also_retry_candidate(self):
        updated = make_announcement(4)
        self.save_announcements.return_value = make_save_result(
            updated_count=1, updated_announcements=[updated],
        )
        self.get_pending.return_value = [make_announcement(4)]  # failed state из прошлого запуска
        self.process_announcements.return_value = make_enrichment_result(
            failed_count=1,
            failures=[{"resource_url": updated["resource_url"], "error_type": "E", "error_message": "m"}],
        )

        monitor.run_monitor()

        self.assertEqual(self.processed_urls(), [updated["resource_url"]])
        self.assertEqual(self.document_urls(), [])

    def test_document_pending_retry_keeps_limit_and_priority_source(self):
        # Приоритет never-attempted над failed определяет репозиторий; monitor берёт первые limit.
        self.get_document_pending.return_value = [make_combined(21), make_combined(22)]

        result = monitor.run_monitor()

        self.get_document_pending.assert_called_once_with(limit=3)
        self.assertEqual(self.document_urls(), [
            "https://example.test/resource/21", "https://example.test/resource/22",
        ])
        self.assertEqual(result["pending_document_retry_selected_count"], 2)

    def test_retry_counters_and_legacy_fields_survive(self):
        self.count_pending.side_effect = [5, 3]
        self.get_pending.return_value = [make_announcement(1), make_announcement(2)]
        self.process_announcements.return_value = self.enrichment_success(1)

        result = monitor.run_monitor()

        self.assertEqual(result["pending_enrichment_before"], 5)
        self.assertEqual(result["pending_retry_selected_count"], 2)
        self.assertEqual(result["pending_enrichment_after"], 3)
        self.assertEqual(result["enrichment_candidate_count"], 2)
        for key in (
            "fetched_count", "unique_resource_count", "duplicate_count", "new_count",
            "updated_count", "existing_count", "failed_count", "total_count",
            "new_announcements", "updated_announcements", "enrichment_result",
            "document_candidate_count", "pending_documents_before",
            "pending_document_retry_selected_count", "document_result", "pending_documents_after",
        ):
            self.assertIn(key, result)


class RunMonitorRealDatabaseTest(unittest.TestCase):
    """
    run_monitor на временной SQLite БД (DEFAULT_DB_PATH трёх репозиториев подменён).
    Замоканы только сеть (fetch_all_sections, enrich_announcement) и document pipeline.
    """

    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "test.db"

        for module in (announcement_repository, enrichment_repository, document_repository):
            patcher = mock.patch.object(module, "DEFAULT_DB_PATH", self.db_path)
            patcher.start()
            self.addCleanup(patcher.stop)

        self.enrich_ok = True
        patches = {
            "configure_tls": mock.patch.object(monitor, "configure_tls"),
            "fetch_all_sections": mock.patch.object(monitor, "fetch_all_sections"),
            "enrich": mock.patch.object(
                enrichment_pipeline, "enrich_announcement", side_effect=self.fake_enrich
            ),
            "process_documents": mock.patch.object(
                monitor, "process_enriched_announcements", return_value=make_document_result()
            ),
        }
        for name, patcher in patches.items():
            setattr(self, name, patcher.start())
            self.addCleanup(patcher.stop)

    def fake_enrich(self, announcement: dict) -> dict:
        if not self.enrich_ok:
            raise RuntimeError("enrichment недоступен")
        return {
            "enrichment_status": "success", "detail_title": announcement["title"],
            "cpv_codes": [], "documents": [],
        }

    def run_once(self, *announcements, enrich_ok=True) -> dict:
        self.enrich_ok = enrich_ok
        self.fetch_all_sections.return_value = list(announcements)
        self.process_documents.reset_mock()
        return monitor.run_monitor()

    def document_urls(self) -> list[str]:
        candidates = self.process_documents.call_args.args[0]
        return [a["resource_url"] for a in candidates]

    def state(self, number):
        return enrichment_repository.get_enrichment_processing_state(
            f"https://example.test/resource/{number}", self.db_path
        )

    def test_failed_refresh_of_updated_announcement_is_retried_next_run(self):
        url = "https://example.test/resource/1"

        # Запуск 1: новое объявление, enrichment успешен -> документы.
        first = self.run_once(make_announcement(1))
        self.assertEqual(first["new_count"], 1)
        self.assertEqual(self.state(1)["status"], "success")
        self.assertEqual(self.document_urls(), [url])
        old_enrichment = enrichment_repository.get_enrichment(url, self.db_path)
        self.assertEqual(old_enrichment["detail_title"], "Тестовое объявление 1")

        # Запуск 2: объявление обновилось, refresh упал. Старый enrichment остаётся,
        # документы по устаревшему enrichment НЕ обрабатываются (в том числе как pending).
        second = self.run_once(make_announcement(1, title="Изменённое название"), enrich_ok=False)
        self.assertEqual(second["updated_count"], 1)
        self.assertEqual(second["enrichment_result"]["failed_count"], 1)
        self.assertEqual(self.state(1)["status"], "failed")
        self.assertEqual(
            enrichment_repository.get_enrichment(url, self.db_path)["detail_title"],
            "Тестовое объявление 1",
        )
        self.assertEqual(self.document_urls(), [])
        self.assertEqual(second["pending_enrichment_after"], 1)

        # Запуск 3: объявление не менялось (existing), но failed refresh попадает в retry.
        # Успех очищает ошибку, и только теперь объявление идёт в document pipeline.
        third = self.run_once(make_announcement(1, title="Изменённое название"))
        self.assertEqual(third["existing_count"], 1)
        self.assertEqual(third["pending_retry_selected_count"], 1)
        self.assertEqual(third["enrichment_result"]["success_count"], 1)
        state = self.state(1)
        self.assertEqual(state["status"], "success")
        self.assertIsNone(state["error_type"])
        self.assertEqual(
            enrichment_repository.get_enrichment(url, self.db_path)["detail_title"],
            "Изменённое название",
        )
        self.assertEqual(self.document_urls(), [url])
        self.assertEqual(third["pending_enrichment_after"], 0)

        # Запуск 4: всё обработано, retry-кандидатов нет.
        fourth = self.run_once(make_announcement(1, title="Изменённое название"))
        self.assertEqual(fourth["pending_retry_selected_count"], 0)
        self.assertEqual(fourth["enrichment_candidate_count"], 0)

    def test_never_enriched_announcement_is_retried_before_old_failed_refresh(self):
        # Старые failed refresh не должны занимать слоты retry раньше объявления без enrichment.
        v2 = lambda a: {**a, "title": a["title"] + " (v2)"}
        old = [make_announcement(n) for n in (1, 2, 3)]
        self.run_once(*old)                                    # 1-3 обогащены
        self.run_once(*map(v2, old), enrich_ok=False)          # refresh 1-3 упал
        self.assertTrue(all(self.state(n)["status"] == "failed" for n in (1, 2, 3)))
        self.run_once(*map(v2, old), make_announcement(4), enrich_ok=False)  # 4 без enrichment
        self.assertIsNone(enrichment_repository.get_enrichment(
            "https://example.test/resource/4", self.db_path
        ))

        # Ничего нового: retry-слотов 3, кандидатов 4 (4 без enrichment + failed 1, 2, 3).
        result = self.run_once(*map(v2, old), make_announcement(4))

        processed = [r["resource_url"] for r in result["enrichment_result"]["results"]]
        self.assertEqual(result["pending_enrichment_before"], 4)
        self.assertEqual(result["pending_retry_selected_count"], 3)
        self.assertEqual(processed, [
            "https://example.test/resource/4",
            "https://example.test/resource/1",
            "https://example.test/resource/2",
        ])
        self.assertEqual(result["pending_enrichment_after"], 1)
        self.assertEqual(self.state(3)["status"], "failed")
        self.assertEqual(self.document_urls(), [
            "https://example.test/resource/4",
            "https://example.test/resource/1",
            "https://example.test/resource/2",
        ])



class PrintDocumentSummaryTest(unittest.TestCase):
    def render(self, result: dict) -> str:
        with mock.patch("builtins.print") as fake_print:
            monitor._print_document_summary(result)
        return "\n".join(str(c.args[0]) if c.args else "" for c in fake_print.call_args_list)

    def base_result(self, **document_overrides) -> dict:
        return {
            "document_candidate_count": 4,
            "pending_documents_before": 9,
            "pending_document_retry_selected_count": 3,
            "pending_documents_after": 6,
            "document_result": make_document_result(**document_overrides),
        }

    def test_summary_lines(self):
        text = self.render(self.base_result(
            success_count=2, no_supported_document_count=1, failed_count=0,
            download_new_count=1, download_existing_count=1,
            extraction_new_count=3, extraction_updated_count=1, extraction_existing_count=2,
        ))

        for line in (
            "Документы:", "Кандидатов: 4", "Pending до обработки: 9", "Старых pending выбрано: 3",
            "Успешно обработано: 2", "Без поддерживаемого документа: 1", "Ошибок: 0",
            "Новых downloads: 1", "Существующих downloads: 1",
            "Новых extractions: 3", "Обновлённых extractions: 1", "Существующих extractions: 2",
            "Pending после обработки: 6",
        ):
            self.assertIn(line, text)
        self.assertNotIn("Ошибки обработки документов", text)

    def test_failures_are_listed(self):
        text = self.render(self.base_result(
            failed_count=1,
            failures=[{"resource_url": "https://example.test/x", "error_type": "SSLError",
                       "error_message": "handshake failed"}],
        ))

        self.assertIn("Ошибки обработки документов:", text)
        self.assertIn("URL: https://example.test/x", text)
        self.assertIn("Тип ошибки: SSLError", text)
        self.assertIn("Сообщение: handshake failed", text)


if __name__ == "__main__":
    unittest.main()
