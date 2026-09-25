"""
Тесты monitor.run_monitor без сети и без базы: все внешние вызовы замоканы.

Запуск из корня проекта:
    python -m unittest tests.test_monitor -v
"""

import copy
import unittest
from unittest import mock

from src import monitor


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
                monitor.enrichment_repository, "get_announcements_without_enrichment"
            ),
            "count_pending": mock.patch.object(
                monitor.enrichment_repository, "count_announcements_without_enrichment"
            ),
            "process_announcements": mock.patch.object(monitor, "process_announcements"),
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

    def processed_urls(self) -> list[str]:
        candidates = self.process_announcements.call_args.args[0]
        return [a["resource_url"] for a in candidates]

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


if __name__ == "__main__":
    unittest.main()
