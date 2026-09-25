"""
Тесты monitor.run_monitor без сети и без базы: все внешние вызовы замоканы.

Запуск из корня проекта:
    python -m unittest tests.test_monitor -v
"""

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


class RunMonitorTest(unittest.TestCase):
    def setUp(self):
        patches = {
            "configure_tls": mock.patch.object(monitor, "configure_tls"),
            "init_db": mock.patch.object(monitor, "init_db"),
            "fetch_all_sections": mock.patch.object(monitor, "fetch_all_sections"),
            "save_announcements": mock.patch.object(monitor, "save_announcements"),
            "count_announcements": mock.patch.object(monitor, "count_announcements"),
        }
        for name, patcher in patches.items():
            setattr(self, name, patcher.start())
            self.addCleanup(patcher.stop)

        self.fetch_all_sections.return_value = []
        self.save_announcements.return_value = make_save_result()
        self.count_announcements.return_value = 0

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


if __name__ == "__main__":
    unittest.main()
