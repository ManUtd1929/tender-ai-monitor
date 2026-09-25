"""
Тесты backfill_enrichment. Сеть и SQLite не используются: configure_tls,
enrichment_repository.init_db / count / get и process_announcements замоканы.

Запуск из корня проекта:
    python -m unittest tests.test_backfill_enrichment -v
"""

import copy
import unittest
from unittest import mock

from src import backfill_enrichment as backfill

DB_PATH = "test.db"

EAUCTION = "eauction_tender_page"
ARMEPS = "armeps_documents_page"
DIRECT = "direct_file"


def make_announcement(number, resource_type) -> dict:
    return {
        "title": f"Объявление {number}",
        "resource_url": f"https://example.test/resource/{number}",
        "resource_type": resource_type,
    }


class SelectSampleAnnouncementsTest(unittest.TestCase):
    def test_selects_one_of_each_type(self):
        announcements = [
            make_announcement(1, DIRECT),
            make_announcement(2, ARMEPS),
            make_announcement(3, EAUCTION),
        ]

        selected = backfill.select_sample_announcements(announcements)

        self.assertEqual(len(selected), 3)
        self.assertEqual({a["resource_type"] for a in selected}, {EAUCTION, ARMEPS, DIRECT})

    def test_result_order_is_eauction_armeps_direct(self):
        announcements = [
            make_announcement(1, DIRECT),
            make_announcement(2, ARMEPS),
            make_announcement(3, EAUCTION),
        ]

        selected = backfill.select_sample_announcements(announcements)

        self.assertEqual([a["resource_type"] for a in selected], [EAUCTION, ARMEPS, DIRECT])

    def test_missing_type_returns_the_rest(self):
        announcements = [make_announcement(1, DIRECT), make_announcement(2, EAUCTION)]

        selected = backfill.select_sample_announcements(announcements)

        self.assertEqual([a["resource_type"] for a in selected], [EAUCTION, DIRECT])

    def test_duplicates_of_one_type_give_single_first_record(self):
        announcements = [
            make_announcement(1, ARMEPS),
            make_announcement(2, ARMEPS),
            make_announcement(3, ARMEPS),
        ]

        selected = backfill.select_sample_announcements(announcements)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["resource_url"], "https://example.test/resource/1")

    def test_other_resource_types_are_ignored(self):
        announcements = [make_announcement(1, "unknown_type")]

        self.assertEqual(backfill.select_sample_announcements(announcements), [])

    def test_empty_list(self):
        self.assertEqual(backfill.select_sample_announcements([]), [])

    def test_source_list_is_not_modified(self):
        announcements = [
            make_announcement(1, DIRECT),
            make_announcement(2, ARMEPS),
            make_announcement(3, ARMEPS),
            make_announcement(4, EAUCTION),
        ]
        before = copy.deepcopy(announcements)

        selected = backfill.select_sample_announcements(announcements)

        self.assertEqual(announcements, before)
        self.assertIsNot(selected, announcements)


class RunSampleBackfillTest(unittest.TestCase):
    def setUp(self):
        self.calls = mock.Mock()  # общий журнал вызовов для проверки порядка

        self.pending = [
            make_announcement(1, DIRECT),
            make_announcement(2, ARMEPS),
            make_announcement(3, ARMEPS),
            make_announcement(4, EAUCTION),
            make_announcement(5, EAUCTION),
        ]
        self.pipeline_result = {
            "processed_count": 3,
            "success_count": 2,
            "failed_count": 1,
            "storage_new_count": 2,
            "storage_updated_count": 0,
            "storage_existing_count": 0,
            "results": [{"resource_url": "x"}],
            "failures": [{"resource_url": "y", "error_type": "E", "error_message": "m"}],
        }

        patches = {
            "configure_tls": mock.patch.object(backfill, "configure_tls"),
            "init_db": mock.patch.object(backfill.enrichment_repository, "init_db"),
            "count": mock.patch.object(
                backfill.enrichment_repository, "count_announcements_without_enrichment",
                side_effect=[5, 2],
            ),
            "get": mock.patch.object(
                backfill.enrichment_repository, "get_announcements_without_enrichment",
                return_value=self.pending,
            ),
            "process": mock.patch.object(
                backfill, "process_announcements", return_value=self.pipeline_result,
            ),
        }
        self.mocks = {}
        for name, patcher in patches.items():
            self.mocks[name] = patcher.start()
            self.addCleanup(patcher.stop)
            self.calls.attach_mock(self.mocks[name], name)

    def test_calls_configure_tls(self):
        backfill.run_sample_backfill(db_path=DB_PATH)

        self.mocks["configure_tls"].assert_called_once_with()

    def test_init_db_called_before_reading_pending(self):
        backfill.run_sample_backfill(db_path=DB_PATH)

        names = [call[0] for call in self.calls.mock_calls]
        self.assertLess(names.index("init_db"), names.index("count"))
        self.assertLess(names.index("init_db"), names.index("get"))
        self.mocks["init_db"].assert_called_once_with(db_path=DB_PATH)

    def test_pending_count_before_is_taken_before_processing(self):
        result = backfill.run_sample_backfill(db_path=DB_PATH)

        self.assertEqual(result["pending_count_before"], 5)
        names = [call[0] for call in self.calls.mock_calls]
        self.assertLess(names.index("count"), names.index("process"))

    def test_process_receives_only_selected_three(self):
        result = backfill.run_sample_backfill(db_path=DB_PATH)

        args, kwargs = self.mocks["process"].call_args
        processed = args[0]
        self.assertEqual(len(processed), 3)
        self.assertEqual([a["resource_type"] for a in processed], [EAUCTION, ARMEPS, DIRECT])
        self.assertEqual([a["resource_url"] for a in processed], [
            "https://example.test/resource/4",
            "https://example.test/resource/2",
            "https://example.test/resource/1",
        ])
        self.assertEqual(kwargs, {"db_path": DB_PATH})
        self.assertEqual(result["selected_count"], 3)

    def test_pending_count_after_is_taken_after_processing(self):
        result = backfill.run_sample_backfill(db_path=DB_PATH)

        self.assertEqual(result["pending_count_after"], 2)
        self.assertEqual(self.mocks["count"].call_count, 2)
        names = [call[0] for call in self.calls.mock_calls]
        last_count = len(names) - 1 - names[::-1].index("count")
        self.assertGreater(last_count, names.index("process"))

    def test_pipeline_result_returned_unchanged(self):
        expected = copy.deepcopy(self.pipeline_result)

        result = backfill.run_sample_backfill(db_path=DB_PATH)

        self.assertIs(result["pipeline_result"], self.pipeline_result)
        self.assertEqual(result["pipeline_result"], expected)

    def test_selected_summary_fields(self):
        result = backfill.run_sample_backfill(db_path=DB_PATH)

        self.assertEqual(result["selected"][0], {
            "resource_url": "https://example.test/resource/4",
            "resource_type": EAUCTION,
            "title": "Объявление 4",
        })

    def test_empty_pending_passes_empty_list_to_process(self):
        self.mocks["get"].return_value = []
        self.mocks["count"].side_effect = [0, 0]

        result = backfill.run_sample_backfill(db_path=DB_PATH)

        args, _ = self.mocks["process"].call_args
        self.assertEqual(args[0], [])
        self.assertEqual(result["selected_count"], 0)
        self.assertEqual(result["selected"], [])


if __name__ == "__main__":
    unittest.main()
