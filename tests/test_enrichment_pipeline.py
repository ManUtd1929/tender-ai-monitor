"""
Тесты enrichment_pipeline. Сеть и SQLite не используются: enrich_announcement,
enrichment_repository.init_db и save_enrichment замоканы.

Запуск из корня проекта:
    python -m unittest tests.test_enrichment_pipeline -v
"""

import copy
import logging
import unittest
from unittest import mock

from src import enrichment_pipeline as pipeline

DB_PATH = "test.db"


def make_announcement(number=1, **overrides) -> dict:
    announcement = {
        "title": f"Объявление {number}",
        "source_section": "open_competition",
        "source_section_name": "Открытый конкурс",
        "source_page_url": "https://gnumner.minfin.am/ru/page/test/",
        "resource_url": f"https://example.test/resource/{number}",
        "resource_type": "armeps_documents_page",
        "published_at": "2026-09-25 18:59:34",
        "deadline_at": "2026-10-02 11:10:00",
        "tender_time_raw": None,
    }
    announcement.update(overrides)
    return announcement


def fake_enrich(announcement: dict) -> dict:
    """Как настоящая: новый dict = копия announcement + enrichment_status."""
    enriched = dict(announcement)
    enriched["enrichment_status"] = "success"
    return enriched


class PipelineTest(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)

        self.enrich = self._patch("enrich_announcement", side_effect=fake_enrich)
        self.init_db = self._patch_repo("init_db")
        self.save = self._patch_repo("save_enrichment", return_value="new")

    def _patch(self, name, **kwargs):
        patcher = mock.patch.object(pipeline, name, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def _patch_repo(self, name, **kwargs):
        patcher = mock.patch.object(pipeline.enrichment_repository, name, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    # --- process_announcement ---

    def test_process_announcement_calls_enrich(self):
        announcement = make_announcement()

        pipeline.process_announcement(announcement, db_path=DB_PATH)

        self.enrich.assert_called_once_with(announcement)

    def test_enrichment_saved_under_correct_resource_url(self):
        announcement = make_announcement(7)

        pipeline.process_announcement(announcement, db_path=DB_PATH)

        self.save.assert_called_once_with(
            "https://example.test/resource/7", fake_enrich(announcement), db_path=DB_PATH
        )

    def test_storage_status_returned(self):
        self.save.return_value = "updated"
        announcement = make_announcement()

        result = pipeline.process_announcement(announcement, db_path=DB_PATH)

        self.assertEqual(result["resource_url"], announcement["resource_url"])
        self.assertEqual(result["enrichment_status"], "success")
        self.assertEqual(result["storage_status"], "updated")
        self.assertEqual(result["enrichment"], fake_enrich(announcement))

    def test_source_announcement_not_modified(self):
        announcement = make_announcement()
        original = copy.deepcopy(announcement)

        pipeline.process_announcement(announcement, db_path=DB_PATH)

        self.assertEqual(announcement, original)
        self.assertNotIn("enrichment_status", announcement)

    def test_direct_file_not_required_is_still_saved(self):
        announcement = make_announcement(resource_type="direct_file")
        self.enrich.side_effect = lambda a: {
            **a, "direct_file_url": a["resource_url"], "enrichment_status": "not_required",
        }

        result = pipeline.process_announcement(announcement, db_path=DB_PATH)

        self.save.assert_called_once()
        resource_url, enrichment = self.save.call_args.args
        self.assertEqual(resource_url, announcement["resource_url"])
        self.assertEqual(enrichment["enrichment_status"], "not_required")
        self.assertEqual(result["enrichment_status"], "not_required")

    def test_unsupported_is_still_saved(self):
        announcement = make_announcement(resource_type="unknown")
        self.enrich.side_effect = lambda a: {**a, "enrichment_status": "unsupported"}

        result = pipeline.process_announcement(announcement, db_path=DB_PATH)

        self.assertEqual(self.save.call_args.args[1]["enrichment_status"], "unsupported")
        self.assertEqual(result["enrichment_status"], "unsupported")

    # --- process_announcements ---

    def test_init_db_called_once(self):
        announcements = [make_announcement(n) for n in (1, 2, 3)]

        pipeline.process_announcements(announcements, db_path=DB_PATH)

        self.init_db.assert_called_once_with(db_path=DB_PATH)

    def test_three_successful_announcements(self):
        announcements = [make_announcement(n) for n in (1, 2, 3)]

        summary = pipeline.process_announcements(announcements, db_path=DB_PATH)

        self.assertEqual(summary["processed_count"], 3)
        self.assertEqual(summary["success_count"], 3)
        self.assertEqual(summary["failed_count"], 0)
        self.assertEqual(len(summary["results"]), 3)
        self.assertEqual(summary["failures"], [])
        self.assertEqual(self.save.call_count, 3)

    def test_enrich_failure_does_not_stop_others(self):
        def enrich(announcement):
            if announcement["resource_url"].endswith("/2"):
                raise RuntimeError("сеть недоступна")
            return fake_enrich(announcement)

        self.enrich.side_effect = enrich
        announcements = [make_announcement(n) for n in (1, 2, 3)]

        summary = pipeline.process_announcements(announcements, db_path=DB_PATH)

        self.assertEqual(summary["processed_count"], 3)
        self.assertEqual(summary["success_count"], 2)
        self.assertEqual(summary["failed_count"], 1)
        saved_urls = [call.args[0] for call in self.save.call_args_list]
        self.assertEqual(
            saved_urls, ["https://example.test/resource/1", "https://example.test/resource/3"]
        )

    def test_save_failure_does_not_stop_others(self):
        def save(resource_url, enrichment, db_path=None):
            if resource_url.endswith("/1"):
                raise ValueError("Объявление не найдено")
            return "new"

        self.save.side_effect = save
        announcements = [make_announcement(n) for n in (1, 2, 3)]

        summary = pipeline.process_announcements(announcements, db_path=DB_PATH)

        self.assertEqual(summary["processed_count"], 3)
        self.assertEqual(summary["success_count"], 2)
        self.assertEqual(summary["failed_count"], 1)
        self.assertEqual(self.enrich.call_count, 3)

    def test_storage_counts(self):
        self.save.side_effect = ["new", "updated", "existing", "existing", "new"]
        announcements = [make_announcement(n) for n in range(1, 6)]

        summary = pipeline.process_announcements(announcements, db_path=DB_PATH)

        self.assertEqual(summary["storage_new_count"], 2)
        self.assertEqual(summary["storage_updated_count"], 1)
        self.assertEqual(summary["storage_existing_count"], 2)
        self.assertEqual(
            [r["storage_status"] for r in summary["results"]],
            ["new", "updated", "existing", "existing", "new"],
        )

    def test_failure_contains_details(self):
        self.enrich.side_effect = RuntimeError("сеть недоступна")

        summary = pipeline.process_announcements([make_announcement(5)], db_path=DB_PATH)

        self.assertEqual(
            summary["failures"],
            [{
                "resource_url": "https://example.test/resource/5",
                "error_type": "RuntimeError",
                "error_message": "сеть недоступна",
            }],
        )
        self.assertNotIn("traceback", summary["failures"][0])

    def test_empty_list(self):
        summary = pipeline.process_announcements([], db_path=DB_PATH)

        self.assertEqual(summary["processed_count"], 0)
        self.assertEqual(summary["success_count"], 0)
        self.assertEqual(summary["failed_count"], 0)
        self.assertEqual(summary["results"], [])
        self.assertEqual(summary["failures"], [])
        self.enrich.assert_not_called()
        self.save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
