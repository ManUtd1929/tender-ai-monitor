"""
Тесты enrichment_pipeline. PipelineTest не использует сеть и SQLite: enrich_announcement
и функции enrichment_repository замоканы. PipelineWithRepositoryTest работает на временной
SQLite БД (замокан только enrich_announcement).

Запуск из корня проекта:
    python -m unittest tests.test_enrichment_pipeline -v
"""

import copy
import logging
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

from src import enrichment_pipeline as pipeline
from src.database import announcement_repository, enrichment_repository

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
        self.save_state = self._patch_repo("save_enrichment_processing_state")

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

    def test_success_state_is_saved_after_enrichment(self):
        calls = mock.Mock()
        calls.attach_mock(self.save, "save")
        calls.attach_mock(self.save_state, "save_state")
        announcement = make_announcement(3)

        pipeline.process_announcement(announcement, db_path=DB_PATH)

        self.assertEqual([c[0] for c in calls.mock_calls], ["save", "save_state"])
        self.save_state.assert_called_once_with(
            "https://example.test/resource/3", "success", db_path=DB_PATH
        )

    def test_save_failure_does_not_save_success_state(self):
        self.save.side_effect = ValueError("Объявление не найдено")

        with self.assertRaises(ValueError):
            pipeline.process_announcement(make_announcement(), db_path=DB_PATH)

        self.save_state.assert_not_called()

    def test_process_announcement_itself_never_saves_failed_state(self):
        # failed пишет только process_announcements, который перехватывает ошибку.
        self.enrich.side_effect = RuntimeError("сеть недоступна")

        with self.assertRaises(RuntimeError):
            pipeline.process_announcement(make_announcement(), db_path=DB_PATH)

        self.save_state.assert_not_called()

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

    def test_enrich_failure_saves_failed_state(self):
        self.enrich.side_effect = RuntimeError("сеть недоступна")

        pipeline.process_announcements([make_announcement(5)], db_path=DB_PATH)

        self.save_state.assert_called_once_with(
            "https://example.test/resource/5", "failed",
            error_type="RuntimeError", error_message="сеть недоступна", db_path=DB_PATH,
        )

    def test_save_failure_saves_failed_state(self):
        self.save.side_effect = ValueError("Объявление не найдено")

        pipeline.process_announcements([make_announcement(6)], db_path=DB_PATH)

        self.save_state.assert_called_once_with(
            "https://example.test/resource/6", "failed",
            error_type="ValueError", error_message="Объявление не найдено", db_path=DB_PATH,
        )

    def test_states_saved_per_item_success_and_failed(self):
        def enrich(announcement):
            if announcement["resource_url"].endswith("/2"):
                raise RuntimeError("сбой")
            return fake_enrich(announcement)

        self.enrich.side_effect = enrich

        pipeline.process_announcements([make_announcement(n) for n in (1, 2, 3)], db_path=DB_PATH)

        self.assertEqual(
            [(c.args[0], c.args[1]) for c in self.save_state.call_args_list],
            [
                ("https://example.test/resource/1", "success"),
                ("https://example.test/resource/2", "failed"),
                ("https://example.test/resource/3", "success"),
            ],
        )

    def test_failed_state_write_error_keeps_original_failure(self):
        self.enrich.side_effect = RuntimeError("сеть недоступна")
        self.save_state.side_effect = sqlite3.OperationalError("database is locked")

        with mock.patch.object(pipeline.logger, "exception") as log_exception:
            summary = pipeline.process_announcements([make_announcement(5)], db_path=DB_PATH)

        self.assertEqual(summary["failed_count"], 1)
        self.assertEqual(summary["failures"], [{
            "resource_url": "https://example.test/resource/5",
            "error_type": "RuntimeError",
            "error_message": "сеть недоступна",
        }])
        # Один вызов — исходная ошибка enrichment, второй — сбой записи состояния.
        self.assertEqual(log_exception.call_count, 2)

    def test_failed_state_write_error_does_not_stop_others(self):
        def enrich(announcement):
            if announcement["resource_url"].endswith("/1"):
                raise RuntimeError("сбой")
            return fake_enrich(announcement)

        self.enrich.side_effect = enrich
        self.save_state.side_effect = [sqlite3.OperationalError("locked"), None, None]

        summary = pipeline.process_announcements(
            [make_announcement(n) for n in (1, 2, 3)], db_path=DB_PATH
        )

        self.assertEqual(summary["success_count"], 2)
        self.assertEqual(summary["failed_count"], 1)

    def test_success_state_write_error_is_a_failure_with_failed_state(self):
        # Enrichment сохранён, но состояние success записать не удалось: считаем ошибкой
        # и пытаемся записать failed (тогда объявление будет повторено, а не потеряно).
        self.save_state.side_effect = [sqlite3.OperationalError("locked"), None]

        summary = pipeline.process_announcements([make_announcement(1)], db_path=DB_PATH)

        self.assertEqual(summary["failed_count"], 1)
        self.assertEqual(self.save_state.call_args_list[-1].args[1], "failed")

    def test_failure_without_resource_url_saves_no_state(self):
        self.enrich.side_effect = RuntimeError("сбой")

        summary = pipeline.process_announcements(
            [make_announcement(1, resource_url=None)], db_path=DB_PATH
        )

        self.assertEqual(summary["failed_count"], 1)
        self.save_state.assert_not_called()

    def test_fatal_init_db_error_creates_no_states(self):
        self.init_db.side_effect = RuntimeError("db locked")

        with self.assertRaises(RuntimeError):
            pipeline.process_announcements([make_announcement(1)], db_path=DB_PATH)

        self.enrich.assert_not_called()
        self.save.assert_not_called()
        self.save_state.assert_not_called()

    def test_empty_list(self):
        summary = pipeline.process_announcements([], db_path=DB_PATH)

        self.assertEqual(summary["processed_count"], 0)
        self.assertEqual(summary["success_count"], 0)
        self.assertEqual(summary["failed_count"], 0)
        self.assertEqual(summary["results"], [])
        self.assertEqual(summary["failures"], [])
        self.enrich.assert_not_called()
        self.save.assert_not_called()
        self.save_state.assert_not_called()


class PipelineWithRepositoryTest(unittest.TestCase):
    """Реальная временная БД: состояние enrichment и повторные попытки."""

    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "test.db"

        enrichment_repository.init_db(self.db_path)
        self.announcement = make_announcement(1)
        announcement_repository.save_announcement(self.announcement, self.db_path)

    def process(self, enrich):
        with mock.patch.object(pipeline, "enrich_announcement", side_effect=enrich):
            return pipeline.process_announcements([self.announcement], db_path=self.db_path)

    def state(self):
        return enrichment_repository.get_enrichment_processing_state(
            self.announcement["resource_url"], self.db_path
        )

    def candidate_urls(self):
        return [
            a["resource_url"]
            for a in enrichment_repository.get_enrichment_processing_candidates(self.db_path)
        ]

    def fail(self, message="сеть недоступна"):
        def enrich(announcement):
            raise RuntimeError(message)
        return enrich

    def test_success_saves_success_state_without_error(self):
        summary = self.process(fake_enrich)

        self.assertEqual(summary["success_count"], 1)
        state = self.state()
        self.assertEqual(state["status"], "success")
        self.assertIsNone(state["error_type"])
        self.assertIsNone(state["error_message"])
        self.assertEqual(self.candidate_urls(), [])

    def test_new_announcement_failure_saves_failed_state_and_stays_candidate(self):
        summary = self.process(self.fail())

        self.assertEqual(summary["failed_count"], 1)
        state = self.state()
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["error_type"], "RuntimeError")
        self.assertEqual(state["error_message"], "сеть недоступна")
        self.assertIsNone(enrichment_repository.get_enrichment(
            self.announcement["resource_url"], self.db_path
        ))
        self.assertEqual(self.candidate_urls(), [self.announcement["resource_url"]])

    def test_failed_refresh_keeps_old_enrichment_and_becomes_retry_candidate(self):
        self.process(fake_enrich)
        old_enrichment = enrichment_repository.get_enrichment(
            self.announcement["resource_url"], self.db_path
        )
        self.assertEqual(self.candidate_urls(), [])

        self.process(self.fail("refresh упал"))

        self.assertEqual(self.state()["status"], "failed")
        self.assertEqual(
            enrichment_repository.get_enrichment(self.announcement["resource_url"], self.db_path),
            old_enrichment,
        )
        self.assertEqual(self.candidate_urls(), [self.announcement["resource_url"]])

    def test_success_after_failed_clears_error_and_candidate(self):
        self.process(self.fail())
        self.assertEqual(self.state()["status"], "failed")

        self.process(fake_enrich)

        state = self.state()
        self.assertEqual(state["status"], "success")
        self.assertIsNone(state["error_type"])
        self.assertIsNone(state["error_message"])
        self.assertEqual(self.candidate_urls(), [])

    def test_failed_state_write_failure_keeps_failure_in_result(self):
        with mock.patch.object(
            enrichment_repository, "save_enrichment_processing_state",
            side_effect=sqlite3.OperationalError("database is locked"),
        ):
            summary = self.process(self.fail())

        self.assertEqual(summary["failures"][0]["error_type"], "RuntimeError")
        self.assertIsNone(self.state())

    def test_fatal_init_db_error_leaves_no_states(self):
        with mock.patch.object(enrichment_repository, "init_db", side_effect=RuntimeError("locked")):
            with self.assertRaises(RuntimeError):
                pipeline.process_announcements([self.announcement], db_path=self.db_path)

        with closing(sqlite3.connect(self.db_path)) as conn:
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM enrichment_processing_state").fetchone()[0], 0
            )


if __name__ == "__main__":
    unittest.main()
