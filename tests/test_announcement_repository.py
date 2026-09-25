"""
Тесты announcement_repository на временной SQLite БД.

Запуск из корня проекта:
    python -m unittest tests.test_announcement_repository -v
"""

import logging
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

from src.database import announcement_repository as repo
from src.database import tender_repository


def make_announcement(**overrides) -> dict:
    announcement = {
        "title": "Тестовое объявление",
        "source_section": "open_competition",
        "source_section_name": "Открытый конкурс",
        "source_page_url": "https://gnumner.minfin.am/ru/page/test/",
        "resource_url": "https://example.test/resource/1",
        "resource_type": "direct_file",
        "published_at": "2026-09-25 18:59:34",
        "deadline_at": "2026-10-02 11:10:00",
        "tender_time_raw": "(опубликован 2026-09-25 18:59:34-от До 2026-10-02 11:10:00 час включен)",
    }
    announcement.update(overrides)
    return announcement


class AnnouncementRepositoryTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "test.db"
        repo.init_db(self.db_path)

        # Ожидаемые ошибки в тестах не должны засорять вывод логами с traceback.
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)

    def test_init_db_creates_table(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'announcements'"
            ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(repo.count_announcements(self.db_path), 0)

    def test_init_db_does_not_touch_tenders_table(self):
        tender_repository.init_db(self.db_path)
        tender_repository.save_tender(
            {"title": "Старый", "attachment_url": "https://example.test/old.pdf", "filename": "old.pdf"},
            self.db_path,
        )

        repo.init_db(self.db_path)

        self.assertEqual(tender_repository.count_tenders(self.db_path), 1)

    def test_first_save_is_new(self):
        status = repo.save_announcement(make_announcement(), self.db_path)

        self.assertEqual(status, "new")
        self.assertEqual(repo.count_announcements(self.db_path), 1)

    def test_identical_repeat_is_existing(self):
        repo.save_announcement(make_announcement(), self.db_path)

        status = repo.save_announcement(make_announcement(), self.db_path)

        self.assertEqual(status, "existing")
        self.assertEqual(repo.count_announcements(self.db_path), 1)

    def test_existing_updates_only_last_seen_at(self):
        with mock.patch.object(repo, "_utc_now", return_value="2026-01-01T00:00:00+00:00"):
            repo.save_announcement(make_announcement(), self.db_path)
        with mock.patch.object(repo, "_utc_now", return_value="2026-01-02T00:00:00+00:00"):
            repo.save_announcement(make_announcement(), self.db_path)

        row = repo.get_announcement_by_resource_url("https://example.test/resource/1", self.db_path)
        self.assertEqual(row["first_seen_at"], "2026-01-01T00:00:00+00:00")
        self.assertEqual(row["last_seen_at"], "2026-01-02T00:00:00+00:00")

    def test_changed_deadline_is_updated(self):
        repo.save_announcement(make_announcement(), self.db_path)

        status = repo.save_announcement(
            make_announcement(deadline_at="2026-10-09 12:00:00"), self.db_path
        )

        self.assertEqual(status, "updated")
        self.assertEqual(repo.count_announcements(self.db_path), 1)
        row = repo.get_announcement_by_resource_url("https://example.test/resource/1", self.db_path)
        self.assertEqual(row["deadline_at"], "2026-10-09 12:00:00")

    def test_deadline_removed_is_updated_to_null(self):
        repo.save_announcement(make_announcement(), self.db_path)

        status = repo.save_announcement(make_announcement(deadline_at=None), self.db_path)

        self.assertEqual(status, "updated")
        row = repo.get_announcement_by_resource_url("https://example.test/resource/1", self.db_path)
        self.assertIsNone(row["deadline_at"])

    def test_update_keeps_first_seen_at(self):
        with mock.patch.object(repo, "_utc_now", return_value="2026-01-01T00:00:00+00:00"):
            repo.save_announcement(make_announcement(), self.db_path)
        with mock.patch.object(repo, "_utc_now", return_value="2026-01-02T00:00:00+00:00"):
            repo.save_announcement(make_announcement(title="Новое название"), self.db_path)

        row = repo.get_announcement_by_resource_url("https://example.test/resource/1", self.db_path)
        self.assertEqual(row["title"], "Новое название")
        self.assertEqual(row["first_seen_at"], "2026-01-01T00:00:00+00:00")
        self.assertEqual(row["last_seen_at"], "2026-01-02T00:00:00+00:00")

    def test_batch_of_two_identical_announcements(self):
        result = repo.save_announcements(
            [make_announcement(), make_announcement()], self.db_path
        )

        self.assertEqual(repo.count_announcements(self.db_path), 1)
        self.assertEqual(result["new_count"], 1)
        self.assertEqual(result["existing_count"], 1)
        self.assertEqual(result["updated_count"], 0)
        self.assertEqual(result["failed_count"], 0)

    def test_batch_same_url_with_different_data_is_updated(self):
        result = repo.save_announcements(
            [make_announcement(), make_announcement(deadline_at="2026-10-09 12:00:00")],
            self.db_path,
        )

        self.assertEqual(repo.count_announcements(self.db_path), 1)
        self.assertEqual(result["new_count"], 1)
        self.assertEqual(result["updated_count"], 1)
        self.assertEqual(len(result["updated_announcements"]), 1)

    def test_batch_failure_does_not_stop_others(self):
        broken = make_announcement(resource_url="")
        good = make_announcement(resource_url="https://example.test/resource/2")

        result = repo.save_announcements([broken, good], self.db_path)

        self.assertEqual(result["failed_count"], 1)
        self.assertEqual(result["new_count"], 1)
        self.assertEqual(result["new_announcements"], [good])
        self.assertEqual(repo.count_announcements(self.db_path), 1)

    def test_missing_resource_url_raises(self):
        announcement = make_announcement()
        del announcement["resource_url"]

        with self.assertRaises(ValueError):
            repo.save_announcement(announcement, self.db_path)
        self.assertEqual(repo.count_announcements(self.db_path), 0)

    def test_each_required_field_is_checked(self):
        for field in repo.REQUIRED_FIELDS:
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    repo.save_announcement(make_announcement(**{field: ""}), self.db_path)

    def test_get_unknown_url_returns_none(self):
        self.assertIsNone(
            repo.get_announcement_by_resource_url("https://example.test/none", self.db_path)
        )


if __name__ == "__main__":
    unittest.main()
