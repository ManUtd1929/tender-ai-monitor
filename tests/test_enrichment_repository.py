"""
Тесты enrichment_repository на временной SQLite БД.

Запуск из корня проекта:
    python -m unittest tests.test_enrichment_repository -v
"""

import logging
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

from src.database import announcement_repository, enrichment_repository as repo

RESOURCE_URL = "https://example.test/resource/1"

ENRICHMENT_TABLES = ("announcement_enrichment", "announcement_cpv", "announcement_documents")


def make_announcement(**overrides) -> dict:
    announcement = {
        "title": "Тестовое объявление",
        "source_section": "open_competition",
        "source_section_name": "Открытый конкурс",
        "source_page_url": "https://gnumner.minfin.am/ru/page/test/",
        "resource_url": RESOURCE_URL,
        "resource_type": "armeps_documents_page",
        "published_at": "2026-09-25 18:59:34",
        "deadline_at": "2026-10-02 11:10:00",
        "tender_time_raw": "(опубликован 2026-09-25 18:59:34-от До 2026-10-02 11:10:00 час включен)",
    }
    announcement.update(overrides)
    return announcement


def make_enrichment(**overrides) -> dict:
    enrichment = {
        "enrichment_status": "success",
        "procedure_code": "TEST-001",
        "contracting_authority": "Тестовый заказчик",
        "detail_title": "Тестовый заголовок",
        "detail_title_ru": "Тестовый заголовок RU",
        "detail_title_en": "Test title EN",
        "procurement_type": "Услуги",
        "procedure_type": "Открытый конкурс",
        "description": "Описание",
        "estimated_value_amd": "1000000",
        "published_at_detail": "2026-09-25 18:59:34",
        "deadline_at_detail": "2026-10-02 11:10:00",
        "number_of_lots": 2,
        "resource_id": "12345",
        "document_url": None,
        "consistency": {"published_at_match": True, "deadline_at_match": True},
    }
    enrichment.update(overrides)
    return enrichment


CPV_A = {"code": "60100000", "name": "Road transport services"}
CPV_B = {"code": "63100000", "name": "Cargo handling"}
DOC_1 = {"document_id": "101", "filename": "a.pdf", "language": "hy", "title": "A", "description": "da"}
DOC_2 = {"document_id": "102", "filename": "b.docx", "language": "en", "title": "B", "description": None}


class EnrichmentRepositoryTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "test.db"

        announcement_repository.init_db(self.db_path)
        announcement_repository.save_announcement(make_announcement(), self.db_path)
        repo.init_db(self.db_path)

        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)

    def save(self, enrichment=None, resource_url=RESOURCE_URL):
        if enrichment is None:
            enrichment = make_enrichment()
        return repo.save_enrichment(resource_url, enrichment, self.db_path)

    def get(self):
        return repo.get_enrichment(RESOURCE_URL, self.db_path)

    def count_rows(self, table):
        with closing(sqlite3.connect(self.db_path)) as conn:
            return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def announcements_snapshot(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            schema = conn.execute("PRAGMA table_info(announcements)").fetchall()
            rows = conn.execute("SELECT * FROM announcements ORDER BY id").fetchall()
        return schema, rows

    # --- схема ---

    def test_init_db_creates_three_tables(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            names = {
                row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
        for table in ENRICHMENT_TABLES:
            self.assertIn(table, names)

    def test_announcements_table_and_row_unchanged(self):
        before = self.announcements_snapshot()

        repo.init_db(self.db_path)
        self.save(make_enrichment(cpv_codes=[CPV_A], documents=[DOC_1]))

        self.assertEqual(self.announcements_snapshot(), before)
        self.assertEqual(announcement_repository.count_announcements(self.db_path), 1)

    # --- статусы и scalar ---

    def test_first_save_is_new(self):
        self.assertEqual(self.save(), "new")

    def test_same_save_is_existing(self):
        self.save()
        self.assertEqual(self.save(), "existing")

    def test_changed_detail_title_is_updated(self):
        self.save()
        status = self.save(make_enrichment(detail_title="Другой заголовок"))

        self.assertEqual(status, "updated")
        self.assertEqual(self.get()["detail_title"], "Другой заголовок")

    def test_first_save_and_same_save_keep_single_row(self):
        self.save()
        self.save()

        self.assertEqual(self.count_rows("announcement_enrichment"), 1)
        self.assertEqual(repo.count_enrichments(self.db_path), 1)

    def test_scalar_fields_round_trip(self):
        self.save()
        result = self.get()

        for name, value in make_enrichment().items():
            if name != "consistency":
                self.assertEqual(result[name], value, name)
        self.assertTrue(result["enriched_at"].endswith("+00:00"))
        self.assertNotIn("resource_type", result)

    def test_consistency_round_trip(self):
        cases = [
            {"published_at_match": True, "deadline_at_match": False},
            {"published_at_match": False, "deadline_at_match": None},
            {"published_at_match": None, "deadline_at_match": True},
        ]
        for consistency in cases:
            with self.subTest(consistency=consistency):
                self.save(make_enrichment(consistency=consistency))
                self.assertEqual(self.get()["consistency"], consistency)

    def test_consistency_stored_as_integers(self):
        self.save(make_enrichment(consistency={"published_at_match": True, "deadline_at_match": False}))

        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute(
                "SELECT published_at_match, deadline_at_match FROM announcement_enrichment"
            ).fetchone()
        self.assertEqual(row, (1, 0))

    def test_missing_consistency_is_none(self):
        enrichment = make_enrichment()
        del enrichment["consistency"]
        self.save(enrichment)

        self.assertEqual(
            self.get()["consistency"], {"published_at_match": None, "deadline_at_match": None}
        )

    def test_get_unknown_returns_none(self):
        self.assertIsNone(repo.get_enrichment("https://example.test/none", self.db_path))

    # --- CPV ---

    def test_two_cpv_saved_and_read(self):
        self.save(make_enrichment(cpv_codes=[CPV_A, CPV_B]))
        self.assertEqual(self.get()["cpv_codes"], [CPV_A, CPV_B])

    def test_cpv_update_replaces_old(self):
        self.save(make_enrichment(cpv_codes=[CPV_A, CPV_B]))
        status = self.save(make_enrichment(cpv_codes=[CPV_B]))

        self.assertEqual(status, "updated")
        self.assertEqual(self.get()["cpv_codes"], [CPV_B])

    def test_missing_cpv_key_keeps_existing(self):
        self.save(make_enrichment(cpv_codes=[CPV_A, CPV_B]))
        status = self.save(make_enrichment())

        self.assertEqual(status, "existing")
        self.assertEqual(self.get()["cpv_codes"], [CPV_A, CPV_B])

    def test_none_cpv_value_keeps_existing(self):
        self.save(make_enrichment(cpv_codes=[CPV_A]))
        self.save(make_enrichment(cpv_codes=None))

        self.assertEqual(self.get()["cpv_codes"], [CPV_A])

    def test_empty_cpv_list_deletes_existing(self):
        self.save(make_enrichment(cpv_codes=[CPV_A, CPV_B]))
        status = self.save(make_enrichment(cpv_codes=[]))

        self.assertEqual(status, "updated")
        self.assertEqual(self.get()["cpv_codes"], [])

    def test_only_cpv_change_is_updated(self):
        self.save(make_enrichment(cpv_codes=[CPV_A]))
        self.assertEqual(self.save(make_enrichment(cpv_codes=[CPV_A])), "existing")
        self.assertEqual(self.save(make_enrichment(cpv_codes=[CPV_A, CPV_B])), "updated")

    def test_cpv_without_code_skipped_and_duplicates_collapsed(self):
        self.save(make_enrichment(cpv_codes=[{"code": None, "name": "без кода"}, CPV_A, CPV_A]))
        self.assertEqual(self.get()["cpv_codes"], [CPV_A])

    def test_cpv_of_other_announcement_untouched(self):
        other_url = "https://example.test/resource/2"
        announcement_repository.save_announcement(
            make_announcement(resource_url=other_url), self.db_path
        )
        self.save(make_enrichment(cpv_codes=[CPV_A]), other_url)

        self.save(make_enrichment(cpv_codes=[CPV_B]))

        self.assertEqual(repo.get_enrichment(other_url, self.db_path)["cpv_codes"], [CPV_A])

    # --- документы ---

    def test_multiple_documents_saved(self):
        self.save(make_enrichment(documents=[DOC_1, DOC_2]))
        self.assertEqual(self.get()["documents"], [DOC_1, DOC_2])

    def test_documents_update_replaces_old(self):
        self.save(make_enrichment(documents=[DOC_1, DOC_2]))
        status = self.save(make_enrichment(documents=[DOC_2]))

        self.assertEqual(status, "updated")
        self.assertEqual(self.get()["documents"], [DOC_2])

    def test_missing_documents_key_keeps_existing(self):
        self.save(make_enrichment(documents=[DOC_1, DOC_2]))
        status = self.save(make_enrichment())

        self.assertEqual(status, "existing")
        self.assertEqual(self.get()["documents"], [DOC_1, DOC_2])

    def test_empty_documents_list_deletes_existing(self):
        self.save(make_enrichment(documents=[DOC_1, DOC_2]))
        status = self.save(make_enrichment(documents=[]))

        self.assertEqual(status, "updated")
        self.assertEqual(self.get()["documents"], [])

    def test_document_without_filename_skipped(self):
        broken = {**DOC_2, "filename": None}
        blank = {**DOC_2, "filename": "  "}
        status = self.save(make_enrichment(documents=[DOC_1, broken, blank]))

        self.assertEqual(status, "new")
        self.assertEqual(self.get()["documents"], [DOC_1])

    def test_only_documents_change_is_updated(self):
        self.save(make_enrichment(documents=[DOC_1]))
        self.assertEqual(self.save(make_enrichment(documents=[DOC_1])), "existing")
        self.assertEqual(self.save(make_enrichment(documents=[DOC_1, DOC_2])), "updated")

    def test_documents_without_document_id_deduplicated(self):
        doc = {"document_id": None, "filename": "x.pdf"}
        self.save(make_enrichment(documents=[doc, dict(doc)]))

        self.assertEqual(self.count_rows("announcement_documents"), 1)
        self.assertEqual(self.save(make_enrichment(documents=[doc])), "existing")

    # --- eAuction ---

    def test_eauction_document_url_not_in_documents_table(self):
        zip_url = "https://example.test/tender_12345.zip"
        self.save(make_enrichment(document_url=zip_url))

        self.assertEqual(self.get()["document_url"], zip_url)
        self.assertEqual(self.get()["documents"], [])
        self.assertEqual(self.count_rows("announcement_documents"), 0)

    # --- валидация ---

    def test_unknown_resource_url_raises(self):
        with self.assertRaises(ValueError):
            self.save(resource_url="https://example.test/unknown")
        self.assertEqual(repo.count_enrichments(self.db_path), 0)

    def test_empty_resource_url_raises(self):
        for value in ("", "   ", None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.save(resource_url=value)

    def test_missing_enrichment_status_raises(self):
        enrichment = make_enrichment()
        del enrichment["enrichment_status"]

        with self.assertRaises(ValueError):
            self.save(enrichment)
        with self.assertRaises(ValueError):
            self.save(make_enrichment(enrichment_status=""))
        self.assertEqual(repo.count_enrichments(self.db_path), 0)

    def test_foreign_keys_enabled_on_connection(self):
        with repo._connect(self.db_path) as conn:
            self.assertEqual(conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)

    # --- транзакция ---

    def test_sql_error_rolls_back_everything(self):
        with mock.patch.object(repo, "INSERT_DOCUMENT", "INSERT INTO no_such_table VALUES (?)"):
            with self.assertRaises(sqlite3.Error):
                self.save(make_enrichment(cpv_codes=[CPV_A], documents=[DOC_1]))

        self.assertEqual(self.count_rows("announcement_enrichment"), 0)
        self.assertEqual(self.count_rows("announcement_cpv"), 0)
        self.assertEqual(self.count_rows("announcement_documents"), 0)

    # --- счётчик ---

    def test_count_enrichments(self):
        self.assertEqual(repo.count_enrichments(self.db_path), 0)

        other_url = "https://example.test/resource/2"
        announcement_repository.save_announcement(
            make_announcement(resource_url=other_url), self.db_path
        )
        self.save()
        self.save(make_enrichment(), other_url)
        self.save()

        self.assertEqual(repo.count_enrichments(self.db_path), 2)


if __name__ == "__main__":
    unittest.main()
