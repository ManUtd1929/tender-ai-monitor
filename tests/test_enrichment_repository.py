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
from datetime import datetime
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

    # --- announcements без enrichment (pending) ---

    def add_announcement(self, number, first_seen_at=None):
        url = f"https://example.test/resource/{number}"
        announcement_repository.save_announcement(
            make_announcement(resource_url=url, title=f"Объявление {number}"), self.db_path
        )
        if first_seen_at is not None:
            with closing(sqlite3.connect(self.db_path)) as conn, conn:
                conn.execute(
                    "UPDATE announcements SET first_seen_at = ? WHERE resource_url = ?",
                    (first_seen_at, url),
                )
        return url

    def pending_urls(self, **kwargs):
        pending = repo.get_announcements_without_enrichment(self.db_path, **kwargs)
        return [item["resource_url"] for item in pending]

    def test_announcement_without_enrichment_is_pending(self):
        self.assertEqual(self.pending_urls(), [RESOURCE_URL])

    def test_saved_enrichment_is_not_pending(self):
        self.save()
        self.assertEqual(self.pending_urls(), [])

    def test_any_enrichment_status_is_not_pending(self):
        for status in ("success", "partial", "not_required", "unsupported"):
            with self.subTest(status=status):
                url = self.add_announcement(status)
                self.assertIn(url, self.pending_urls())

                self.save(make_enrichment(enrichment_status=status), url)

                self.assertNotIn(url, self.pending_urls())

    def test_pending_ordered_by_first_seen_at_then_id(self):
        # RESOURCE_URL (id=1) получает самую позднюю дату; b и c — одинаковую дату, порядок по id.
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute(
                "UPDATE announcements SET first_seen_at = '2026-09-03T00:00:00+00:00' "
                "WHERE resource_url = ?",
                (RESOURCE_URL,),
            )
        url_b = self.add_announcement("b", "2026-09-01T00:00:00+00:00")
        url_c = self.add_announcement("c", "2026-09-01T00:00:00+00:00")
        url_d = self.add_announcement("d", "2026-09-02T00:00:00+00:00")

        self.assertEqual(self.pending_urls(), [url_b, url_c, url_d, RESOURCE_URL])

    def test_limit_one_returns_single_oldest(self):
        self.add_announcement(2)
        self.assertEqual(self.pending_urls(limit=1), [RESOURCE_URL])

    def test_limit_none_returns_all(self):
        for number in (2, 3):
            self.add_announcement(number)
        self.assertEqual(len(self.pending_urls(limit=None)), 3)
        self.assertEqual(len(self.pending_urls()), 3)

    def test_non_positive_limit_raises(self):
        for limit in (0, -1):
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    repo.get_announcements_without_enrichment(self.db_path, limit=limit)

    def test_count_announcements_without_enrichment(self):
        self.assertEqual(repo.count_announcements_without_enrichment(self.db_path), 1)

        url_2 = self.add_announcement(2)
        self.add_announcement(3)
        self.assertEqual(repo.count_announcements_without_enrichment(self.db_path), 3)

        self.save()
        self.save(make_enrichment(enrichment_status="unsupported"), url_2)
        self.assertEqual(repo.count_announcements_without_enrichment(self.db_path), 1)

    def test_pending_dict_has_announcement_fields_for_enrich_announcement(self):
        pending = repo.get_announcements_without_enrichment(self.db_path)

        self.assertEqual(pending, [make_announcement()])
        for name in ("id", "first_seen_at", "last_seen_at"):
            self.assertNotIn(name, pending[0])

    # --- get_announcement_with_enrichment ---

    def test_combined_contains_resource_type_from_announcements(self):
        self.save()

        combined = repo.get_announcement_with_enrichment(RESOURCE_URL, self.db_path)

        self.assertEqual(combined["resource_type"], "armeps_documents_page")
        self.assertNotIn("resource_type", self.get())  # именно поэтому нужен join

    def test_combined_contains_announcement_and_enrichment_fields(self):
        self.save(make_enrichment(document_url="https://example.test/doc.zip"))

        combined = repo.get_announcement_with_enrichment(RESOURCE_URL, self.db_path)

        expected_names = set(repo.ANNOUNCEMENT_FIELDS) | set(repo.SCALAR_FIELDS) | {
            "consistency", "cpv_codes", "documents",
        }
        self.assertEqual(set(combined), expected_names)
        for name in repo.ANNOUNCEMENT_FIELDS:
            self.assertEqual(combined[name], make_announcement()[name])
        self.assertEqual(combined["enrichment_status"], "success")
        self.assertEqual(combined["procedure_code"], "TEST-001")
        self.assertEqual(combined["document_url"], "https://example.test/doc.zip")
        self.assertEqual(combined["consistency"], {"published_at_match": True, "deadline_at_match": True})

    def test_combined_contains_cpv_codes_and_documents(self):
        self.save(make_enrichment(cpv_codes=[CPV_A, CPV_B], documents=[DOC_1, DOC_2]))

        combined = repo.get_announcement_with_enrichment(RESOURCE_URL, self.db_path)

        self.assertEqual(combined["cpv_codes"], [CPV_A, CPV_B])
        self.assertEqual(combined["documents"], [DOC_1, DOC_2])

    def test_combined_without_enrichment_is_none(self):
        self.assertIsNone(repo.get_announcement_with_enrichment(RESOURCE_URL, self.db_path))

    def test_combined_for_unknown_resource_is_none(self):
        self.assertIsNone(
            repo.get_announcement_with_enrichment("https://example.test/unknown", self.db_path)
        )

    def test_combined_reflects_updated_announcement_type(self):
        self.save()
        announcement_repository.save_announcement(
            make_announcement(resource_type="eauction_tender_page"), self.db_path
        )

        combined = repo.get_announcement_with_enrichment(RESOURCE_URL, self.db_path)

        self.assertEqual(combined["resource_type"], "eauction_tender_page")

    # --- enrichment_processing_state ---

    def state(self, url=RESOURCE_URL):
        return repo.get_enrichment_processing_state(url, self.db_path)

    def save_state(self, url, status, **kwargs):
        repo.save_enrichment_processing_state(url, status, db_path=self.db_path, **kwargs)

    def test_init_db_creates_processing_state_table(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            columns = [
                (row[1], row[2], row[3], row[5])
                for row in conn.execute("PRAGMA table_info(enrichment_processing_state)")
            ]
            foreign_keys = conn.execute(
                "PRAGMA foreign_key_list(enrichment_processing_state)"
            ).fetchall()
            sql = conn.execute(
                "SELECT sql FROM sqlite_master WHERE name = 'enrichment_processing_state'"
            ).fetchone()[0]

        self.assertEqual(columns, [
            ("resource_url", "TEXT", 0, 1),
            ("status", "TEXT", 1, 0),
            ("last_attempt_at", "TEXT", 1, 0),
            ("error_type", "TEXT", 0, 0),
            ("error_message", "TEXT", 0, 0),
        ])
        self.assertEqual([(fk[2], fk[3], fk[4]) for fk in foreign_keys], [
            ("announcements", "resource_url", "resource_url"),
        ])
        self.assertNotIn("CHECK", sql.upper())

    def test_init_db_keeps_existing_states(self):
        self.save_state(RESOURCE_URL, "failed", error_type="E", error_message="m")

        repo.init_db(self.db_path)

        self.assertEqual(self.state()["status"], "failed")

    def test_init_db_adds_state_table_to_legacy_db(self):
        legacy_path = Path(self._tmp.name) / "legacy.db"
        announcement_repository.init_db(legacy_path)
        with closing(sqlite3.connect(legacy_path)) as conn, conn:
            conn.execute(repo.CREATE_ENRICHMENT_TABLE)  # БД до появления таблицы состояния

        repo.init_db(legacy_path)

        self.assertIsNone(repo.get_enrichment_processing_state(RESOURCE_URL, legacy_path))

    def test_success_state_round_trip(self):
        self.save_state(RESOURCE_URL, "success")

        state = self.state()

        self.assertEqual(state["resource_url"], RESOURCE_URL)
        self.assertEqual(state["status"], "success")
        self.assertIsNone(state["error_type"])
        self.assertIsNone(state["error_message"])
        self.assertIsNotNone(datetime.fromisoformat(state["last_attempt_at"]).tzinfo)

    def test_failed_state_round_trip(self):
        self.save_state(RESOURCE_URL, "failed", error_type="RuntimeError", error_message="сбой")

        state = self.state()

        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["error_type"], "RuntimeError")
        self.assertEqual(state["error_message"], "сбой")

    def test_success_after_failed_clears_error_fields(self):
        self.save_state(RESOURCE_URL, "failed", error_type="RuntimeError", error_message="сбой")

        self.save_state(RESOURCE_URL, "success")

        state = self.state()
        self.assertEqual(state["status"], "success")
        self.assertIsNone(state["error_type"])
        self.assertIsNone(state["error_message"])
        self.assertEqual(self.count_rows("enrichment_processing_state"), 1)

    def test_failed_after_success_keeps_single_row(self):
        self.save_state(RESOURCE_URL, "success")

        self.save_state(RESOURCE_URL, "failed", error_type="E", error_message="m")

        self.assertEqual(self.state()["status"], "failed")
        self.assertEqual(self.count_rows("enrichment_processing_state"), 1)

    def test_last_attempt_at_is_refreshed(self):
        self.save_state(RESOURCE_URL, "failed", error_type="E", error_message="m")
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute("UPDATE enrichment_processing_state SET last_attempt_at = '2000-01-01T00:00:00+00:00'")

        self.save_state(RESOURCE_URL, "success")

        self.assertGreater(self.state()["last_attempt_at"], "2000-01-01T00:00:00+00:00")

    def test_saving_state_keeps_existing_enrichment_untouched(self):
        self.save(make_enrichment(cpv_codes=[CPV_A], documents=[DOC_1]))
        before = self.get()

        self.save_state(RESOURCE_URL, "failed", error_type="E", error_message="m")

        self.assertEqual(self.get(), before)

    def test_state_for_unknown_announcement_raises(self):
        with self.assertRaises(ValueError):
            self.save_state("https://example.test/unknown", "success")
        self.assertEqual(self.count_rows("enrichment_processing_state"), 0)

    def test_blank_resource_url_or_status_raises(self):
        for url, status in ((None, "success"), ("", "success"), ("  ", "success"),
                            (RESOURCE_URL, None), (RESOURCE_URL, ""), (RESOURCE_URL, "  ")):
            with self.subTest(url=url, status=status):
                with self.assertRaises(ValueError):
                    self.save_state(url, status)

    def test_missing_state_is_none(self):
        self.assertIsNone(self.state())
        self.assertIsNone(self.state("https://example.test/unknown"))

    # --- get_enrichment_processing_candidates ---

    def candidate_urls(self, **kwargs) -> list[str]:
        candidates = repo.get_enrichment_processing_candidates(self.db_path, **kwargs)
        return [item["resource_url"] for item in candidates]

    def url(self, number) -> str:
        return f"https://example.test/resource/{number}"

    def add_enriched(self, number, first_seen_at=None) -> str:
        """Объявление с enrichment (как после успешной обработки)."""
        url = self.add_announcement(number, first_seen_at)
        self.save(make_enrichment(), url)
        return url

    def test_candidate_without_enrichment_and_state(self):
        self.assertEqual(self.candidate_urls(), [RESOURCE_URL])

    def test_candidate_without_enrichment_regardless_of_state(self):
        for status in ("failed", "success"):
            with self.subTest(status=status):
                url = self.add_announcement(status)
                self.save_state(url, status)

                self.assertIn(url, self.candidate_urls())

    def test_enrichment_with_success_state_is_not_candidate(self):
        self.save()
        self.save_state(RESOURCE_URL, "success")

        self.assertEqual(self.candidate_urls(), [])

    def test_enrichment_without_state_is_not_candidate(self):
        # Строки enrichment, сохранённые до появления таблицы состояния.
        self.save()

        self.assertEqual(self.candidate_urls(), [])

    def test_enrichment_with_failed_state_is_candidate(self):
        # updated-объявление: старый enrichment остался, refresh упал.
        self.save()
        self.save_state(RESOURCE_URL, "failed", error_type="E", error_message="m")

        self.assertEqual(self.candidate_urls(), [RESOURCE_URL])

    def test_candidate_disappears_after_success_state(self):
        self.save()
        self.save_state(RESOURCE_URL, "failed", error_type="E", error_message="m")
        self.assertEqual(self.candidate_urls(), [RESOURCE_URL])

        self.save_state(RESOURCE_URL, "success")

        self.assertEqual(self.candidate_urls(), [])

    def test_any_enrichment_status_with_success_state_is_not_candidate(self):
        for status in ("success", "partial", "not_required", "unsupported"):
            with self.subTest(status=status):
                url = self.add_announcement(status)
                self.save(make_enrichment(enrichment_status=status), url)
                self.save_state(url, "success")

                self.assertNotIn(url, self.candidate_urls())

    def test_missing_enrichment_goes_before_failed_refresh_even_if_newer(self):
        # RESOURCE_URL — самый старый и failed; новые без enrichment всё равно первые.
        self.save()
        self.save_state(RESOURCE_URL, "failed", error_type="E", error_message="m")
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute(
                "UPDATE announcements SET first_seen_at = '2026-01-01T00:00:00+00:00' "
                "WHERE resource_url = ?",
                (RESOURCE_URL,),
            )
        url_b = self.add_announcement("b", "2026-09-02T00:00:00+00:00")
        url_c = self.add_announcement("c", "2026-09-01T00:00:00+00:00")

        self.assertEqual(self.candidate_urls(), [url_c, url_b, RESOURCE_URL])

    def test_order_inside_groups_is_first_seen_at_then_id(self):
        self.save()
        self.save_state(RESOURCE_URL, "failed", error_type="E", error_message="m")
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute(
                "UPDATE announcements SET first_seen_at = '2026-09-03T00:00:00+00:00' "
                "WHERE resource_url = ?",
                (RESOURCE_URL,),
            )
        missing_b = self.add_announcement("b", "2026-09-01T00:00:00+00:00")
        missing_c = self.add_announcement("c", "2026-09-01T00:00:00+00:00")  # тот же день: по id
        failed_d = self.add_enriched("d", "2026-09-02T00:00:00+00:00")
        self.save_state(failed_d, "failed", error_type="E", error_message="m")
        failed_e = self.add_enriched("e", "2026-09-02T00:00:00+00:00")
        self.save_state(failed_e, "failed", error_type="E", error_message="m")

        self.assertEqual(
            self.candidate_urls(), [missing_b, missing_c, failed_d, failed_e, RESOURCE_URL]
        )

    def test_old_failed_refresh_does_not_block_new_unenriched_with_limit(self):
        failed = []
        for number in (2, 3, 4):
            url = self.add_enriched(number, f"2026-08-0{number}T00:00:00+00:00")
            self.save_state(url, "failed", error_type="E", error_message="m")
            failed.append(url)
        new_a = self.add_announcement("new_a", "2026-09-01T00:00:00+00:00")
        new_b = self.add_announcement("new_b", "2026-09-02T00:00:00+00:00")
        # RESOURCE_URL тоже без enrichment, но появился раньше остальных новых.
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute(
                "UPDATE announcements SET first_seen_at = '2026-08-31T00:00:00+00:00' "
                "WHERE resource_url = ?",
                (RESOURCE_URL,),
            )

        self.assertEqual(self.candidate_urls(limit=3), [RESOURCE_URL, new_a, new_b])
        self.assertEqual(self.candidate_urls(limit=4), [RESOURCE_URL, new_a, new_b, failed[0]])

    def test_candidates_limit(self):
        for number in (2, 3):
            self.add_announcement(number)

        self.assertEqual(len(self.candidate_urls(limit=1)), 1)
        self.assertEqual(len(self.candidate_urls(limit=2)), 2)
        self.assertEqual(len(self.candidate_urls(limit=10)), 3)
        self.assertEqual(len(self.candidate_urls(limit=None)), 3)
        self.assertEqual(len(self.candidate_urls()), 3)

    def test_candidates_invalid_limit_raises(self):
        for limit in (0, -1, True, False, 1.5, "3", "1; DROP TABLE announcements"):
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    repo.get_enrichment_processing_candidates(self.db_path, limit=limit)

        self.assertEqual(self.count_rows("announcements"), 1)

    def test_candidate_dict_has_announcement_fields_for_enrich_announcement(self):
        candidates = repo.get_enrichment_processing_candidates(self.db_path)

        self.assertEqual(candidates, [make_announcement()])
        for name in ("id", "first_seen_at", "last_seen_at"):
            self.assertNotIn(name, candidates[0])

    def test_count_processing_candidates_matches_query_without_limit(self):
        self.assertEqual(repo.count_enrichment_processing_candidates(self.db_path), 1)

        url_2 = self.add_enriched(2)
        url_3 = self.add_enriched(3)
        url_4 = self.add_announcement(4)
        self.save_state(url_2, "success")
        self.save_state(url_3, "failed", error_type="E", error_message="m")
        self.save_state(url_4, "success")  # enrichment нет — всё равно кандидат

        self.assertEqual(repo.count_enrichment_processing_candidates(self.db_path), 3)
        self.assertEqual(len(self.candidate_urls()), 3)

    def test_legacy_pending_function_is_unchanged_by_failed_state(self):
        self.save()
        self.save_state(RESOURCE_URL, "failed", error_type="E", error_message="m")

        # Старая функция по-прежнему про «нет строки enrichment» и failed refresh не видит.
        self.assertEqual(self.pending_urls(), [])
        self.assertEqual(repo.count_announcements_without_enrichment(self.db_path), 0)


if __name__ == "__main__":
    unittest.main()
