"""
Тесты document_repository на временной SQLite БД.

Запуск из корня проекта:
    python -m unittest tests.test_document_repository -v
"""

import logging
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

from src.database import announcement_repository, document_repository as repo

RESOURCE_URL = "https://example.test/resource/1"
SOURCE_KIND = "armeps_document"
SOURCE_REF = "doc-101"
SHA_1 = "a" * 64
SHA_2 = "b" * 64


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


def make_download_result(**overrides) -> dict:
    result = {
        "source_url": "https://example.test/files/hraver.docx",
        "saved_path": "data/documents/test/hraver.docx",
        "filename": "hraver.docx",
        "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "size_bytes": 1234,
        "sha256": SHA_1,
    }
    result.update(overrides)
    return result


def make_docx_result(text="Текст документа", **overrides) -> dict:
    result = {
        "file_type": "docx",
        "text": text,
        "char_count": len(text),
        "paragraph_count": 2,
        "table_count": 1,
    }
    result.update(overrides)
    return result


def make_zip_result(documents=(), skipped_members=(), failures=()) -> dict:
    return {
        "file_type": "zip",
        "member_count": len(documents) + len(skipped_members) + len(failures),
        "docx_count": len(documents),
        "documents": list(documents),
        "skipped_members": list(skipped_members),
        "failures": list(failures),
    }


class DocumentRepositoryTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "test.db"

        announcement_repository.init_db(self.db_path)
        announcement_repository.save_announcement(make_announcement(), self.db_path)
        repo.init_db(self.db_path)

        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)

    def save_download(self, download_result=None, **kwargs):
        args = {
            "resource_url": RESOURCE_URL,
            "source_kind": SOURCE_KIND,
            "source_ref": SOURCE_REF,
            "download_result": download_result if download_result is not None else make_download_result(),
        }
        args.update(kwargs)
        return repo.save_download(db_path=self.db_path, **args)

    def new_download_id(self, **kwargs):
        return self.save_download(**kwargs)["download_id"]

    def extractions(self, download_id):
        return repo.get_extractions_for_download(download_id, self.db_path)

    def count_rows(self, table):
        with closing(sqlite3.connect(self.db_path)) as conn:
            return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def announcements_snapshot(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            schema = conn.execute("PRAGMA table_info(announcements)").fetchall()
            rows = conn.execute("SELECT * FROM announcements ORDER BY id").fetchall()
        return schema, rows

    # --- схема ---

    def test_init_db_creates_two_tables(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            names = {
                row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
        self.assertIn("document_downloads", names)
        self.assertIn("document_extractions", names)

    def test_announcements_table_and_row_unchanged(self):
        before = self.announcements_snapshot()

        repo.init_db(self.db_path)
        download_id = self.new_download_id()
        repo.save_docx_extraction(download_id, make_docx_result(), self.db_path)

        self.assertEqual(self.announcements_snapshot(), before)
        self.assertEqual(announcement_repository.count_announcements(self.db_path), 1)

    def test_foreign_keys_enabled_on_connection(self):
        with repo._connect(self.db_path) as conn:
            self.assertEqual(conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)

    # --- downloads ---

    def test_first_download_is_new(self):
        result = self.save_download()

        self.assertEqual(result["status"], "new")
        self.assertIsInstance(result["download_id"], int)

    def test_download_fields_round_trip(self):
        self.save_download(document_id="101")
        [row] = repo.get_downloads_for_resource(RESOURCE_URL, self.db_path)

        expected = make_download_result()
        for name, value in expected.items():
            self.assertEqual(row[name], value, name)
        self.assertEqual(row["resource_url"], RESOURCE_URL)
        self.assertEqual(row["source_kind"], SOURCE_KIND)
        self.assertEqual(row["source_ref"], SOURCE_REF)
        self.assertEqual(row["document_id"], "101")
        self.assertTrue(row["downloaded_at"].endswith("+00:00"))

    def test_same_sha_is_existing_with_same_id(self):
        first = self.save_download()
        second = self.save_download()

        self.assertEqual(second, {"status": "existing", "download_id": first["download_id"]})
        self.assertEqual(repo.count_downloads(self.db_path), 1)

    def test_same_source_ref_new_sha_creates_second_version(self):
        first = self.save_download()
        second = self.save_download(make_download_result(sha256=SHA_2, size_bytes=99))

        self.assertEqual(second["status"], "new")
        self.assertNotEqual(second["download_id"], first["download_id"])
        self.assertEqual(repo.count_downloads(self.db_path), 2)

    def test_different_source_ref_creates_separate_download(self):
        first = self.save_download()
        second = self.save_download(source_ref="doc-102")

        self.assertEqual(second["status"], "new")
        self.assertNotEqual(second["download_id"], first["download_id"])
        self.assertEqual(repo.count_downloads(self.db_path), 2)

    def test_different_source_kind_creates_separate_download(self):
        self.save_download()
        second = self.save_download(source_kind="direct_file")

        self.assertEqual(second["status"], "new")
        self.assertEqual(repo.count_downloads(self.db_path), 2)

    def test_unknown_resource_url_raises(self):
        with self.assertRaises(ValueError):
            self.save_download(resource_url="https://example.test/unknown")
        self.assertEqual(repo.count_downloads(self.db_path), 0)

    def test_required_download_fields_checked(self):
        for name in ("source_url", "saved_path", "filename", "sha256", "size_bytes"):
            with self.subTest(field=name):
                broken = make_download_result()
                del broken[name]
                with self.assertRaises(ValueError):
                    self.save_download(broken)

        for name in ("source_url", "saved_path", "filename", "sha256"):
            with self.subTest(blank=name):
                with self.assertRaises(ValueError):
                    self.save_download(make_download_result(**{name: "  "}))

        for name in ("resource_url", "source_kind", "source_ref"):
            for value in ("", "  ", None):
                with self.subTest(argument=name, value=value):
                    with self.assertRaises(ValueError):
                        self.save_download(**{name: value})

        self.assertEqual(repo.count_downloads(self.db_path), 0)

    def test_zero_size_and_missing_optional_fields_allowed(self):
        result = make_download_result(size_bytes=0, content_type=None)
        self.assertEqual(self.save_download(result)["status"], "new")

    def test_get_latest_download_returns_newest_version(self):
        first = self.save_download()
        second = self.save_download(make_download_result(sha256=SHA_2))

        latest = repo.get_latest_download(RESOURCE_URL, SOURCE_KIND, SOURCE_REF, self.db_path)

        self.assertEqual(latest["id"], second["download_id"])
        self.assertEqual(latest["sha256"], SHA_2)
        self.assertNotEqual(latest["id"], first["download_id"])

    def test_get_latest_download_unknown_returns_none(self):
        self.assertIsNone(
            repo.get_latest_download(RESOURCE_URL, SOURCE_KIND, "unknown", self.db_path)
        )

    def test_get_downloads_for_resource_returns_all_versions_by_id(self):
        first = self.save_download()
        second = self.save_download(make_download_result(sha256=SHA_2))
        third = self.save_download(source_ref="doc-102")

        rows = repo.get_downloads_for_resource(RESOURCE_URL, self.db_path)

        self.assertEqual(
            [row["id"] for row in rows],
            [first["download_id"], second["download_id"], third["download_id"]],
        )
        self.assertEqual(repo.get_downloads_for_resource("https://example.test/none", self.db_path), [])

    def test_old_version_kept_after_new_version(self):
        first = self.save_download()
        self.save_download(make_download_result(sha256=SHA_2))

        shas = [row["sha256"] for row in repo.get_downloads_for_resource(RESOURCE_URL, self.db_path)]
        self.assertEqual(shas, [SHA_1, SHA_2])
        self.assertEqual(repo.get_downloads_for_resource(RESOURCE_URL, self.db_path)[0]["id"], first["download_id"])

    # --- одиночный extraction ---

    def test_standalone_docx_saved_with_empty_member_name(self):
        download_id = self.new_download_id()

        counts = repo.save_docx_extraction(download_id, make_docx_result(), self.db_path)

        self.assertEqual(counts, {"new_count": 1, "updated_count": 0, "existing_count": 0})
        [row] = self.extractions(download_id)
        self.assertEqual(row["member_name"], "")
        self.assertEqual(row["extraction_status"], "success")
        self.assertEqual(row["file_type"], "docx")
        self.assertEqual(row["text"], "Текст документа")
        self.assertEqual(row["char_count"], len("Текст документа"))
        self.assertEqual(row["paragraph_count"], 2)
        self.assertEqual(row["table_count"], 1)
        self.assertIsNone(row["error_type"])
        self.assertIsNone(row["error_message"])
        self.assertTrue(row["extracted_at"].endswith("+00:00"))

    def test_empty_text_allowed(self):
        download_id = self.new_download_id()

        repo.save_docx_extraction(download_id, make_docx_result(text=""), self.db_path)

        [row] = self.extractions(download_id)
        self.assertEqual(row["text"], "")
        self.assertEqual(row["char_count"], 0)
        self.assertEqual(row["extraction_status"], "success")

    def test_save_extraction_missing_member_name_uses_empty_string(self):
        download_id = self.new_download_id()
        extraction = {"file_type": "docx", "extraction_status": "success", "text": "x"}

        self.assertEqual(repo.save_extraction(download_id, extraction, self.db_path), "new")
        self.assertEqual(self.extractions(download_id)[0]["member_name"], "")

        extraction["member_name"] = None
        self.assertEqual(repo.save_extraction(download_id, extraction, self.db_path), "existing")
        self.assertEqual(self.count_rows("document_extractions"), 1)

    def test_same_extraction_is_existing(self):
        download_id = self.new_download_id()
        repo.save_docx_extraction(download_id, make_docx_result(), self.db_path)

        counts = repo.save_docx_extraction(download_id, make_docx_result(), self.db_path)

        self.assertEqual(counts, {"new_count": 0, "updated_count": 0, "existing_count": 1})
        self.assertEqual(repo.count_extractions(self.db_path), 1)

    def test_changed_text_is_updated_in_same_row(self):
        download_id = self.new_download_id()
        repo.save_docx_extraction(download_id, make_docx_result(), self.db_path)
        [before] = self.extractions(download_id)

        counts = repo.save_docx_extraction(download_id, make_docx_result(text="Другой"), self.db_path)

        self.assertEqual(counts, {"new_count": 0, "updated_count": 1, "existing_count": 0})
        [after] = self.extractions(download_id)
        self.assertEqual(after["id"], before["id"])
        self.assertEqual(after["text"], "Другой")
        self.assertEqual(repo.count_extractions(self.db_path), 1)

    def test_save_extraction_returns_status_strings(self):
        download_id = self.new_download_id()
        extraction = {"extraction_status": "success", "text": "a"}

        self.assertEqual(repo.save_extraction(download_id, extraction, self.db_path), "new")
        self.assertEqual(repo.save_extraction(download_id, extraction, self.db_path), "existing")
        self.assertEqual(
            repo.save_extraction(download_id, {**extraction, "text": "b"}, self.db_path), "updated"
        )

    def test_unknown_download_id_raises(self):
        with self.assertRaises(ValueError):
            repo.save_extraction(9999, {"extraction_status": "success"}, self.db_path)
        with self.assertRaises(ValueError):
            repo.save_docx_extraction(9999, make_docx_result(), self.db_path)
        with self.assertRaises(ValueError):
            repo.save_zip_extraction(9999, make_zip_result(skipped_members=["a.pdf"]), self.db_path)
        self.assertEqual(repo.count_extractions(self.db_path), 0)

    def test_missing_extraction_status_raises(self):
        download_id = self.new_download_id()

        with self.assertRaises(ValueError):
            repo.save_extraction(download_id, {"text": "x"}, self.db_path)
        with self.assertRaises(ValueError):
            repo.save_extraction(download_id, {"extraction_status": " "}, self.db_path)

    # --- ZIP ---

    def test_zip_with_two_docx_creates_two_rows(self):
        download_id = self.new_download_id()
        result = make_zip_result(documents=[
            {"member_name": "hraver.docx", **make_docx_result("Первый")},
            {"member_name": "hraver_ru.docx", **make_docx_result("Второй")},
        ])

        counts = repo.save_zip_extraction(download_id, result, self.db_path)

        self.assertEqual(counts, {"new_count": 2, "updated_count": 0, "existing_count": 0})
        rows = self.extractions(download_id)
        self.assertEqual([r["member_name"] for r in rows], ["hraver.docx", "hraver_ru.docx"])
        self.assertEqual([r["text"] for r in rows], ["Первый", "Второй"])
        self.assertTrue(all(r["extraction_status"] == "success" for r in rows))

    def test_zip_skipped_member_saved_as_skipped(self):
        download_id = self.new_download_id()

        counts = repo.save_zip_extraction(
            download_id, make_zip_result(skipped_members=["scan.pdf"]), self.db_path
        )

        self.assertEqual(counts["new_count"], 1)
        [row] = self.extractions(download_id)
        self.assertEqual(row["member_name"], "scan.pdf")
        self.assertEqual(row["extraction_status"], "skipped")
        self.assertIsNone(row["text"])
        self.assertIsNone(row["error_type"])
        self.assertIsNone(row["error_message"])

    def test_zip_failure_saved_as_failed_with_error(self):
        download_id = self.new_download_id()
        failure = {
            "member_name": "broken.docx",
            "error_type": "BadZipFile",
            "error_message": "File is not a zip file",
        }

        repo.save_zip_extraction(download_id, make_zip_result(failures=[failure]), self.db_path)

        [row] = self.extractions(download_id)
        self.assertEqual(row["member_name"], "broken.docx")
        self.assertEqual(row["extraction_status"], "failed")
        self.assertEqual(row["error_type"], "BadZipFile")
        self.assertEqual(row["error_message"], "File is not a zip file")
        self.assertIsNone(row["text"])

    def make_mixed_zip_result(self):
        return make_zip_result(
            documents=[{"member_name": "hraver.docx", **make_docx_result()}],
            skipped_members=["scan.pdf"],
            failures=[{"member_name": "broken.docx", "error_type": "KeyError", "error_message": "x"}],
        )

    def test_zip_save_is_idempotent(self):
        download_id = self.new_download_id()
        result = self.make_mixed_zip_result()

        first = repo.save_zip_extraction(download_id, result, self.db_path)
        second = repo.save_zip_extraction(download_id, result, self.db_path)

        self.assertEqual(first, {"new_count": 3, "updated_count": 0, "existing_count": 0})
        self.assertEqual(second, {"new_count": 0, "updated_count": 0, "existing_count": 3})
        self.assertEqual(repo.count_extractions(self.db_path), 3)

    def test_empty_zip_result_saves_nothing(self):
        download_id = self.new_download_id()

        counts = repo.save_zip_extraction(download_id, make_zip_result(), self.db_path)

        self.assertEqual(counts, {"new_count": 0, "updated_count": 0, "existing_count": 0})
        self.assertEqual(repo.count_extractions(self.db_path), 0)

    def test_zip_sql_error_rolls_back_all_members(self):
        download_id = self.new_download_id()
        real_save = repo._save_extraction
        calls = []

        def fail_on_third(conn, dl_id, extraction):
            calls.append(extraction["member_name"])
            if len(calls) == 3:
                raise sqlite3.OperationalError("simulated failure")
            return real_save(conn, dl_id, extraction)

        with mock.patch.object(repo, "_save_extraction", side_effect=fail_on_third):
            with self.assertRaises(sqlite3.Error):
                repo.save_zip_extraction(download_id, self.make_mixed_zip_result(), self.db_path)

        self.assertEqual(len(calls), 3)
        self.assertEqual(repo.count_extractions(self.db_path), 0)

    # --- счётчики и foreign keys ---

    def test_count_downloads(self):
        self.assertEqual(repo.count_downloads(self.db_path), 0)

        self.save_download()
        self.save_download()
        self.save_download(make_download_result(sha256=SHA_2))
        self.save_download(source_ref="doc-102")

        self.assertEqual(repo.count_downloads(self.db_path), 3)

    def test_count_extractions(self):
        first_id = self.new_download_id()
        second_id = self.new_download_id(source_ref="doc-102")
        self.assertEqual(repo.count_extractions(self.db_path), 0)

        repo.save_docx_extraction(first_id, make_docx_result(), self.db_path)
        repo.save_docx_extraction(first_id, make_docx_result(), self.db_path)
        repo.save_zip_extraction(second_id, self.make_mixed_zip_result(), self.db_path)

        self.assertEqual(repo.count_extractions(self.db_path), 4)

    def test_foreign_keys_enforced(self):
        with repo._connect(self.db_path) as conn:
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO document_extractions (download_id, extraction_status, extracted_at) "
                    "VALUES (9999, 'success', 'now')"
                )
        with repo._connect(self.db_path) as conn:
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO document_downloads (resource_url, source_kind, source_ref, "
                    "source_url, filename, saved_path, size_bytes, sha256, downloaded_at) "
                    "VALUES ('https://example.test/unknown', 'k', 'r', 'u', 'f', 'p', 1, 's', 'now')"
                )

    def test_deleting_download_cascades_to_extractions(self):
        download_id = self.new_download_id()
        repo.save_docx_extraction(download_id, make_docx_result(), self.db_path)

        with repo._connect(self.db_path) as conn:
            conn.execute("DELETE FROM document_downloads WHERE id = ?", (download_id,))

        self.assertEqual(repo.count_extractions(self.db_path), 0)


if __name__ == "__main__":
    unittest.main()
