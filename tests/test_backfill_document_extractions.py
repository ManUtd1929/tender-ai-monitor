"""
Тесты локального backfill extraction (src/backfill_document_extractions.py) и связанных
функций document_repository. Всё — во временной директории: SQLite БД и файлы создаются на
лету, сеть и production data/ не используются.

Запуск из корня проекта:
    python -m unittest tests.test_backfill_document_extractions -v
"""

import contextlib
import hashlib
import io
import logging
import socket
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

from src import backfill_document_extractions as backfill
from src.database import announcement_repository, document_repository as repo
from tests.test_document_extractor import make_docx_bytes, make_xlsx_bytes, make_zip_bytes
from tests.test_document_repository import make_announcement

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESOURCE_URL = "https://example.test/resource/1"
SOURCE_KIND = "eauction_document"
SOURCE_REF = "https://example.test/files/tender_1.zip"

XLSX_SHEETS = {"Lot": [["Item", "Qty"], ["Bolt", 10]]}


def docx_bytes(text="Hraver text"):
    return make_docx_bytes([text])


def xlsx_bytes():
    return make_xlsx_bytes(XLSX_SHEETS)


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path) -> str:
    return sha256_of(Path(path).read_bytes())


@contextlib.contextmanager
def forbid_network():
    """Дополнительно к глобальной защите tests/: любое обращение к сети — ошибка теста."""
    with mock.patch.object(socket.socket, "connect") as connect, \
            mock.patch.object(socket, "getaddrinfo") as getaddrinfo, \
            mock.patch.object(socket, "create_connection") as create_connection:
        yield
    connect.assert_not_called()
    getaddrinfo.assert_not_called()
    create_connection.assert_not_called()


class BackfillTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.db_path = self.root / "test.db"
        self.files_dir = self.root / "documents"
        self.files_dir.mkdir()

        announcement_repository.init_db(self.db_path)
        announcement_repository.save_announcement(make_announcement(), self.db_path)
        repo.init_db(self.db_path)

        self._file_counter = 0
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)

    # --- фикстуры ---

    def add_announcement(self, resource_url):
        announcement_repository.save_announcement(
            make_announcement(resource_url=resource_url), self.db_path
        )

    def write_file(self, filename, data) -> Path:
        self._file_counter += 1
        directory = self.files_dir / str(self._file_counter)
        directory.mkdir()
        path = directory / filename
        path.write_bytes(data)
        return path

    def add_download(
        self, filename="tender_1.zip", data=None, resource_url=RESOURCE_URL,
        source_kind=SOURCE_KIND, source_ref=SOURCE_REF, saved_path=None, sha256=None,
        create_file=True,
    ) -> int:
        """Скачанный документ: файл на диске + строка document_downloads. Возвращает id."""
        data = data if data is not None else make_zip_bytes([("hraver.docx", docx_bytes())])
        path = saved_path if saved_path is not None else self.files_dir / "missing" / filename
        if create_file:
            path = self.write_file(filename, data)
        result = {
            "source_url": "https://example.test/files/" + filename,
            "saved_path": str(path),
            "filename": filename,
            "content_type": "application/octet-stream",
            "size_bytes": len(data),
            "sha256": sha256 if sha256 is not None else sha256_of(data),
        }
        return repo.save_download(
            resource_url, source_kind, source_ref, result, db_path=self.db_path
        )["download_id"]

    def add_old_extraction(self, download_id, member_name, status, **extra):
        repo.save_extraction(
            download_id,
            {"member_name": member_name, "extraction_status": status, **extra},
            self.db_path,
        )

    def add_success_extraction(self, download_id, member_name=""):
        self.add_old_extraction(
            download_id, member_name, "success", file_type="docx", text="x", char_count=1,
            paragraph_count=1, table_count=0,
        )

    def extractions(self, download_id):
        return repo.get_extractions_for_download(download_id, self.db_path)

    def members(self, download_id):
        return {row["member_name"]: row["extraction_status"] for row in self.extractions(download_id)}

    def state(self, resource_url=RESOURCE_URL):
        return repo.get_processing_state(resource_url, self.db_path)

    def candidate(self, download_id):
        for candidate in repo.get_document_extraction_backfill_candidates(self.db_path):
            if candidate["id"] == download_id:
                return candidate
        self.fail(f"download {download_id} не является кандидатом")

    def candidate_ids(self, **kwargs):
        return [
            candidate["id"] for candidate in
            repo.get_document_extraction_backfill_candidates(self.db_path, **kwargs)
        ]

    def run_apply(self, **kwargs):
        return backfill.run_backfill(db_path=self.db_path, apply=True, **kwargs)

    def db_snapshot(self):
        """Всё содержимое БД: любое изменение (в том числе DDL) меняет snapshot."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            return "\n".join(conn.iterdump())

    def files_snapshot(self):
        return {
            str(path): file_sha256(path) for path in sorted(self.files_dir.rglob("*"))
            if path.is_file()
        }


# --------------------------------------------------------------------------
# Candidates
# --------------------------------------------------------------------------

class CandidateTests(BackfillTestCase):
    def test_download_without_extraction_is_candidate(self):
        download_id = self.add_download()
        candidate = self.candidate(download_id)
        self.assertEqual(candidate["reasons"], [repo.BACKFILL_REASON_NO_EXTRACTION])
        self.assertEqual(candidate["extraction_count"], 0)

    def test_failed_extraction_is_candidate(self):
        download_id = self.add_download()
        self.add_success_extraction(download_id, "a.docx")
        self.add_old_extraction(
            download_id, "b.docx", "failed", error_type="BadZipFile", error_message="bad"
        )
        candidate = self.candidate(download_id)
        self.assertEqual(candidate["reasons"], [repo.BACKFILL_REASON_FAILED_EXTRACTION])
        self.assertEqual(candidate["failed_extraction_count"], 1)

    def test_skipped_xlsx_is_candidate(self):
        download_id = self.add_download()
        self.add_success_extraction(download_id, "a.docx")
        self.add_old_extraction(download_id, "lot_1.xlsx", "skipped")
        self.add_old_extraction(download_id, "lot_2.XLSX", "skipped")
        candidate = self.candidate(download_id)
        self.assertEqual(candidate["reasons"], [repo.BACKFILL_REASON_SKIPPED_SUPPORTED])
        self.assertEqual(candidate["skipped_supported_count"], 2)

    def test_skipped_nested_zip_is_candidate(self):
        download_id = self.add_download()
        self.add_success_extraction(download_id, "a.docx")
        self.add_old_extraction(download_id, "lot_1.zip", "skipped")
        self.assertIn(download_id, self.candidate_ids())

    def test_skipped_zip_at_max_depth_is_not_candidate(self):
        """Такой ZIP skipped и текущим extractor-ом: иначе download был бы кандидатом всегда."""
        download_id = self.add_download()
        self.add_success_extraction(download_id, "a.docx")
        self.add_old_extraction(download_id, "outer.zip!/deep.zip", "skipped")
        self.assertNotIn(download_id, self.candidate_ids())

    def test_skipped_unsupported_formats_are_not_candidates(self):
        for name in ("lot.rar", "scan.pdf", "old.doc", "data.xml", "sheet.xls"):
            with self.subTest(name=name):
                download_id = self.add_download(
                    filename=f"d_{name}.zip", source_ref=f"ref-{name}",
                )
                self.add_success_extraction(download_id, "a.docx")
                self.add_old_extraction(download_id, name, "skipped")
                self.assertNotIn(download_id, self.candidate_ids())

    def test_success_only_is_not_candidate(self):
        download_id = self.add_download()
        self.add_success_extraction(download_id, "a.docx")
        self.add_success_extraction(download_id, "b.docx")
        self.assertEqual(self.candidate_ids(), [])
        self.assertNotIn(download_id, self.candidate_ids())

    def test_several_reasons_are_all_reported(self):
        download_id = self.add_download()
        self.add_old_extraction(download_id, "a.docx", "failed", error_type="E")
        self.add_old_extraction(download_id, "b.xlsx", "skipped")
        self.assertEqual(
            self.candidate(download_id)["reasons"],
            [repo.BACKFILL_REASON_FAILED_EXTRACTION, repo.BACKFILL_REASON_SKIPPED_SUPPORTED],
        )

    def test_candidate_contains_download_metadata(self):
        data = make_zip_bytes([("hraver.docx", docx_bytes())])
        download_id = self.add_download(data=data)
        candidate = self.candidate(download_id)
        stored = repo.get_downloads_for_resource(RESOURCE_URL, self.db_path)[0]
        for name in (
            "id", "resource_url", "source_kind", "source_ref", "filename", "saved_path",
            "sha256", "downloaded_at",
        ):
            self.assertEqual(candidate[name], stored[name], name)
        self.assertEqual(candidate["sha256"], sha256_of(data))

    def test_order_is_download_id_ascending(self):
        ids = [
            self.add_download(filename=f"d{n}.zip", source_ref=f"ref-{n}") for n in range(4)
        ]
        self.assertEqual(self.candidate_ids(), ids)
        self.assertEqual(ids, sorted(ids))

    def test_limit(self):
        ids = [
            self.add_download(filename=f"d{n}.zip", source_ref=f"ref-{n}") for n in range(4)
        ]
        self.assertEqual(self.candidate_ids(limit=2), ids[:2])
        self.assertEqual(self.candidate_ids(limit=10), ids)
        self.assertEqual(self.candidate_ids(limit=None), ids)

    def test_limit_is_counted_after_filtering(self):
        done = self.add_download(filename="done.zip", source_ref="done")
        self.add_success_extraction(done, "a.docx")
        pending = self.add_download(filename="pending.zip", source_ref="pending")
        self.assertEqual(self.candidate_ids(limit=1), [pending])

    def test_invalid_limit_raises(self):
        self.add_download()
        for limit in (0, -1, True, False, "1", 1.5):
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    repo.get_document_extraction_backfill_candidates(self.db_path, limit=limit)

    def test_query_does_not_change_database(self):
        download_id = self.add_download()
        self.add_old_extraction(download_id, "a.xlsx", "skipped")
        before = self.db_snapshot()
        repo.get_document_extraction_backfill_candidates(self.db_path)
        self.assertEqual(self.db_snapshot(), before)

    def test_query_connection_is_read_only(self):
        with repo._connect_read_only(self.db_path) as conn:
            with self.assertRaises(sqlite3.OperationalError):
                conn.execute("DELETE FROM document_downloads")

    def test_query_does_not_create_missing_database(self):
        missing = self.root / "missing.db"
        with self.assertRaises(sqlite3.OperationalError):
            repo.get_document_extraction_backfill_candidates(missing)
        self.assertFalse(missing.exists())

    def test_query_works_with_path_containing_spaces_and_unicode(self):
        directory = self.root / "папка с пробелом"
        directory.mkdir()
        path = directory / "тест.db"
        announcement_repository.init_db(path)
        announcement_repository.save_announcement(make_announcement(), path)
        repo.init_db(path)
        self.assertEqual(repo.get_document_extraction_backfill_candidates(path), [])


# --------------------------------------------------------------------------
# Repository: extraction records and reconciliation
# --------------------------------------------------------------------------

class BuildExtractionRecordsTests(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)

    def test_docx_record(self):
        from src.parser.document_extractor import extract_docx
        result = extract_docx(docx_bytes("abc"))
        (record,) = repo.build_extraction_records("docx", result)
        self.assertEqual(record["member_name"], "")
        self.assertEqual(record["extraction_status"], "success")
        self.assertEqual(record["char_count"], 3)
        self.assertEqual(record["paragraph_count"], 1)
        self.assertEqual(record["table_count"], 0)
        self.assertIsNone(record["sheet_count"])

    def test_xlsx_record(self):
        from src.parser.document_extractor import extract_xlsx
        (record,) = repo.build_extraction_records("xlsx", extract_xlsx(xlsx_bytes()))
        self.assertEqual(record["member_name"], "")
        self.assertEqual(record["file_type"], "xlsx")
        self.assertEqual(
            (record["sheet_count"], record["row_count"], record["cell_count"]), (1, 2, 4)
        )
        self.assertIsNone(record["paragraph_count"])

    def test_zip_records_cover_success_skipped_and_failed(self):
        from src.parser.document_extractor import extract_supported_from_zip
        data = make_zip_bytes([
            ("a.docx", docx_bytes()),
            ("nested.zip", make_zip_bytes([("spec.xlsx", xlsx_bytes())])),
            ("notes.pdf", b"%PDF"),
            ("broken.docx", b"not a docx"),
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.zip"
            path.write_bytes(data)
            result = extract_supported_from_zip(path)
        records = repo.build_extraction_records("zip", result)
        statuses = {record["member_name"]: record["extraction_status"] for record in records}
        self.assertEqual(statuses, {
            "a.docx": "success",
            "nested.zip!/spec.xlsx": "success",
            "notes.pdf": "skipped",
            "broken.docx": "failed",
        })
        failed = next(record for record in records if record["member_name"] == "broken.docx")
        self.assertTrue(failed["error_type"])

    def test_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            repo.build_extraction_records("pdf", {})


class ReplaceExtractionsTests(BackfillTestCase):
    def record(self, member_name, status="success", **extra):
        return {"member_name": member_name, "extraction_status": status, **extra}

    def test_new_updated_existing_and_deleted_counts(self):
        download_id = self.add_download()
        self.add_old_extraction(download_id, "same.docx", "success", text="s", char_count=1)
        self.add_old_extraction(download_id, "change.xlsx", "skipped")
        self.add_old_extraction(download_id, "old.zip", "skipped")

        counts = repo.replace_extractions_for_download(
            download_id,
            [
                self.record("same.docx", text="s", char_count=1),
                self.record("change.xlsx", text="t", char_count=1),
                self.record("old.zip!/new.xlsx", text="n", char_count=1),
            ],
            self.db_path,
        )

        self.assertEqual(counts, {
            "new_count": 1, "updated_count": 1, "existing_count": 1, "deleted_count": 1,
        })
        self.assertEqual(
            self.members(download_id),
            {"same.docx": "success", "change.xlsx": "success", "old.zip!/new.xlsx": "success"},
        )

    def test_rows_of_other_downloads_are_not_touched(self):
        first = self.add_download(filename="a.zip", source_ref="a")
        second = self.add_download(filename="b.zip", source_ref="b")
        self.add_old_extraction(second, "keep.docx", "success")
        self.add_old_extraction(first, "drop.docx", "success")

        repo.replace_extractions_for_download(first, [self.record("new.docx")], self.db_path)

        self.assertEqual(self.members(second), {"keep.docx": "success"})
        self.assertEqual(self.members(first), {"new.docx": "success"})

    def test_standalone_row_with_empty_member_name_is_kept(self):
        download_id = self.add_download(filename="x.docx")
        self.add_old_extraction(download_id, "", "success", text="old", char_count=3)
        counts = repo.replace_extractions_for_download(
            download_id, [self.record("", text="new", char_count=3)], self.db_path
        )
        self.assertEqual((counts["updated_count"], counts["deleted_count"]), (1, 0))
        self.assertEqual(self.extractions(download_id)[0]["text"], "new")

    def test_empty_snapshot_removes_all_rows(self):
        download_id = self.add_download()
        self.add_old_extraction(download_id, "a.xlsx", "skipped")
        counts = repo.replace_extractions_for_download(download_id, [], self.db_path)
        self.assertEqual(counts["deleted_count"], 1)
        self.assertEqual(self.extractions(download_id), [])

    def test_unknown_download_raises_and_changes_nothing(self):
        download_id = self.add_download()
        self.add_old_extraction(download_id, "a.xlsx", "skipped")
        before = self.db_snapshot()
        with self.assertRaises(ValueError):
            repo.replace_extractions_for_download(9999, [self.record("a")], self.db_path)
        self.assertEqual(self.db_snapshot(), before)

    def test_record_without_status_rolls_back_everything(self):
        download_id = self.add_download()
        self.add_old_extraction(download_id, "old.xlsx", "skipped")
        before = self.db_snapshot()
        with self.assertRaises(ValueError):
            repo.replace_extractions_for_download(
                download_id,
                [self.record("new.docx"), {"member_name": "bad"}],
                self.db_path,
            )
        self.assertEqual(self.db_snapshot(), before)

    def test_sql_error_while_saving_rolls_back_everything(self):
        download_id = self.add_download()
        self.add_old_extraction(download_id, "old.xlsx", "skipped")
        before = self.db_snapshot()
        real_save = repo._save_extraction
        calls = []

        def fail_on_second(conn, dl_id, extraction):
            calls.append(extraction["member_name"])
            if len(calls) == 2:
                raise sqlite3.OperationalError("simulated failure")
            return real_save(conn, dl_id, extraction)

        with mock.patch.object(repo, "_save_extraction", side_effect=fail_on_second):
            with self.assertRaises(sqlite3.Error):
                repo.replace_extractions_for_download(
                    download_id, [self.record("a.docx"), self.record("b.docx")], self.db_path
                )
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.db_snapshot(), before)

    def test_sql_error_while_deleting_rolls_back_saved_rows(self):
        download_id = self.add_download()
        self.add_old_extraction(download_id, "old.zip", "skipped")
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute(
                "CREATE TRIGGER block_delete BEFORE DELETE ON document_extractions "
                "BEGIN SELECT RAISE(ABORT, 'delete blocked'); END"
            )
        before = self.db_snapshot()

        with self.assertRaises(sqlite3.Error):
            repo.replace_extractions_for_download(
                download_id, [self.record("old.zip!/spec.xlsx")], self.db_path
            )

        self.assertEqual(self.db_snapshot(), before)
        self.assertEqual(self.members(download_id), {"old.zip": "skipped"})

    def test_downloads_table_is_not_changed(self):
        download_id = self.add_download()
        before = repo.get_downloads_for_resource(RESOURCE_URL, self.db_path)
        repo.replace_extractions_for_download(download_id, [self.record("a.docx")], self.db_path)
        self.assertEqual(repo.get_downloads_for_resource(RESOURCE_URL, self.db_path), before)


# --------------------------------------------------------------------------
# Local file safety
# --------------------------------------------------------------------------

class LocalFileSafetyTests(BackfillTestCase):
    def test_missing_file_fails_with_file_not_found(self):
        download_id = self.add_download(create_file=False)
        summary = self.run_apply()
        (item,) = summary["items"]
        self.assertEqual(item["status"], "failed")
        self.assertEqual(item["error_type"], "FileNotFoundError")
        self.assertEqual(summary["failed_count"], 1)
        self.assertEqual(summary["failures"][0]["download_id"], download_id)
        self.assertEqual(summary["failures"][0]["error_type"], "FileNotFoundError")
        self.assertEqual(self.extractions(download_id), [])

    def test_hash_mismatch_is_not_extracted_and_hash_is_not_updated(self):
        data = make_zip_bytes([("hraver.docx", docx_bytes())])
        download_id = self.add_download(data=data, sha256="0" * 64)
        summary = self.run_apply()
        (item,) = summary["items"]
        self.assertEqual(item["status"], "failed")
        self.assertEqual(item["error_type"], "DocumentHashMismatchError")
        self.assertIn("0" * 64, item["error_message"])
        self.assertIn(sha256_of(data), item["error_message"])
        self.assertEqual(self.extractions(download_id), [])
        stored = repo.get_downloads_for_resource(RESOURCE_URL, self.db_path)[0]
        self.assertEqual(stored["sha256"], "0" * 64)

    def test_extractor_is_not_called_when_hash_mismatches(self):
        self.add_download(sha256="0" * 64)
        extract = mock.Mock()
        with mock.patch.dict(backfill.EXTRACTORS, {".zip": ("zip", extract)}):
            self.run_apply()
        extract.assert_not_called()

    def test_directory_instead_of_file_fails(self):
        directory = self.files_dir / "somedir.zip"
        directory.mkdir()
        self.add_download(saved_path=directory, create_file=False)
        (item,) = self.run_apply()["items"]
        self.assertEqual(item["status"], "failed")
        self.assertEqual(item["error_type"], "OSError")

    def test_read_verified_file_recomputes_size_and_hash(self):
        data = b"hello"
        path = self.write_file("x.bin", data)
        verified = backfill.read_verified_file(str(path), sha256_of(data))
        self.assertEqual(verified["size_bytes"], 5)
        self.assertEqual(verified["sha256"], sha256_of(data))
        self.assertEqual(verified["data"], data)

    def test_unsupported_local_file_is_no_supported_without_touching_it(self):
        download_id = self.add_download(filename="scan.pdf", data=b"%PDF-1.4", source_ref="pdf")
        summary = self.run_apply()
        (item,) = summary["items"]
        self.assertEqual(item["status"], "no_supported")
        self.assertEqual(summary["no_supported_count"], 1)
        self.assertEqual(summary["failed_count"], 0)
        self.assertEqual(self.extractions(download_id), [])
        self.assertIsNone(self.state())

    def test_unsupported_file_does_not_stop_batch(self):
        self.add_download(filename="scan.pdf", data=b"%PDF", source_ref="pdf")
        good = self.add_download(
            filename="a.docx", data=docx_bytes(), source_ref="docx",
        )
        summary = self.run_apply()
        self.assertEqual(summary["no_supported_count"], 1)
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(self.members(good), {"": "success"})


# --------------------------------------------------------------------------
# Extraction from local files
# --------------------------------------------------------------------------

class LocalExtractionTests(BackfillTestCase):
    def test_local_docx(self):
        download_id = self.add_download(
            filename="hraver.docx", data=make_docx_bytes(["Alpha", "Beta"]), source_ref="docx",
        )
        summary = self.run_apply()
        self.assertEqual(summary["success_count"], 1)
        (row,) = self.extractions(download_id)
        self.assertEqual(row["member_name"], "")
        self.assertEqual(row["extraction_status"], "success")
        self.assertEqual(row["text"], "Alpha\nBeta")
        self.assertEqual(row["paragraph_count"], 2)
        self.assertEqual(summary["items"][0]["extraction_type"], "docx")

    def test_local_xlsx(self):
        download_id = self.add_download(
            filename="lot.xlsx", data=xlsx_bytes(), source_ref="xlsx",
        )
        summary = self.run_apply()
        self.assertEqual(summary["success_count"], 1)
        (row,) = self.extractions(download_id)
        self.assertEqual(row["member_name"], "")
        self.assertEqual(row["file_type"], "xlsx")
        self.assertEqual((row["sheet_count"], row["row_count"], row["cell_count"]), (1, 2, 4))
        self.assertIn("Bolt\t10", row["text"])
        self.assertEqual(summary["items"][0]["extraction_type"], "xlsx")

    def test_local_zip(self):
        data = make_zip_bytes([
            ("hraver.docx", docx_bytes()),
            ("lot_1.xlsx", xlsx_bytes()),
            ("old.rar", b"Rar!"),
        ])
        download_id = self.add_download(data=data)
        summary = self.run_apply()
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(self.members(download_id), {
            "hraver.docx": "success", "lot_1.xlsx": "success", "old.rar": "skipped",
        })
        self.assertEqual(summary["items"][0]["extraction_type"], "zip")
        self.assertEqual(summary["extractions_new_count"], 3)

    def test_nested_zip(self):
        inner = make_zip_bytes([("spec.xlsx", xlsx_bytes())])
        data = make_zip_bytes([("hraver.docx", docx_bytes()), ("lot_1.zip", inner)])
        download_id = self.add_download(data=data)
        self.run_apply()
        self.assertEqual(self.members(download_id), {
            "hraver.docx": "success", "lot_1.zip!/spec.xlsx": "success",
        })

    def test_zip_with_more_than_old_member_limit(self):
        """Прежний max_members=50 ронял такие ZIP; теперь лимит 200."""
        data = make_zip_bytes([(f"lot_{n}.xlsx", xlsx_bytes()) for n in range(60)])
        download_id = self.add_download(data=data)
        summary = self.run_apply()
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(len(self.extractions(download_id)), 60)

    def test_corrupted_member_is_saved_as_failed_but_item_is_success(self):
        data = make_zip_bytes([("ok.docx", docx_bytes()), ("broken.docx", b"garbage")])
        download_id = self.add_download(data=data)
        summary = self.run_apply()
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(
            self.members(download_id), {"ok.docx": "success", "broken.docx": "failed"}
        )

    def test_extractor_error_marks_item_failed_and_keeps_old_rows(self):
        data = make_zip_bytes([("a.docx", docx_bytes())])
        download_id = self.add_download(data=data)
        self.add_old_extraction(download_id, "old.xlsx", "skipped")
        with mock.patch.dict(
            backfill.EXTRACTORS,
            {".zip": ("zip", mock.Mock(side_effect=ValueError("ZIP has too many members")))},
        ):
            summary = self.run_apply()
        self.assertEqual(summary["failed_count"], 1)
        self.assertEqual(summary["failures"][0]["error_type"], "ValueError")
        self.assertIn("too many members", summary["failures"][0]["error_message"])
        self.assertEqual(self.members(download_id), {"old.xlsx": "skipped"})

    def test_one_failed_item_does_not_stop_next(self):
        first = self.add_download(
            filename="a.zip", source_ref="a", create_file=False,
        )
        second = self.add_download(
            filename="b.zip", source_ref="b", data=make_zip_bytes([("x.docx", docx_bytes())]),
        )
        third = self.add_download(
            filename="c.docx", source_ref="c", data=docx_bytes(),
        )
        summary = self.run_apply()
        self.assertEqual(summary["candidate_count"], 3)
        self.assertEqual(summary["processed_count"], 3)
        self.assertEqual(summary["success_count"], 2)
        self.assertEqual(summary["failed_count"], 1)
        self.assertEqual([item["download_id"] for item in summary["items"]], [first, second, third])
        self.assertEqual(self.members(second), {"x.docx": "success"})
        self.assertEqual(self.members(third), {"": "success"})

    def test_limit_applies_to_apply(self):
        for n in range(3):
            self.add_download(
                filename=f"d{n}.docx", data=docx_bytes(f"t{n}"), source_ref=f"ref-{n}",
            )
        summary = self.run_apply(limit=2)
        self.assertEqual(summary["candidate_count"], 2)
        self.assertEqual(summary["processed_count"], 2)
        self.assertEqual(len(self.candidate_ids()), 1)

    def test_apply_adds_missing_metric_columns_to_legacy_schema(self):
        download_id = self.add_download(filename="lot.xlsx", data=xlsx_bytes(), source_ref="x")
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            for name in ("sheet_count", "row_count", "cell_count"):
                conn.execute(f"ALTER TABLE document_extractions DROP COLUMN {name}")
        summary = self.run_apply()
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(self.extractions(download_id)[0]["sheet_count"], 1)


# --------------------------------------------------------------------------
# Reconciliation through backfill
# --------------------------------------------------------------------------

class ReconciliationTests(BackfillTestCase):
    def test_skipped_xlsx_becomes_success(self):
        data = make_zip_bytes([("hraver.docx", docx_bytes()), ("lot_1.xlsx", xlsx_bytes())])
        download_id = self.add_download(data=data)
        self.add_success_extraction(download_id, "hraver.docx")
        self.add_old_extraction(download_id, "lot_1.xlsx", "skipped")
        rows_before = {row["member_name"]: row["id"] for row in self.extractions(download_id)}

        summary = self.run_apply()

        self.assertEqual(self.members(download_id), {
            "hraver.docx": "success", "lot_1.xlsx": "success",
        })
        row = next(row for row in self.extractions(download_id) if row["member_name"] == "lot_1.xlsx")
        self.assertEqual(row["id"], rows_before["lot_1.xlsx"])
        self.assertEqual(row["sheet_count"], 1)
        self.assertEqual(summary["extractions_updated_count"], 2)  # xlsx и docx (метрики XLSX/тексты)
        self.assertEqual(summary["extractions_deleted_count"], 0)

    def test_obsolete_skipped_nested_zip_row_is_removed(self):
        inner = make_zip_bytes([("spec.xlsx", xlsx_bytes())])
        data = make_zip_bytes([("hraver.docx", docx_bytes()), ("lot_1.zip", inner)])
        download_id = self.add_download(data=data)
        self.add_success_extraction(download_id, "hraver.docx")
        self.add_old_extraction(download_id, "lot_1.zip", "skipped")

        summary = self.run_apply()

        self.assertEqual(self.members(download_id), {
            "hraver.docx": "success", "lot_1.zip!/spec.xlsx": "success",
        })
        self.assertEqual(summary["extractions_deleted_count"], 1)
        self.assertEqual(summary["extractions_new_count"], 1)

    def test_backfill_does_not_change_downloads_or_files(self):
        data = make_zip_bytes([("hraver.docx", docx_bytes())])
        download_id = self.add_download(data=data, sha256=None)
        self.add_old_extraction(download_id, "old.xlsx", "skipped")
        downloads_before = repo.get_downloads_for_resource(RESOURCE_URL, self.db_path)
        files_before = self.files_snapshot()

        self.run_apply()

        self.assertEqual(repo.get_downloads_for_resource(RESOURCE_URL, self.db_path), downloads_before)
        self.assertEqual(self.files_snapshot(), files_before)

    def test_second_run_has_no_candidates_and_changes_nothing(self):
        inner = make_zip_bytes([("spec.xlsx", xlsx_bytes())])
        data = make_zip_bytes([
            ("hraver.docx", docx_bytes()),
            ("lot_1.xlsx", xlsx_bytes()),
            ("lot_2.zip", inner),
            ("old.rar", b"Rar!"),
        ])
        download_id = self.add_download(data=data)
        self.add_old_extraction(download_id, "lot_1.xlsx", "skipped")
        self.add_old_extraction(download_id, "lot_2.zip", "skipped")
        self.run_apply()
        before = self.db_snapshot()

        second = self.run_apply()

        self.assertEqual(second["candidate_count"], 0)
        self.assertEqual(self.db_snapshot(), before)

    def test_rerun_of_same_candidate_reports_existing_rows(self):
        download_id = self.add_download(
            filename="a.docx", data=docx_bytes(), source_ref="docx",
        )
        self.run_apply()
        item = backfill.process_candidate(
            {**self.candidate_like(download_id), "reasons": []}, db_path=self.db_path
        )
        self.assertEqual(item["extraction_storage"]["existing_count"], 1)
        self.assertEqual(item["extraction_storage"]["updated_count"], 0)

    def candidate_like(self, download_id):
        return next(
            download for download in repo.get_downloads_for_resource(RESOURCE_URL, self.db_path)
            if download["id"] == download_id
        )


# --------------------------------------------------------------------------
# Processing state
# --------------------------------------------------------------------------

class ProcessingStateTests(BackfillTestCase):
    def test_latest_download_success_updates_state(self):
        download_id = self.add_download()
        repo.save_processing_state(
            RESOURCE_URL, "failed", error_type="ValueError", error_message="ZIP has too many members",
            db_path=self.db_path,
        )

        summary = self.run_apply()

        state = self.state()
        self.assertEqual(state["status"], "success")
        self.assertEqual((state["source_kind"], state["source_ref"]), (SOURCE_KIND, SOURCE_REF))
        self.assertIsNone(state["error_type"])
        self.assertIsNone(state["error_message"])
        self.assertEqual(summary["processing_state_updated_count"], 1)
        self.assertEqual(summary["items"][0]["processing_state"], backfill.STATE_UPDATED)
        self.assertEqual(summary["items"][0]["download_id"], download_id)

    def test_state_is_created_when_missing(self):
        self.add_download()
        self.run_apply()
        self.assertEqual(self.state()["status"], "success")

    def test_old_version_does_not_change_state(self):
        old_data = make_zip_bytes([("old.docx", docx_bytes("old"))])
        new_data = make_zip_bytes([("new.docx", docx_bytes("new"))])
        old_id = self.add_download(data=old_data)
        new_id = self.add_download(data=new_data)
        self.assertNotEqual(old_id, new_id)
        self.add_success_extraction(new_id, "new.docx")
        repo.save_processing_state(
            RESOURCE_URL, "failed", error_type="X", error_message="y", db_path=self.db_path
        )
        before = self.state()

        summary = self.run_apply()

        self.assertEqual(summary["candidate_count"], 1)
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(self.members(old_id), {"old.docx": "success"})
        self.assertEqual(summary["items"][0]["processing_state"], backfill.STATE_NOT_LATEST)
        self.assertEqual(summary["processing_state_updated_count"], 0)
        self.assertEqual(self.state(), before)

    def test_failed_latest_sets_failed_state(self):
        self.add_download(sha256="0" * 64)
        repo.save_processing_state(
            RESOURCE_URL, "success", SOURCE_KIND, SOURCE_REF, db_path=self.db_path
        )

        summary = self.run_apply()

        state = self.state()
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["error_type"], "DocumentHashMismatchError")
        self.assertIn("sha256", state["error_message"])
        self.assertEqual((state["source_kind"], state["source_ref"]), (SOURCE_KIND, SOURCE_REF))
        self.assertEqual(summary["processing_state_updated_count"], 1)

    def test_failed_old_version_does_not_change_state(self):
        old_id = self.add_download(
            data=make_zip_bytes([("old.docx", docx_bytes("old"))]), sha256="0" * 64
        )
        new_id = self.add_download(data=make_zip_bytes([("new.docx", docx_bytes("new"))]))
        self.add_success_extraction(new_id, "new.docx")
        repo.save_processing_state(
            RESOURCE_URL, "success", SOURCE_KIND, SOURCE_REF, db_path=self.db_path
        )
        before = self.state()

        summary = self.run_apply()

        self.assertEqual(summary["failed_count"], 1)
        self.assertEqual(summary["failures"][0]["download_id"], old_id)
        self.assertEqual(self.state(), before)

    def test_state_bound_to_other_source_is_not_changed(self):
        self.add_download()
        repo.save_processing_state(
            RESOURCE_URL, "success", "armeps_document", "12345", db_path=self.db_path
        )
        before = self.state()

        summary = self.run_apply()

        self.assertEqual(summary["items"][0]["processing_state"], backfill.STATE_OTHER_SOURCE)
        self.assertEqual(self.state(), before)

    def test_identical_state_is_not_rewritten(self):
        self.add_download()
        repo.save_processing_state(
            RESOURCE_URL, "success", SOURCE_KIND, SOURCE_REF, db_path=self.db_path
        )
        before = self.state()

        summary = self.run_apply()

        self.assertEqual(summary["items"][0]["processing_state"], backfill.STATE_UNCHANGED)
        self.assertEqual(summary["processing_state_updated_count"], 0)
        self.assertEqual(self.state(), before)

    def test_states_of_other_resources_are_not_touched(self):
        other_url = "https://example.test/resource/other"
        self.add_announcement(other_url)
        repo.save_processing_state(other_url, "failed", error_type="E", db_path=self.db_path)
        before = self.state(other_url)
        self.add_download()
        self.run_apply()
        self.assertEqual(self.state(other_url), before)

    def test_state_update_error_marks_item_failed_but_keeps_extraction(self):
        download_id = self.add_download()
        with mock.patch.object(
            repo, "save_processing_state", side_effect=sqlite3.OperationalError("locked")
        ):
            summary = self.run_apply()
        (item,) = summary["items"]
        self.assertEqual(item["status"], "failed")
        self.assertIn("processing state update failed", item["error_message"])
        self.assertEqual(self.members(download_id), {"hraver.docx": "success"})


# --------------------------------------------------------------------------
# Dry run, CLI, network
# --------------------------------------------------------------------------

class DryRunTests(BackfillTestCase):
    def setUp(self):
        super().setUp()
        inner = make_zip_bytes([("spec.xlsx", xlsx_bytes())])
        data = make_zip_bytes([("hraver.docx", docx_bytes()), ("lot_1.zip", inner)])
        self.skipped_id = self.add_download(data=data)
        self.add_old_extraction(self.skipped_id, "hraver.docx", "success")
        self.add_old_extraction(self.skipped_id, "lot_1.zip", "skipped")
        self.failed_id = self.add_download(
            filename="b.zip", source_ref="b", create_file=False,
        )
        repo.save_processing_state(
            RESOURCE_URL, "failed", error_type="ValueError", db_path=self.db_path
        )

    def test_dry_run_makes_no_database_or_file_changes(self):
        db_before = self.db_snapshot()
        files_before = self.files_snapshot()
        db_file_before = file_sha256(self.db_path)

        summary = backfill.run_backfill(db_path=self.db_path)

        self.assertEqual(self.db_snapshot(), db_before)
        self.assertEqual(self.files_snapshot(), files_before)
        self.assertEqual(file_sha256(self.db_path), db_file_before)
        self.assertEqual(summary["mode"], "dry_run")

    def test_dry_run_is_default_and_does_not_extract_or_write(self):
        extractors = {ext: (name, mock.Mock()) for ext, (name, _) in backfill.EXTRACTORS.items()}
        with mock.patch.dict(backfill.EXTRACTORS, extractors), \
                mock.patch.object(repo, "init_db") as init_db, \
                mock.patch.object(repo, "replace_extractions_for_download") as replace, \
                mock.patch.object(repo, "save_processing_state") as save_state, \
                mock.patch.object(backfill, "read_verified_file") as read_file:
            backfill.run_backfill(db_path=self.db_path)

        for _, extract in extractors.values():
            extract.assert_not_called()
        init_db.assert_not_called()
        replace.assert_not_called()
        save_state.assert_not_called()
        read_file.assert_not_called()

    def test_dry_run_reports_candidates_reasons_and_file_existence(self):
        summary = backfill.run_backfill(db_path=self.db_path)
        self.assertEqual(summary["candidate_count"], 2)
        self.assertEqual(summary["processed_count"], 0)
        first, second = summary["items"]
        self.assertEqual(first["download_id"], self.skipped_id)
        self.assertEqual(first["reasons"], [repo.BACKFILL_REASON_SKIPPED_SUPPORTED])
        self.assertEqual(first["skipped_supported_count"], 1)
        self.assertTrue(first["file_exists"])
        self.assertEqual(second["download_id"], self.failed_id)
        self.assertEqual(second["reasons"], [repo.BACKFILL_REASON_NO_EXTRACTION])
        self.assertFalse(second["file_exists"])

    def test_dry_run_does_not_use_network(self):
        with forbid_network():
            backfill.run_backfill(db_path=self.db_path)

    def test_apply_does_not_use_network(self):
        with forbid_network():
            summary = self.run_apply()
        self.assertEqual(summary["processed_count"], 2)

    def test_dry_run_with_limit(self):
        summary = backfill.run_backfill(db_path=self.db_path, limit=1)
        self.assertEqual(summary["candidate_count"], 1)

    def test_dry_run_invalid_limit(self):
        with self.assertRaises(ValueError):
            backfill.run_backfill(db_path=self.db_path, limit=0)


class CliTests(BackfillTestCase):
    def setUp(self):
        super().setUp()
        self.skipped_id = self.add_download(
            data=make_zip_bytes([("hraver.docx", docx_bytes()), ("lot_1.xlsx", xlsx_bytes())])
        )
        self.add_old_extraction(self.skipped_id, "lot_1.xlsx", "skipped")
        self.add_download(filename="missing.zip", source_ref="m", create_file=False)

    def run_main(self, argv):
        """main() с временной БД вместо production и перехватом stdout."""
        real = backfill.run_backfill
        output = io.StringIO()
        with mock.patch.object(
            backfill, "run_backfill", side_effect=lambda **kw: real(db_path=self.db_path, **kw)
        ) as run, mock.patch.object(sys, "stdout", output), \
                mock.patch.object(logging, "basicConfig"):
            code = backfill.main(argv)
        return code, output.getvalue(), run

    def test_default_is_dry_run(self):
        db_before = self.db_snapshot()
        code, output, run = self.run_main([])
        self.assertEqual(code, 0)
        run.assert_called_once_with(apply=False, limit=None)
        self.assertEqual(self.db_snapshot(), db_before)
        self.assertIn("DRY RUN", output)
        self.assertIn("Candidates: 2", output)
        self.assertIn(f"download_id={self.skipped_id}", output)
        self.assertIn("reason=skipped supported members", output)
        self.assertIn("skipped_supported=1", output)
        self.assertIn("reason=no extraction", output)
        self.assertIn("MISSING", output)

    def test_apply_flag(self):
        code, output, run = self.run_main(["--apply", "--limit", "1"])
        run.assert_called_once_with(apply=True, limit=1)
        self.assertEqual(code, 0)
        self.assertIn("candidate_count: 1", output)
        self.assertIn("success_count: 1", output)
        self.assertIn("processing_state_updated_count: 1", output)

    def test_apply_returns_error_code_and_prints_failures(self):
        code, output, _ = self.run_main(["--apply"])
        self.assertEqual(code, 1)
        self.assertIn("failed_count: 1", output)
        self.assertIn("error_type: FileNotFoundError", output)
        self.assertIn(RESOURCE_URL, output)

    def test_invalid_limit_is_rejected(self):
        for value in ("0", "-1", "abc"):
            with self.subTest(value=value):
                with mock.patch.object(sys, "stderr", io.StringIO()):
                    with self.assertRaises(SystemExit):
                        backfill.parse_args(["--limit", value])

    def test_parse_args_defaults(self):
        args = backfill.parse_args([])
        self.assertFalse(args.apply)
        self.assertIsNone(args.limit)


class ModuleIsolationTests(unittest.TestCase):
    def test_module_does_not_import_downloader_or_requests(self):
        code = (
            "import sys, src.backfill_document_extractions; "
            "bad = [m for m in ('requests', 'src.parser.document_downloader', "
            "'src.document_pipeline') if m in sys.modules]; "
            "sys.exit(1 if bad else 0)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
