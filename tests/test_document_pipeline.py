"""
Тесты document_pipeline. Сеть, SQLite и реальные файлы не используются:
download_*, extract_* и функции document_repository замоканы.

Запуск из корня проекта:
    python -m unittest tests.test_document_pipeline -v
"""

import copy
import logging
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import document_pipeline as pipeline
from src.database import announcement_repository, document_repository, enrichment_repository
from tests.test_document_extractor import make_docx_bytes, make_xlsx_bytes, make_zip_bytes

DB_PATH = "test.db"
ROOT_DIR = "test_root"

EAUCTION_ZIP_URL = "https://eauction.example.test/public_invitation/tender_1.zip"
ARMEPS_PAGE_URL = "https://armeps.example.test/epps/cft/listContractDocuments.do?d-3680175-p=1&id=1"


def armeps_document(document_id, filename, language, **extra) -> dict:
    document = {
        "document_id": document_id,
        "filename": filename,
        "language": language,
        "title": None,
        "description": None,
    }
    document.update(extra)
    return document


def make_eauction(number=1, document_url=EAUCTION_ZIP_URL) -> dict:
    return {
        "resource_url": f"https://eauction.example.test/tender/{number}",
        "resource_type": "eauction_tender_page",
        "document_url": document_url,
    }


def make_armeps(number=1, documents=None) -> dict:
    return {
        "resource_url": ARMEPS_PAGE_URL + str(number),
        "resource_type": "armeps_documents_page",
        "documents": documents if documents is not None else [
            armeps_document("11", "a.docx", "EN"),
            armeps_document("12", "b.docx", "RU"),
        ],
    }


def make_direct(url="https://files.example.test/x/file.docx") -> dict:
    return {"resource_url": url, "resource_type": "direct_file"}


def make_download_result(filename="file.zip", sha256="abc") -> dict:
    return {
        "source_url": "https://example.test/file",
        "saved_path": f"data/documents/hash/{filename}",
        "filename": filename,
        "content_type": "application/octet-stream",
        "size_bytes": 10,
        "sha256": sha256,
    }


class GetFileExtensionTests(unittest.TestCase):
    def test_url_with_query_string(self):
        self.assertEqual(pipeline.get_file_extension("https://example/x.DOCX?foo=1"), ".docx")

    def test_uppercase_docx_filename(self):
        self.assertEqual(pipeline.get_file_extension("REPORT.DOCX"), ".docx")

    def test_url_encoded_path(self):
        self.assertEqual(pipeline.get_file_extension("https://e/a%20b.zip"), ".zip")

    def test_no_extension(self):
        self.assertEqual(pipeline.get_file_extension("https://e/download?id=1.docx"), "")

    def test_windows_path_with_dotted_directory(self):
        self.assertEqual(pipeline.get_file_extension("C:\\data\\x.y\\file.docx"), ".docx")


class SelectArmepsDocumentTests(unittest.TestCase):
    def test_ru_preferred_over_en_and_hy(self):
        documents = [
            armeps_document("1", "hy.docx", "HY"),
            armeps_document("2", "en.docx", "EN"),
            armeps_document("3", "ru.docx", "RU"),
        ]
        self.assertEqual(pipeline.select_armeps_document(documents)["document_id"], "3")

    def test_en_when_no_ru(self):
        documents = [
            armeps_document("1", "hy.docx", "HY"),
            armeps_document("2", "en.docx", "EN"),
        ]
        self.assertEqual(pipeline.select_armeps_document(documents)["document_id"], "2")

    def test_hy_when_no_ru_and_en(self):
        documents = [
            armeps_document("1", "x.zip", "FR"),
            armeps_document("2", "hy.docx", "HY"),
        ]
        self.assertEqual(pipeline.select_armeps_document(documents)["document_id"], "2")

    def test_unknown_language_fallback_keeps_source_order(self):
        documents = [
            armeps_document("1", "first.docx", None),
            armeps_document("2", "second.docx", "FR"),
        ]
        self.assertEqual(pipeline.select_armeps_document(documents)["document_id"], "1")

    def test_first_of_same_language(self):
        documents = [
            armeps_document("1", "a.docx", "RU"),
            armeps_document("2", "b.docx", "RU"),
        ]
        self.assertEqual(pipeline.select_armeps_document(documents)["document_id"], "1")

    def test_xml_is_not_selected(self):
        documents = [
            armeps_document("1", "data.xml", "RU"),
            armeps_document("2", "tender.docx", "HY"),
        ]
        self.assertEqual(pipeline.select_armeps_document(documents)["document_id"], "2")

    def test_only_unsupported_documents(self):
        documents = [
            armeps_document("1", "data.xml", "RU"),
            armeps_document("2", "old.doc", "EN"),
            armeps_document("3", "scan.pdf", "HY"),
        ]
        self.assertIsNone(pipeline.select_armeps_document(documents))

    def test_document_without_document_id_is_not_selected(self):
        documents = [
            armeps_document(None, "ru.docx", "RU"),
            armeps_document("  ", "en.docx", "EN"),
            armeps_document("3", "hy.docx", "HY"),
        ]
        self.assertEqual(pipeline.select_armeps_document(documents)["document_id"], "3")

    def test_document_without_filename_is_not_selected(self):
        self.assertIsNone(pipeline.select_armeps_document([armeps_document("1", "", "RU")]))

    def test_empty_and_none(self):
        self.assertIsNone(pipeline.select_armeps_document([]))
        self.assertIsNone(pipeline.select_armeps_document(None))

    def test_source_list_is_not_modified(self):
        documents = [
            armeps_document("1", "hy.docx", "HY"),
            armeps_document("2", "ru.docx", "RU"),
        ]
        original = copy.deepcopy(documents)
        pipeline.select_armeps_document(documents)
        self.assertEqual(documents, original)


class SelectArmepsFormatPriorityTests(unittest.TestCase):
    def select(self, documents):
        return pipeline.select_armeps_document(documents)["document_id"]

    def test_supported_extensions_order_is_format_priority(self):
        self.assertEqual(pipeline.SUPPORTED_DOCUMENT_EXTENSIONS, (".docx", ".xlsx", ".zip"))

    def test_xlsx_is_selected_when_it_is_the_only_document(self):
        self.assertEqual(self.select([armeps_document("1", "spec.xlsx", "RU")]), "1")

    def test_docx_preferred_over_xlsx_in_same_language(self):
        documents = [
            armeps_document("1", "spec.xlsx", "RU"),
            armeps_document("2", "hraver.docx", "RU"),
        ]
        self.assertEqual(self.select(documents), "2")

    def test_xlsx_preferred_over_zip_in_same_language(self):
        documents = [
            armeps_document("1", "all.zip", "RU"),
            armeps_document("2", "spec.xlsx", "RU"),
        ]
        self.assertEqual(self.select(documents), "2")

    def test_language_priority_beats_format_priority(self):
        documents = [
            armeps_document("1", "en.docx", "EN"),
            armeps_document("2", "ru.xlsx", "RU"),
        ]
        self.assertEqual(self.select(documents), "2")

    def test_xlsx_fallback_keeps_ru_en_hy_order(self):
        documents = [
            armeps_document("1", "hy.xlsx", "HY"),
            armeps_document("2", "en.xlsx", "EN"),
            armeps_document("3", "ru.xlsx", "RU"),
        ]
        self.assertEqual(self.select(documents), "3")
        self.assertEqual(self.select(documents[:2]), "2")
        self.assertEqual(self.select(documents[:1]), "1")

    def test_ru_xlsx_when_ru_has_no_docx_but_en_has_docx(self):
        documents = [
            armeps_document("1", "en.docx", "EN"),
            armeps_document("2", "ru.xlsx", "RU"),
            armeps_document("3", "ru.pdf", "RU"),
        ]
        self.assertEqual(self.select(documents), "2")

    def test_same_format_same_language_keeps_source_order(self):
        documents = [
            armeps_document("1", "a.xlsx", "RU"),
            armeps_document("2", "b.xlsx", "RU"),
        ]
        self.assertEqual(self.select(documents), "1")

    def test_unknown_language_group_uses_format_priority(self):
        documents = [
            armeps_document("1", "a.zip", "FR"),
            armeps_document("2", "b.xlsx", None),
            armeps_document("3", "c.docx", "DE"),
        ]
        self.assertEqual(self.select(documents), "3")

    def test_rar_and_xls_are_not_selected(self):
        documents = [
            armeps_document("1", "a.rar", "RU"),
            armeps_document("2", "b.xls", "RU"),
        ]
        self.assertIsNone(pipeline.select_armeps_document(documents))


class PlanDocumentSourceTests(unittest.TestCase):
    def test_eauction_zip(self):
        announcement = make_eauction()
        self.assertEqual(pipeline.plan_document_source(announcement), {
            "resource_url": announcement["resource_url"],
            "source_kind": "eauction_document",
            "source_ref": EAUCTION_ZIP_URL,
            "download_type": "eauction",
            "source_url": EAUCTION_ZIP_URL,
            "document_id": None,
            "expected_filename": None,
        })

    def test_eauction_without_document_url(self):
        self.assertIsNone(pipeline.plan_document_source(make_eauction(document_url=None)))
        self.assertIsNone(pipeline.plan_document_source(make_eauction(document_url="")))

    def test_eauction_unsupported_extension(self):
        self.assertIsNone(pipeline.plan_document_source(
            make_eauction(document_url="https://e/tender_1.pdf")
        ))

    def test_armeps_selects_ru_docx(self):
        announcement = make_armeps()
        self.assertEqual(pipeline.plan_document_source(announcement), {
            "resource_url": announcement["resource_url"],
            "source_kind": "armeps_document",
            "source_ref": "12",
            "download_type": "armeps",
            "source_url": announcement["resource_url"],
            "document_id": "12",
            "expected_filename": "b.docx",
        })

    def test_armeps_without_suitable_document(self):
        announcement = make_armeps(documents=[armeps_document("1", "data.xml", "RU")])
        self.assertIsNone(pipeline.plan_document_source(announcement))

    def test_direct_docx(self):
        url = "https://files.example.test/x/My%20File.DOCX?v=1"
        self.assertEqual(pipeline.plan_document_source(make_direct(url)), {
            "resource_url": url,
            "source_kind": "direct_file",
            "source_ref": url,
            "download_type": "direct",
            "source_url": url,
            "document_id": None,
            "expected_filename": "My File.DOCX",
        })

    def test_direct_zip(self):
        url = "https://files.example.test/x/archive.zip"
        plan = pipeline.plan_document_source(make_direct(url))
        self.assertEqual(plan["download_type"], "direct")
        self.assertEqual(plan["expected_filename"], "archive.zip")

    def test_direct_xlsx(self):
        url = "https://files.example.test/x/Spec.XLSX"
        plan = pipeline.plan_document_source(make_direct(url))
        self.assertEqual(plan["download_type"], "direct")
        self.assertEqual(plan["source_kind"], "direct_file")
        self.assertEqual(plan["expected_filename"], "Spec.XLSX")

    def test_direct_rar_and_xls_are_not_supported(self):
        for name in ("archive.rar", "old.xls"):
            self.assertIsNone(pipeline.plan_document_source(
                make_direct(f"https://files.example.test/x/{name}")
            ))

    def test_armeps_xlsx_fallback(self):
        announcement = make_armeps(documents=[
            armeps_document("21", "en.xlsx", "EN"),
            armeps_document("22", "data.xml", "RU"),
        ])
        plan = pipeline.plan_document_source(announcement)
        self.assertEqual(plan["source_kind"], "armeps_document")
        self.assertEqual(plan["document_id"], "21")
        self.assertEqual(plan["expected_filename"], "en.xlsx")

    def test_armeps_prefers_ru_docx_over_ru_xlsx(self):
        announcement = make_armeps(documents=[
            armeps_document("31", "spec.xlsx", "RU"),
            armeps_document("32", "hraver.docx", "RU"),
        ])
        self.assertEqual(pipeline.plan_document_source(announcement)["document_id"], "32")

    def test_eauction_xlsx_document_url(self):
        announcement = make_eauction(document_url="https://e.test/public_invitation/spec.xlsx")
        self.assertEqual(pipeline.plan_document_source(announcement)["download_type"], "eauction")

    def test_direct_doc_is_not_supported(self):
        self.assertIsNone(pipeline.plan_document_source(
            make_direct("https://files.example.test/x/old.doc")
        ))

    def test_unknown_resource_type(self):
        announcement = {
            "resource_url": "https://x.test/1",
            "resource_type": "something_else",
            "document_url": EAUCTION_ZIP_URL,
        }
        self.assertIsNone(pipeline.plan_document_source(announcement))
        self.assertIsNone(pipeline.plan_document_source({"resource_url": "https://x.test/1"}))


class DownloadPlannedSourceTests(unittest.TestCase):
    def test_routes_eauction(self):
        plan = pipeline.plan_document_source(make_eauction())
        with mock.patch.object(pipeline, "download_eauction_document") as eauction, \
                mock.patch.object(pipeline, "download_armeps_document") as armeps, \
                mock.patch.object(pipeline, "download_direct_file") as direct:
            result = pipeline.download_planned_source(plan, root_dir=ROOT_DIR)

        eauction.assert_called_once_with(
            document_url=EAUCTION_ZIP_URL, resource_url=plan["resource_url"], root_dir=ROOT_DIR,
        )
        armeps.assert_not_called()
        direct.assert_not_called()
        self.assertIs(result, eauction.return_value)

    def test_routes_armeps(self):
        plan = pipeline.plan_document_source(make_armeps())
        with mock.patch.object(pipeline, "download_eauction_document") as eauction, \
                mock.patch.object(pipeline, "download_armeps_document") as armeps, \
                mock.patch.object(pipeline, "download_direct_file") as direct:
            result = pipeline.download_planned_source(plan, root_dir=ROOT_DIR)

        armeps.assert_called_once_with(
            resource_url=plan["resource_url"], document_id="12",
            expected_filename="b.docx", root_dir=ROOT_DIR,
        )
        eauction.assert_not_called()
        direct.assert_not_called()
        self.assertIs(result, armeps.return_value)

    def test_routes_direct(self):
        plan = pipeline.plan_document_source(make_direct())
        with mock.patch.object(pipeline, "download_eauction_document") as eauction, \
                mock.patch.object(pipeline, "download_armeps_document") as armeps, \
                mock.patch.object(pipeline, "download_direct_file") as direct:
            result = pipeline.download_planned_source(plan, root_dir=ROOT_DIR)

        direct.assert_called_once_with(
            url=plan["source_url"], resource_url=plan["resource_url"], root_dir=ROOT_DIR,
        )
        eauction.assert_not_called()
        armeps.assert_not_called()
        self.assertIs(result, direct.return_value)

    def test_unknown_download_type(self):
        with self.assertRaises(ValueError):
            pipeline.download_planned_source({"download_type": "other"})


class ExtractDownloadedFileTests(unittest.TestCase):
    def test_docx_routing(self):
        download = make_download_result("tender.DOCX")
        with mock.patch.object(pipeline, "extract_docx") as extract_docx, \
                mock.patch.object(pipeline, "extract_supported_from_zip") as extract_zip:
            result = pipeline.extract_downloaded_file(download)

        extract_docx.assert_called_once_with(download["saved_path"])
        extract_zip.assert_not_called()
        self.assertEqual(result, {"extraction_type": "docx", "result": extract_docx.return_value})

    def test_zip_routing(self):
        download = make_download_result("tender.zip")
        with mock.patch.object(pipeline, "extract_docx") as extract_docx, \
                mock.patch.object(pipeline, "extract_supported_from_zip") as extract_zip:
            result = pipeline.extract_downloaded_file(download)

        extract_zip.assert_called_once_with(download["saved_path"])
        extract_docx.assert_not_called()
        self.assertEqual(result, {"extraction_type": "zip", "result": extract_zip.return_value})

    def test_xlsx_routing(self):
        download = make_download_result("spec.XLSX")
        with mock.patch.object(pipeline, "extract_docx") as extract_docx, \
                mock.patch.object(pipeline, "extract_xlsx") as extract_xlsx, \
                mock.patch.object(pipeline, "extract_supported_from_zip") as extract_zip:
            result = pipeline.extract_downloaded_file(download)

        extract_xlsx.assert_called_once_with(download["saved_path"])
        extract_docx.assert_not_called()
        extract_zip.assert_not_called()
        self.assertEqual(result, {"extraction_type": "xlsx", "result": extract_xlsx.return_value})

    def test_rar_and_xls_raise(self):
        for name in ("docs.rar", "old.xls"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    pipeline.extract_downloaded_file(make_download_result(name))

    def test_unsupported_file_raises(self):
        with mock.patch.object(pipeline, "extract_docx") as extract_docx, \
                mock.patch.object(pipeline, "extract_supported_from_zip") as extract_zip:
            with self.assertRaises(ValueError):
                pipeline.extract_downloaded_file(make_download_result("old.doc"))

        extract_docx.assert_not_called()
        extract_zip.assert_not_called()


class PipelineMocks:
    """Мокает все внешние зависимости pipeline; используется как контекст-менеджер."""

    def __init__(self, download_result=None, extraction_counts=None, storage_status="new"):
        self.download_result = download_result or make_download_result("file.zip")
        self.extraction_counts = extraction_counts or {
            "new_count": 1, "updated_count": 0, "existing_count": 0,
        }
        self.storage_status = storage_status
        self._patchers = []

    def __enter__(self):
        def patch(name, **kwargs):
            patcher = mock.patch.object(pipeline, name, **kwargs)
            self._patchers.append(patcher)
            return patcher.start()

        def patch_repo(name, **kwargs):
            patcher = mock.patch.object(pipeline.document_repository, name, **kwargs)
            self._patchers.append(patcher)
            return patcher.start()

        self.download_eauction = patch("download_eauction_document", return_value=self.download_result)
        self.download_armeps = patch("download_armeps_document", return_value=self.download_result)
        self.download_direct = patch("download_direct_file", return_value=self.download_result)
        self.extract_docx = patch("extract_docx", return_value={"text": "docx text"})
        self.extract_xlsx = patch("extract_xlsx", return_value={"text": "xlsx text"})
        self.extract_zip = patch("extract_supported_from_zip", return_value={"documents": []})
        self.init_db = patch_repo("init_db")
        self.save_download = patch_repo(
            "save_download", return_value={"status": self.storage_status, "download_id": 7},
        )
        self.save_docx = patch_repo(
            "save_docx_extraction", return_value=dict(self.extraction_counts),
        )
        self.save_xlsx = patch_repo(
            "save_xlsx_extraction", return_value=dict(self.extraction_counts),
        )
        self.save_zip = patch_repo(
            "save_zip_extraction", return_value=dict(self.extraction_counts),
        )
        self.save_state = patch_repo("save_processing_state")
        return self

    def __exit__(self, *exc_info):
        for patcher in reversed(self._patchers):
            patcher.stop()
        return False

    def downloaders(self):
        return (self.download_eauction, self.download_armeps, self.download_direct)


class ProcessEnrichedAnnouncementTests(unittest.TestCase):
    def test_no_supported_source_does_not_download_or_save(self):
        announcement = make_direct("https://files.example.test/x/old.doc")
        with PipelineMocks() as mocks:
            result = pipeline.process_enriched_announcement(
                announcement, db_path=DB_PATH, root_dir=ROOT_DIR,
            )

        self.assertEqual(result, {
            "resource_url": announcement["resource_url"],
            "status": "no_supported_document",
            "plan": None,
        })
        for downloader in mocks.downloaders():
            downloader.assert_not_called()
        mocks.save_download.assert_not_called()
        mocks.extract_docx.assert_not_called()
        mocks.extract_zip.assert_not_called()

    def test_direct_doc_saves_no_supported_document_state(self):
        announcement = make_direct("https://files.example.test/x/old.doc")
        with PipelineMocks() as mocks:
            pipeline.process_enriched_announcement(announcement, db_path=DB_PATH, root_dir=ROOT_DIR)

        mocks.save_state.assert_called_once_with(
            announcement["resource_url"], "no_supported_document", db_path=DB_PATH,
        )

    def test_successful_docx_saves_success_state_with_source(self):
        announcement = make_armeps()
        with PipelineMocks(download_result=make_download_result("b.docx")) as mocks:
            pipeline.process_enriched_announcement(announcement, db_path=DB_PATH, root_dir=ROOT_DIR)

        mocks.save_state.assert_called_once_with(
            announcement["resource_url"], "success",
            source_kind="armeps_document", source_ref="12", db_path=DB_PATH,
        )

    def test_success_state_is_saved_after_extraction(self):
        calls = mock.Mock()
        with PipelineMocks() as mocks:
            calls.attach_mock(mocks.save_download, "save_download")
            calls.attach_mock(mocks.save_zip, "save_zip")
            calls.attach_mock(mocks.save_state, "save_state")
            pipeline.process_enriched_announcement(make_eauction())

        self.assertEqual(
            [c[0] for c in calls.mock_calls], ["save_download", "save_zip", "save_state"],
        )

    def test_extraction_error_does_not_save_success_state(self):
        with PipelineMocks() as mocks:
            mocks.extract_zip.side_effect = RuntimeError("bad zip")
            with self.assertRaises(RuntimeError):
                pipeline.process_enriched_announcement(make_eauction())

        mocks.save_state.assert_not_called()

    def test_successful_eauction_zip(self):
        announcement = make_eauction()
        download = make_download_result("tender_1.zip")
        counts = {"new_count": 2, "updated_count": 0, "existing_count": 1}
        with PipelineMocks(download_result=download, extraction_counts=counts) as mocks:
            result = pipeline.process_enriched_announcement(
                announcement, db_path=DB_PATH, root_dir=ROOT_DIR,
            )

        mocks.download_eauction.assert_called_once_with(
            document_url=EAUCTION_ZIP_URL, resource_url=announcement["resource_url"],
            root_dir=ROOT_DIR,
        )
        mocks.save_download.assert_called_once_with(
            resource_url=announcement["resource_url"],
            source_kind="eauction_document",
            source_ref=EAUCTION_ZIP_URL,
            download_result=download,
            document_id=None,
            db_path=DB_PATH,
        )
        mocks.extract_zip.assert_called_once_with(download["saved_path"])
        mocks.extract_docx.assert_not_called()
        mocks.save_zip.assert_called_once_with(
            7, mocks.extract_zip.return_value, db_path=DB_PATH,
        )
        mocks.save_docx.assert_not_called()

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["resource_url"], announcement["resource_url"])
        self.assertEqual(result["plan"]["source_kind"], "eauction_document")
        self.assertIs(result["download"], download)
        self.assertEqual(result["download_storage_status"], "new")
        self.assertEqual(result["download_id"], 7)
        self.assertEqual(result["extraction_type"], "zip")
        self.assertEqual(result["extraction_storage"], counts)

    def test_successful_armeps_ru_docx(self):
        announcement = make_armeps()
        download = make_download_result("b.docx")
        with PipelineMocks(download_result=download) as mocks:
            result = pipeline.process_enriched_announcement(
                announcement, db_path=DB_PATH, root_dir=ROOT_DIR,
            )

        mocks.download_armeps.assert_called_once_with(
            resource_url=announcement["resource_url"], document_id="12",
            expected_filename="b.docx", root_dir=ROOT_DIR,
        )
        mocks.save_download.assert_called_once_with(
            resource_url=announcement["resource_url"],
            source_kind="armeps_document",
            source_ref="12",
            download_result=download,
            document_id="12",
            db_path=DB_PATH,
        )
        mocks.extract_docx.assert_called_once_with(download["saved_path"])
        mocks.save_docx.assert_called_once_with(
            7, mocks.extract_docx.return_value, db_path=DB_PATH,
        )
        mocks.save_zip.assert_not_called()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["extraction_type"], "docx")

    def test_successful_direct_docx(self):
        announcement = make_direct("https://files.example.test/x/file.docx")
        download = make_download_result("file.docx")
        with PipelineMocks(download_result=download) as mocks:
            result = pipeline.process_enriched_announcement(
                announcement, db_path=DB_PATH, root_dir=ROOT_DIR,
            )

        mocks.download_direct.assert_called_once_with(
            url=announcement["resource_url"], resource_url=announcement["resource_url"],
            root_dir=ROOT_DIR,
        )
        mocks.save_download.assert_called_once_with(
            resource_url=announcement["resource_url"],
            source_kind="direct_file",
            source_ref=announcement["resource_url"],
            download_result=download,
            document_id=None,
            db_path=DB_PATH,
        )
        mocks.save_docx.assert_called_once()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["extraction_type"], "docx")

    def test_successful_direct_xlsx(self):
        announcement = make_direct("https://files.example.test/x/spec.xlsx")
        download = make_download_result("spec.xlsx")
        with PipelineMocks(download_result=download) as mocks:
            result = pipeline.process_enriched_announcement(
                announcement, db_path=DB_PATH, root_dir=ROOT_DIR,
            )

        mocks.download_direct.assert_called_once_with(
            url=announcement["resource_url"], resource_url=announcement["resource_url"],
            root_dir=ROOT_DIR,
        )
        mocks.extract_xlsx.assert_called_once_with(download["saved_path"])
        mocks.save_xlsx.assert_called_once_with(
            7, mocks.extract_xlsx.return_value, db_path=DB_PATH,
        )
        mocks.save_docx.assert_not_called()
        mocks.save_zip.assert_not_called()
        mocks.extract_docx.assert_not_called()
        mocks.save_state.assert_called_once_with(
            announcement["resource_url"], "success",
            source_kind="direct_file", source_ref=announcement["resource_url"], db_path=DB_PATH,
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["extraction_type"], "xlsx")

    def test_armeps_xlsx_fallback_downloads_and_extracts_xlsx(self):
        announcement = make_armeps(documents=[
            armeps_document("11", "en.xlsx", "EN"),
            armeps_document("12", "ru.pdf", "RU"),
        ])
        download = make_download_result("en.xlsx")
        with PipelineMocks(download_result=download) as mocks:
            result = pipeline.process_enriched_announcement(
                announcement, db_path=DB_PATH, root_dir=ROOT_DIR,
            )

        mocks.download_armeps.assert_called_once_with(
            resource_url=announcement["resource_url"], document_id="11",
            expected_filename="en.xlsx", root_dir=ROOT_DIR,
        )
        mocks.save_xlsx.assert_called_once()
        self.assertEqual(result["extraction_type"], "xlsx")

    def test_armeps_ru_docx_preferred_over_ru_xlsx(self):
        announcement = make_armeps(documents=[
            armeps_document("11", "ru.xlsx", "RU"),
            armeps_document("12", "ru.docx", "RU"),
        ])
        with PipelineMocks(download_result=make_download_result("ru.docx")) as mocks:
            result = pipeline.process_enriched_announcement(announcement)

        self.assertEqual(mocks.download_armeps.call_args.kwargs["document_id"], "12")
        self.assertEqual(result["extraction_type"], "docx")

    def test_direct_rar_is_no_supported_document_not_failure(self):
        announcement = make_direct("https://files.example.test/x/docs.rar")
        with PipelineMocks() as mocks:
            summary = pipeline.process_enriched_announcements([announcement], db_path=DB_PATH)

        self.assertEqual(summary["no_supported_document_count"], 1)
        self.assertEqual(summary["failed_count"], 0)
        for downloader in mocks.downloaders():
            downloader.assert_not_called()
        mocks.save_state.assert_called_once_with(
            announcement["resource_url"], "no_supported_document", db_path=DB_PATH,
        )

    def test_repeated_sha_reports_existing_download(self):
        with PipelineMocks(storage_status="existing") as mocks:
            result = pipeline.process_enriched_announcement(make_eauction())

        # Повторное скачивание разрешено: дубль определяет repository по sha256.
        mocks.download_eauction.assert_called_once()
        self.assertEqual(result["download_storage_status"], "existing")
        self.assertEqual(result["download_id"], 7)

    def test_blank_resource_url_fails_before_download(self):
        announcement = make_eauction()
        announcement["resource_url"] = None
        with PipelineMocks() as mocks:
            with self.assertRaises(ValueError):
                pipeline.process_enriched_announcement(announcement)

        for downloader in mocks.downloaders():
            downloader.assert_not_called()
        mocks.save_download.assert_not_called()

    def test_enriched_announcement_is_not_modified(self):
        announcement = make_armeps()
        original = copy.deepcopy(announcement)
        with PipelineMocks(download_result=make_download_result("b.docx")):
            pipeline.process_enriched_announcement(announcement)
        self.assertEqual(announcement, original)


class ProcessEnrichedAnnouncementsTests(unittest.TestCase):
    def test_init_db_called_once(self):
        announcements = [make_eauction(1), make_eauction(2), make_eauction(3)]
        with PipelineMocks() as mocks:
            pipeline.process_enriched_announcements(announcements, db_path=DB_PATH)

        mocks.init_db.assert_called_once_with(db_path=DB_PATH)

    def test_error_does_not_stop_next_announcement(self):
        announcements = [make_eauction(1), make_eauction(2)]
        with PipelineMocks() as mocks:
            mocks.download_eauction.side_effect = [
                RuntimeError("network down"),
                make_download_result("tender_2.zip"),
            ]
            with self.assertLogs(pipeline.logger, level="ERROR"):
                summary = pipeline.process_enriched_announcements(announcements)

        self.assertEqual(mocks.download_eauction.call_count, 2)
        self.assertEqual(summary["processed_count"], 2)
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["failed_count"], 1)
        self.assertEqual(len(summary["results"]), 1)
        self.assertEqual(summary["results"][0]["resource_url"], announcements[1]["resource_url"])

    def test_failure_has_resource_url_error_type_and_message(self):
        announcement = make_eauction(1)
        with PipelineMocks() as mocks:
            mocks.download_eauction.side_effect = RuntimeError("network down")
            with self.assertLogs(pipeline.logger, level="ERROR"):
                summary = pipeline.process_enriched_announcements([announcement])

        self.assertEqual(summary["failures"], [{
            "resource_url": announcement["resource_url"],
            "error_type": "RuntimeError",
            "error_message": "network down",
        }])

    def test_no_supported_document_is_not_a_failure(self):
        announcements = [
            make_direct("https://files.example.test/x/old.doc"),
            make_eauction(2),
        ]
        with PipelineMocks() as mocks:
            summary = pipeline.process_enriched_announcements(announcements)

        self.assertEqual(summary["processed_count"], 2)
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["no_supported_document_count"], 1)
        self.assertEqual(summary["failed_count"], 0)
        self.assertEqual(summary["failures"], [])
        self.assertEqual(len(summary["results"]), 2)
        mocks.download_direct.assert_not_called()

    def test_download_new_and_existing_counters(self):
        announcements = [make_eauction(1), make_eauction(2), make_eauction(3)]
        with PipelineMocks() as mocks:
            mocks.save_download.side_effect = [
                {"status": "new", "download_id": 1},
                {"status": "existing", "download_id": 2},
                {"status": "new", "download_id": 3},
            ]
            summary = pipeline.process_enriched_announcements(announcements)

        self.assertEqual(summary["download_new_count"], 2)
        self.assertEqual(summary["download_existing_count"], 1)

    def test_extraction_counters_are_summed(self):
        announcements = [
            make_eauction(1),
            make_armeps(2, documents=[armeps_document("1", "a.docx", "RU")]),
            make_eauction(3),
        ]
        with PipelineMocks() as mocks:
            mocks.download_armeps.return_value = make_download_result("a.docx")
            mocks.save_zip.side_effect = [
                {"new_count": 2, "updated_count": 1, "existing_count": 0},
                {"new_count": 0, "updated_count": 0, "existing_count": 3},
            ]
            mocks.save_docx.return_value = {"new_count": 1, "updated_count": 0, "existing_count": 0}
            summary = pipeline.process_enriched_announcements(announcements)

        self.assertEqual(summary["success_count"], 3)
        self.assertEqual(summary["extraction_new_count"], 3)
        self.assertEqual(summary["extraction_updated_count"], 1)
        self.assertEqual(summary["extraction_existing_count"], 3)

    def test_empty_list(self):
        with PipelineMocks() as mocks:
            summary = pipeline.process_enriched_announcements([])

        mocks.init_db.assert_called_once()
        self.assertEqual(summary["processed_count"], 0)
        self.assertEqual(summary["results"], [])
        self.assertEqual(summary["failures"], [])
        mocks.save_state.assert_not_called()

    def test_exception_saves_failed_state_with_error(self):
        announcement = make_eauction(1)
        with PipelineMocks() as mocks:
            mocks.download_eauction.side_effect = RuntimeError("network down")
            with self.assertLogs(pipeline.logger, level="ERROR"):
                pipeline.process_enriched_announcements([announcement], db_path=DB_PATH)

        mocks.save_state.assert_called_once_with(
            announcement["resource_url"], "failed",
            error_type="RuntimeError", error_message="network down", db_path=DB_PATH,
        )

    def test_failed_state_does_not_stop_next_announcement(self):
        announcements = [make_eauction(1), make_eauction(2)]
        with PipelineMocks() as mocks:
            mocks.download_eauction.side_effect = [
                RuntimeError("boom"), make_download_result("tender_2.zip"),
            ]
            with self.assertLogs(pipeline.logger, level="ERROR"):
                pipeline.process_enriched_announcements(announcements)

        self.assertEqual(
            [(c.args[0], c.args[1]) for c in mocks.save_state.call_args_list],
            [
                (announcements[0]["resource_url"], "failed"),
                (announcements[1]["resource_url"], "success"),
            ],
        )

    def test_failure_to_save_failed_state_does_not_break_loop(self):
        announcements = [make_eauction(1), make_eauction(2)]
        with PipelineMocks() as mocks:
            mocks.download_eauction.side_effect = [
                RuntimeError("boom"), make_download_result("tender_2.zip"),
            ]
            mocks.save_state.side_effect = [ValueError("no announcement"), None]
            with self.assertLogs(pipeline.logger, level="ERROR"):
                summary = pipeline.process_enriched_announcements(announcements)

        self.assertEqual(summary["failed_count"], 1)
        self.assertEqual(summary["success_count"], 1)

    def test_failed_state_not_saved_without_resource_url(self):
        announcement = make_eauction()
        announcement["resource_url"] = None
        with PipelineMocks() as mocks:
            with self.assertLogs(pipeline.logger, level="ERROR"):
                summary = pipeline.process_enriched_announcements([announcement])

        self.assertEqual(summary["failed_count"], 1)
        mocks.save_state.assert_not_called()

    def test_init_db_failure_creates_no_state(self):
        with PipelineMocks() as mocks:
            mocks.init_db.side_effect = RuntimeError("db locked")
            with self.assertRaises(RuntimeError):
                pipeline.process_enriched_announcements([make_eauction()])

        mocks.save_state.assert_not_called()
        for downloader in mocks.downloaders():
            downloader.assert_not_called()

    def test_state_save_failure_of_success_is_reported_as_failure(self):
        with PipelineMocks() as mocks:
            mocks.save_state.side_effect = [RuntimeError("disk full"), None]
            with self.assertLogs(pipeline.logger, level="ERROR"):
                summary = pipeline.process_enriched_announcements([make_eauction()])

        self.assertEqual(summary["failed_count"], 1)
        self.assertEqual(summary["success_count"], 0)


class ProcessingStateIntegrationTests(unittest.TestCase):
    """Pipeline + реальные repositories на временной БД; мокаются только сеть и парсер."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "test.db"
        enrichment_repository.init_db(self.db_path)
        document_repository.init_db(self.db_path)
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)

    def add(self, url, resource_type, **enrichment) -> dict:
        announcement_repository.save_announcement({
            "title": "Тест", "source_section": "s", "source_section_name": "S",
            "source_page_url": "https://example.test/list", "resource_url": url,
            "resource_type": resource_type,
        }, self.db_path)
        enrichment_repository.save_enrichment(
            url, {"enrichment_status": "not_required", **enrichment}, self.db_path,
        )
        return enrichment_repository.get_announcement_with_enrichment(url, self.db_path)

    def state(self, combined) -> dict:
        return document_repository.get_processing_state(combined["resource_url"], self.db_path)

    def candidate_urls(self) -> list[str]:
        candidates = document_repository.get_document_processing_candidates(self.db_path)
        return [c["resource_url"] for c in candidates]

    def test_direct_doc_gets_no_supported_document_and_leaves_pending(self):
        combined = self.add("https://files.example.test/old.doc", "direct_file")
        self.assertEqual(self.candidate_urls(), [combined["resource_url"]])

        with mock.patch.object(pipeline, "download_direct_file") as download:
            summary = pipeline.process_enriched_announcements([combined], db_path=self.db_path)

        download.assert_not_called()
        self.assertEqual(summary["no_supported_document_count"], 1)
        self.assertEqual(self.state(combined)["status"], "no_supported_document")
        self.assertEqual(self.candidate_urls(), [])

    def test_successful_docx_gets_success_state_and_leaves_pending(self):
        combined = self.add("https://files.example.test/new.docx", "direct_file")
        download = {
            "source_url": combined["resource_url"], "saved_path": "data/documents/x/new.docx",
            "filename": "new.docx", "content_type": None, "size_bytes": 5, "sha256": "f" * 64,
        }
        docx = {"file_type": "docx", "text": "abc", "char_count": 3,
                "paragraph_count": 1, "table_count": 0}
        with mock.patch.object(pipeline, "download_direct_file", return_value=download), \
                mock.patch.object(pipeline, "extract_docx", return_value=docx):
            summary = pipeline.process_enriched_announcements([combined], db_path=self.db_path)

        self.assertEqual(summary["success_count"], 1)
        state = self.state(combined)
        self.assertEqual(state["status"], "success")
        self.assertEqual(state["source_kind"], "direct_file")
        self.assertEqual(state["source_ref"], combined["resource_url"])
        self.assertEqual(self.candidate_urls(), [])

    def test_exception_gets_failed_state_and_stays_pending_until_success(self):
        combined = self.add("https://files.example.test/new.docx", "direct_file")
        with mock.patch.object(pipeline, "download_direct_file", side_effect=OSError("timeout")):
            summary = pipeline.process_enriched_announcements([combined], db_path=self.db_path)

        self.assertEqual(summary["failed_count"], 1)
        state = self.state(combined)
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["error_type"], "OSError")
        self.assertEqual(state["error_message"], "timeout")
        self.assertEqual(self.candidate_urls(), [combined["resource_url"]])

        # Повторная попытка успешна: failed заменяется success, ошибка очищается.
        download = {
            "source_url": combined["resource_url"], "saved_path": "data/documents/x/new.docx",
            "filename": "new.docx", "content_type": None, "size_bytes": 5, "sha256": "f" * 64,
        }
        docx = {"file_type": "docx", "text": "abc", "char_count": 3,
                "paragraph_count": 1, "table_count": 0}
        with mock.patch.object(pipeline, "download_direct_file", return_value=download), \
                mock.patch.object(pipeline, "extract_docx", return_value=docx):
            pipeline.process_enriched_announcements([combined], db_path=self.db_path)

        state = self.state(combined)
        self.assertEqual(state["status"], "success")
        self.assertIsNone(state["error_type"])
        self.assertEqual(self.candidate_urls(), [])

    def test_one_failure_does_not_stop_others_and_states_are_independent(self):
        bad = self.add("https://files.example.test/bad.docx", "direct_file")
        skipped = self.add("https://files.example.test/skip.pdf", "direct_file")
        with mock.patch.object(pipeline, "download_direct_file", side_effect=OSError("boom")):
            pipeline.process_enriched_announcements([bad, skipped], db_path=self.db_path)

        self.assertEqual(self.state(bad)["status"], "failed")
        self.assertEqual(self.state(skipped)["status"], "no_supported_document")
        self.assertEqual(self.candidate_urls(), [bad["resource_url"]])


class XlsxExtractionIntegrationTests(unittest.TestCase):
    """Pipeline + реальные extractor и repository на временных файле и БД; мокается только сеть."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.db_path = self.tmp / "test.db"
        enrichment_repository.init_db(self.db_path)
        document_repository.init_db(self.db_path)
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)

    def add(self, url, resource_type, **enrichment) -> dict:
        announcement_repository.save_announcement({
            "title": "Тест", "source_section": "s", "source_section_name": "S",
            "source_page_url": "https://example.test/list", "resource_url": url,
            "resource_type": resource_type,
        }, self.db_path)
        enrichment_repository.save_enrichment(
            url, {"enrichment_status": "not_required", **enrichment}, self.db_path,
        )
        return enrichment_repository.get_announcement_with_enrichment(url, self.db_path)

    def write_download(self, filename, data, sha256) -> dict:
        path = self.tmp / filename
        path.write_bytes(data)
        return {
            "source_url": "https://example.test/" + filename, "saved_path": str(path),
            "filename": filename, "content_type": None, "size_bytes": len(data), "sha256": sha256,
        }

    def extractions(self, combined) -> dict:
        [download] = document_repository.get_downloads_for_resource(
            combined["resource_url"], self.db_path,
        )
        rows = document_repository.get_extractions_for_download(download["id"], self.db_path)
        return {row["member_name"]: row for row in rows}

    def test_direct_xlsx_is_extracted_and_saved_with_metrics(self):
        combined = self.add("https://files.example.test/spec.xlsx", "direct_file")
        data = make_xlsx_bytes({"Lot": [["Item", "Qty"], ["Bolt", 10]]})
        download = self.write_download("spec.xlsx", data, "a" * 64)

        with mock.patch.object(pipeline, "download_direct_file", return_value=download):
            summary = pipeline.process_enriched_announcements([combined], db_path=self.db_path)

        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["failed_count"], 0)
        self.assertEqual(summary["extraction_new_count"], 1)
        row = self.extractions(combined)[""]
        self.assertEqual(row["file_type"], "xlsx")
        self.assertEqual(row["text"], "[Sheet: Lot]\nItem\tQty\nBolt\t10")
        self.assertEqual((row["sheet_count"], row["row_count"], row["cell_count"]), (1, 2, 4))

    def test_eauction_zip_with_xlsx_and_nested_zip(self):
        combined = self.add(
            "https://eauction.example.test/tender/1", "eauction_tender_page",
            document_url=EAUCTION_ZIP_URL,
        )
        nested = make_zip_bytes([("spec.xlsx", make_xlsx_bytes({"Lot": [["nested", 1]]}))])
        data = make_zip_bytes([
            ("hraver.docx", make_docx_bytes(["Հրավեր"])),
            ("lot_1_i_texnikakan_bnutagir.xlsx", make_xlsx_bytes({"Lot": [["lot 1"]]})),
            ("lot_2.zip", nested),
            ("archive.rar", b"Rar!"),
        ])
        download = self.write_download("tender_1.zip", data, "b" * 64)

        with mock.patch.object(pipeline, "download_eauction_document", return_value=download):
            first = pipeline.process_enriched_announcements([combined], db_path=self.db_path)
            second = pipeline.process_enriched_announcements([combined], db_path=self.db_path)

        self.assertEqual(first["success_count"], 1)
        self.assertEqual(first["failed_count"], 0)
        self.assertEqual(first["download_new_count"], 1)
        self.assertEqual(first["extraction_new_count"], 4)
        self.assertEqual(first["extraction_updated_count"], 0)
        self.assertEqual(first["extraction_existing_count"], 0)

        # Повторный запуск: тот же sha256, ничего не меняется и дубли не создаются.
        self.assertEqual(second["download_existing_count"], 1)
        self.assertEqual(second["extraction_new_count"], 0)
        self.assertEqual(second["extraction_updated_count"], 0)
        self.assertEqual(second["extraction_existing_count"], 4)
        self.assertEqual(document_repository.count_downloads(self.db_path), 1)
        self.assertEqual(document_repository.count_extractions(self.db_path), 4)

        rows = self.extractions(combined)
        self.assertEqual(rows["hraver.docx"]["extraction_status"], "success")
        self.assertEqual(rows["hraver.docx"]["paragraph_count"], 1)
        self.assertEqual(rows["lot_1_i_texnikakan_bnutagir.xlsx"]["text"], "[Sheet: Lot]\nlot 1")
        self.assertEqual(rows["lot_2.zip!/spec.xlsx"]["cell_count"], 2)
        self.assertEqual(rows["archive.rar"]["extraction_status"], "skipped")
        self.assertEqual(
            document_repository.get_processing_state(combined["resource_url"], self.db_path)["status"],
            "success",
        )

    def test_broken_nested_zip_does_not_fail_tender(self):
        combined = self.add(
            "https://eauction.example.test/tender/2", "eauction_tender_page",
            document_url=EAUCTION_ZIP_URL,
        )
        data = make_zip_bytes([
            ("broken.zip", b"not a zip"),
            ("spec.xlsx", make_xlsx_bytes({"Lot": [["ok"]]})),
        ])
        download = self.write_download("tender_2.zip", data, "c" * 64)

        with mock.patch.object(pipeline, "download_eauction_document", return_value=download):
            summary = pipeline.process_enriched_announcements([combined], db_path=self.db_path)

        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["failed_count"], 0)
        rows = self.extractions(combined)
        self.assertEqual(rows["broken.zip"]["extraction_status"], "failed")
        self.assertEqual(rows["spec.xlsx"]["extraction_status"], "success")

    def test_counters_across_mixed_announcements(self):
        xlsx = self.add("https://files.example.test/spec.xlsx", "direct_file")
        rar = self.add("https://files.example.test/docs.rar", "direct_file")
        bad = self.add("https://files.example.test/bad.xlsx", "direct_file")
        download = self.write_download(
            "spec.xlsx", make_xlsx_bytes({"Lot": [["x"]]}), "d" * 64,
        )

        def fake_download(url, resource_url, root_dir=None):
            if resource_url == bad["resource_url"]:
                raise OSError("timeout")
            return download

        with mock.patch.object(pipeline, "download_direct_file", side_effect=fake_download):
            summary = pipeline.process_enriched_announcements(
                [xlsx, rar, bad], db_path=self.db_path,
            )

        self.assertEqual(summary["processed_count"], 3)
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["no_supported_document_count"], 1)
        self.assertEqual(summary["failed_count"], 1)
        self.assertEqual(summary["download_new_count"], 1)
        self.assertEqual(summary["extraction_new_count"], 1)
        self.assertEqual(
            document_repository.get_processing_state(bad["resource_url"], self.db_path)["status"],
            "failed",
        )


class MainTests(unittest.TestCase):
    def test_main_only_prints_message(self):
        with PipelineMocks() as mocks, mock.patch("builtins.print") as fake_print:
            pipeline.main()

        fake_print.assert_called_once_with(
            "Document pipeline module. Use explicit processing functions."
        )
        for downloader in mocks.downloaders():
            downloader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
