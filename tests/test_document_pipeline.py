"""
Тесты document_pipeline. Сеть, SQLite и реальные файлы не используются:
download_*, extract_* и функции document_repository замоканы.

Запуск из корня проекта:
    python -m unittest tests.test_document_pipeline -v
"""

import copy
import unittest
from unittest import mock

from src import document_pipeline as pipeline

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
                mock.patch.object(pipeline, "extract_docx_from_zip") as extract_zip:
            result = pipeline.extract_downloaded_file(download)

        extract_docx.assert_called_once_with(download["saved_path"])
        extract_zip.assert_not_called()
        self.assertEqual(result, {"extraction_type": "docx", "result": extract_docx.return_value})

    def test_zip_routing(self):
        download = make_download_result("tender.zip")
        with mock.patch.object(pipeline, "extract_docx") as extract_docx, \
                mock.patch.object(pipeline, "extract_docx_from_zip") as extract_zip:
            result = pipeline.extract_downloaded_file(download)

        extract_zip.assert_called_once_with(download["saved_path"])
        extract_docx.assert_not_called()
        self.assertEqual(result, {"extraction_type": "zip", "result": extract_zip.return_value})

    def test_unsupported_file_raises(self):
        with mock.patch.object(pipeline, "extract_docx") as extract_docx, \
                mock.patch.object(pipeline, "extract_docx_from_zip") as extract_zip:
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
        self.extract_zip = patch("extract_docx_from_zip", return_value={"documents": []})
        self.init_db = patch_repo("init_db")
        self.save_download = patch_repo(
            "save_download", return_value={"status": self.storage_status, "download_id": 7},
        )
        self.save_docx = patch_repo(
            "save_docx_extraction", return_value=dict(self.extraction_counts),
        )
        self.save_zip = patch_repo(
            "save_zip_extraction", return_value=dict(self.extraction_counts),
        )
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
