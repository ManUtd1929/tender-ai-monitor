"""
Тесты document_downloader без сети: HTTP-ответы и сессии замоканы,
файлы пишутся во временную директорию.

Запуск из корня проекта:
    python -m unittest tests.test_document_downloader -v
"""

import base64
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import requests

from src.parser import document_downloader as downloader

RESOURCE_URL = "https://armeps.am/epps/cft/listContractDocuments.do?resourceId=12486627"
EAUCTION_RESOURCE_URL = "https://eauction.armeps.am/hy/public/tender_details/tmid/abc"
EAUCTION_ZIP_URL = (
    "https://eauction.armeps.am/application/documents/public_invitation/tender_12345.zip"
)
CONTENT = b"document bytes \x00\x01\x02"


class FakeResponse:
    def __init__(self, chunks=(CONTENT,), headers=None):
        self._chunks = list(chunks)
        self.headers = headers or {}
        self.body_read = False
        self.closed = False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=1):
        self.body_read = True
        for chunk in self._chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    def close(self):
        self.closed = True


def make_session(*responses):
    session = mock.Mock()
    session.get.side_effect = list(responses)
    return session


class DownloaderTestCase(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def files_in_document_dir(self, resource_url=RESOURCE_URL):
        directory = downloader.build_document_directory(resource_url, self.root)
        if not directory.exists():
            return []
        return sorted(p.name for p in directory.iterdir())


class DirectoryTests(DownloaderTestCase):
    def test_same_resource_url_gives_stable_directory(self):
        first = downloader.build_document_directory(RESOURCE_URL, self.root)
        second = downloader.build_document_directory(RESOURCE_URL, self.root)
        expected = hashlib.sha256(RESOURCE_URL.encode("utf-8")).hexdigest()[:16]
        self.assertEqual(first, second)
        self.assertEqual(first, self.root / expected)

    def test_different_resource_urls_give_different_directories(self):
        self.assertNotEqual(
            downloader.build_document_directory(RESOURCE_URL, self.root),
            downloader.build_document_directory(EAUCTION_RESOURCE_URL, self.root),
        )

    def test_default_root_is_data_documents(self):
        directory = downloader.build_document_directory(RESOURCE_URL)
        self.assertEqual(directory.parent.parts[-2:], ("data", "documents"))


class SanitizeFilenameTests(unittest.TestCase):
    def test_removes_directory_components_and_traversal(self):
        self.assertEqual(downloader.sanitize_filename("../../etc/passwd.txt"), "passwd.txt")
        self.assertEqual(downloader.sanitize_filename("..\\..\\win\\file.docx"), "file.docx")
        self.assertEqual(downloader.sanitize_filename(".."), "document")

    def test_windows_invalid_chars_replaced(self):
        result = downloader.sanitize_filename('a<b>c:d"e|f?g*h.pdf')
        self.assertEqual(result, "a_b_c_d_e_f_g_h.pdf")

    def test_control_chars_removed_and_trailing_dots_spaces_stripped(self):
        self.assertEqual(downloader.sanitize_filename("fi\x00le\n.docx"), "file.docx")
        self.assertEqual(downloader.sanitize_filename("report. . "), "report")

    def test_empty_name_becomes_document(self):
        self.assertEqual(downloader.sanitize_filename(""), "document")
        self.assertEqual(downloader.sanitize_filename("  . "), "document")
        self.assertEqual(downloader.sanitize_filename("folder/"), "document")

    def test_long_name_truncated_keeping_extension(self):
        result = downloader.sanitize_filename("a" * 500 + ".docx")
        self.assertLessEqual(len(result), downloader.MAX_FILENAME_LENGTH)
        self.assertTrue(result.endswith(".docx"))

    def test_unicode_name_preserved(self):
        self.assertEqual(downloader.sanitize_filename("Հրավեր rus.docx"), "Հրավեր rus.docx")

    def test_windows_reserved_name_prefixed(self):
        self.assertEqual(downloader.sanitize_filename("NUL.txt"), "_NUL.txt")


class ContentDispositionTests(unittest.TestCase):
    def test_plain_filename(self):
        headers = {"Content-Disposition": 'attachment; filename="file.docx"'}
        self.assertEqual(downloader.extract_filename_from_headers(headers), "file.docx")

    def test_header_name_case_insensitive(self):
        headers = {"content-disposition": "attachment; filename=file.pdf"}
        self.assertEqual(downloader.extract_filename_from_headers(headers), "file.pdf")

    def test_mime_encoded_filename(self):
        encoded = base64.b64encode("документ.docx".encode("utf-8")).decode("ascii")
        headers = {"Content-Disposition": f'attachment; filename="=?UTF8?B?{encoded}?="'}
        self.assertEqual(downloader.extract_filename_from_headers(headers), "документ.docx")

    def test_rfc5987_filename_star(self):
        headers = {"Content-Disposition": "attachment; filename*=UTF-8''%D0%B4%D0%BE%D0%BA.pdf"}
        self.assertEqual(downloader.extract_filename_from_headers(headers), "док.pdf")

    def test_utf8_bytes_decoded_as_latin1_are_restored(self):
        raw = "Հրավեր.docx".encode("utf-8").decode("latin-1")
        headers = {"Content-Disposition": f'attachment; filename="{raw}"'}
        self.assertEqual(downloader.extract_filename_from_headers(headers), "Հրավեր.docx")

    def test_missing_filename_gives_none(self):
        self.assertIsNone(downloader.extract_filename_from_headers({}))
        self.assertIsNone(downloader.extract_filename_from_headers({"Content-Disposition": "attachment"}))


class DirectFileTests(DownloaderTestCase):
    def test_saves_bytes_and_returns_metadata(self):
        response = FakeResponse(
            chunks=[b"abc", b"def"],
            headers={
                "Content-Disposition": 'attachment; filename="file.docx"',
                "Content-Type": "application/msword",
            },
        )
        session = make_session(response)

        result = downloader.download_direct_file(
            "https://example.com/files/other.docx", RESOURCE_URL, self.root, session,
        )

        saved = Path(result["saved_path"])
        self.assertEqual(saved.read_bytes(), b"abcdef")
        self.assertEqual(saved.parent, downloader.build_document_directory(RESOURCE_URL, self.root))
        self.assertEqual(result["filename"], "file.docx")
        self.assertEqual(result["content_type"], "application/msword")
        self.assertEqual(result["size_bytes"], 6)
        self.assertEqual(result["source_url"], "https://example.com/files/other.docx")
        self.assertEqual(self.files_in_document_dir(), ["file.docx"])

    def test_sha256_matches_saved_bytes(self):
        session = make_session(FakeResponse(chunks=[b"abc", b"def"]))

        result = downloader.download_direct_file(
            "https://example.com/a.bin", RESOURCE_URL, self.root, session,
        )

        self.assertEqual(result["sha256"], hashlib.sha256(Path(result["saved_path"]).read_bytes()).hexdigest())
        self.assertEqual(result["sha256"], hashlib.sha256(b"abcdef").hexdigest())

    def test_url_basename_used_without_content_disposition(self):
        session = make_session(FakeResponse())

        result = downloader.download_direct_file(
            "https://example.com/files/My%20Tender%20%D5%80.pdf?x=1", RESOURCE_URL, self.root, session,
        )

        self.assertEqual(result["filename"], "My Tender Հ.pdf")

    def test_no_name_anywhere_gives_document(self):
        session = make_session(FakeResponse())

        result = downloader.download_direct_file("https://example.com/", RESOURCE_URL, self.root, session)

        self.assertEqual(result["filename"], "document")

    def test_request_parameters(self):
        session = make_session(FakeResponse())

        downloader.download_direct_file("https://example.com/a.bin", RESOURCE_URL, self.root, session)

        args, kwargs = session.get.call_args
        self.assertEqual(args[0], "https://example.com/a.bin")
        self.assertTrue(kwargs["stream"])
        self.assertIs(kwargs["verify"], True)
        self.assertEqual(kwargs["timeout"], downloader.TIMEOUT)
        self.assertIn("User-Agent", kwargs["headers"])

    def test_own_session_is_closed(self):
        session = make_session(FakeResponse())
        with mock.patch.object(downloader.requests, "Session", return_value=session):
            downloader.download_direct_file("https://example.com/a.bin", RESOURCE_URL, self.root)

        session.close.assert_called_once()

    def test_own_session_closed_on_error(self):
        session = mock.Mock()
        session.get.side_effect = requests.exceptions.ConnectionError("boom")
        with mock.patch.object(downloader.requests, "Session", return_value=session):
            with self.assertRaises(requests.exceptions.ConnectionError):
                downloader.download_direct_file("https://example.com/a.bin", RESOURCE_URL, self.root)

        session.close.assert_called_once()

    def test_passed_session_is_not_closed(self):
        session = make_session(FakeResponse())

        downloader.download_direct_file("https://example.com/a.bin", RESOURCE_URL, self.root, session)

        session.close.assert_not_called()


class EauctionTests(DownloaderTestCase):
    def test_zip_saved_as_is_without_unpacking(self):
        zip_bytes = b"PK\x03\x04 fake zip bytes"
        session = make_session(FakeResponse(chunks=[zip_bytes], headers={"Content-Type": "application/zip"}))

        result = downloader.download_eauction_document(
            EAUCTION_ZIP_URL, EAUCTION_RESOURCE_URL, self.root, session,
        )

        self.assertEqual(result["filename"], "tender_12345.zip")
        self.assertEqual(Path(result["saved_path"]).read_bytes(), zip_bytes)
        self.assertEqual(self.files_in_document_dir(EAUCTION_RESOURCE_URL), ["tender_12345.zip"])


class ArmepsTests(DownloaderTestCase):
    def run_armeps(self, final_response, **kwargs):
        session = make_session(FakeResponse(), FakeResponse(), final_response)
        with mock.patch.object(downloader.requests, "Session", return_value=session):
            result = downloader.download_armeps_document(
                RESOURCE_URL, "12486728", root_dir=self.root, **kwargs,
            )
        return result, session

    def test_page_prepare_final_sequence(self):
        final = FakeResponse(headers={"Content-Disposition": 'attachment; filename="x.docx"'})

        result, session = self.run_armeps(final)

        calls = session.get.call_args_list
        self.assertEqual(len(calls), 3)
        page_url, prepare_url, final_url = (call.args[0] for call in calls)
        self.assertEqual(page_url, RESOURCE_URL)
        self.assertEqual(
            prepare_url,
            "https://armeps.am/epps/cft/prepareAnonymousDownload.do?documentId=12486728",
        )
        self.assertEqual(
            final_url,
            "https://armeps.am/epps/cft/downloadContractDocument.do?documentId=12486728&resourceId=null",
        )
        self.assertEqual(calls[1].kwargs["headers"]["Referer"], RESOURCE_URL)
        self.assertEqual(calls[2].kwargs["headers"]["Referer"], prepare_url)
        for call in calls:
            self.assertIs(call.kwargs["verify"], True)
            self.assertEqual(call.kwargs["timeout"], downloader.TIMEOUT)
            self.assertIn("User-Agent", call.kwargs["headers"])
        self.assertEqual(result["source_url"], final_url)
        session.close.assert_called_once()

    def test_content_disposition_filename_used(self):
        final = FakeResponse(headers={"Content-Disposition": 'attachment; filename="from_header.docx"'})

        result, _ = self.run_armeps(final, expected_filename="expected.docx")

        self.assertEqual(result["filename"], "from_header.docx")

    def test_expected_filename_fallback(self):
        result, _ = self.run_armeps(FakeResponse(), expected_filename="expected.docx")

        self.assertEqual(result["filename"], "expected.docx")

    def test_document_id_fallback_filename(self):
        result, _ = self.run_armeps(FakeResponse())

        self.assertEqual(result["filename"], "document_12486728")

    def test_html_final_response_raises_and_saves_nothing(self):
        final = FakeResponse(headers={"Content-Type": "text/html;charset=UTF-8"})

        with self.assertRaisesRegex(ValueError, "Expected file download but received HTML"):
            self.run_armeps(final, expected_filename="x.docx")

        self.assertEqual(self.files_in_document_dir(), [])
        self.assertFalse(final.body_read)
        self.assertTrue(final.closed)

    def test_octet_stream_allowed(self):
        final = FakeResponse(headers={"Content-Type": "application/octet-stream"})

        result, _ = self.run_armeps(final, expected_filename="x.docx")

        self.assertEqual(result["content_type"], "application/octet-stream")
        self.assertEqual(self.files_in_document_dir(), ["x.docx"])

    def test_session_closed_when_download_fails(self):
        session = mock.Mock()
        session.get.side_effect = requests.exceptions.Timeout("slow")
        with mock.patch.object(downloader.requests, "Session", return_value=session):
            with self.assertRaises(requests.exceptions.Timeout):
                downloader.download_armeps_document(RESOURCE_URL, "1", root_dir=self.root)

        session.close.assert_called_once()


class HtmlValidationTests(DownloaderTestCase):
    def test_html_response_for_html_file_is_allowed(self):
        session = make_session(FakeResponse(headers={"Content-Type": "text/html"}))

        result = downloader.download_direct_file(
            "https://example.com/page.html", RESOURCE_URL, self.root, session,
        )

        self.assertEqual(result["filename"], "page.html")


class MaxSizeTests(DownloaderTestCase):
    def test_content_length_over_limit_raises_without_writing(self):
        response = FakeResponse(headers={"Content-Length": "100"})
        session = make_session(response)

        with self.assertRaises(ValueError):
            downloader.download_direct_file(
                "https://example.com/a.bin", RESOURCE_URL, self.root, session, max_bytes=10,
            )

        self.assertFalse(response.body_read)
        self.assertEqual(self.files_in_document_dir(), [])

    def test_streaming_over_limit_raises_and_removes_part(self):
        session = make_session(FakeResponse(chunks=[b"123456", b"123456"]))

        with self.assertRaises(ValueError):
            downloader.download_direct_file(
                "https://example.com/a.bin", RESOURCE_URL, self.root, session, max_bytes=10,
            )

        self.assertEqual(self.files_in_document_dir(), [])

    def test_exact_limit_is_allowed(self):
        session = make_session(FakeResponse(chunks=[b"1234567890"]))

        result = downloader.download_direct_file(
            "https://example.com/a.bin", RESOURCE_URL, self.root, session, max_bytes=10,
        )

        self.assertEqual(result["size_bytes"], 10)

    def test_default_limit_is_50_mb(self):
        self.assertEqual(downloader.DEFAULT_MAX_BYTES, 50 * 1024 * 1024)


class AtomicWriteTests(DownloaderTestCase):
    def test_part_file_removed_when_download_fails_midway(self):
        session = make_session(
            FakeResponse(chunks=[b"partial", requests.exceptions.ChunkedEncodingError("cut")]),
        )

        with self.assertRaises(requests.exceptions.ChunkedEncodingError):
            downloader.download_direct_file(
                "https://example.com/a.bin", RESOURCE_URL, self.root, session,
            )

        self.assertEqual(self.files_in_document_dir(), [])

    def test_no_part_file_left_after_success(self):
        session = make_session(FakeResponse())

        downloader.download_direct_file("https://example.com/a.bin", RESOURCE_URL, self.root, session)

        self.assertEqual(self.files_in_document_dir(), ["a.bin"])

    def test_response_is_closed(self):
        response = FakeResponse()
        session = make_session(response)

        downloader.download_direct_file("https://example.com/a.bin", RESOURCE_URL, self.root, session)

        self.assertTrue(response.closed)


class CollisionTests(DownloaderTestCase):
    def download(self, content):
        session = make_session(FakeResponse(chunks=[content]))
        return downloader.download_direct_file(
            "https://example.com/tender.docx", RESOURCE_URL, self.root, session,
        )

    def test_identical_existing_file_not_duplicated(self):
        first = self.download(b"same bytes")
        second = self.download(b"same bytes")

        self.assertEqual(second["saved_path"], first["saved_path"])
        self.assertEqual(second["sha256"], first["sha256"])
        self.assertEqual(self.files_in_document_dir(), ["tender.docx"])

    def test_different_bytes_create_versioned_filename_and_keep_old(self):
        first = self.download(b"old version")
        second = self.download(b"new version")

        expected_hash = hashlib.sha256(b"new version").hexdigest()[:8]
        self.assertEqual(second["filename"], f"tender__{expected_hash}.docx")
        self.assertEqual(Path(first["saved_path"]).read_bytes(), b"old version")
        self.assertEqual(Path(second["saved_path"]).read_bytes(), b"new version")
        self.assertEqual(len(self.files_in_document_dir()), 2)

    def test_same_versioned_file_not_duplicated(self):
        self.download(b"old version")
        second = self.download(b"new version")
        third = self.download(b"new version")

        self.assertEqual(third["saved_path"], second["saved_path"])
        self.assertEqual(len(self.files_in_document_dir()), 2)


class ModuleTests(unittest.TestCase):
    def test_main_prints_message_and_does_no_http(self):
        with mock.patch.object(downloader.requests, "Session") as session_cls, \
                mock.patch("builtins.print") as fake_print:
            downloader.main()

        fake_print.assert_called_once_with("Document downloader module. Use explicit download functions.")
        session_cls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
