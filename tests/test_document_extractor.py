"""
Тесты document_extractor: DOCX и ZIP создаются на лету (python-docx, zipfile)
во временной директории или в памяти. Сеть не используется.

Запуск из корня проекта:
    python -m unittest tests.test_document_extractor -v
"""

import io
import os
import tempfile
import unittest
import zipfile
from pathlib import Path

from docx import Document
from docx.opc.exceptions import PackageNotFoundError

from src.parser import document_extractor as extractor

ARMENIAN = "Գնման առարկա՝ ապրանքների ուղեփոխադրում"
RUSSIAN = "Предмет закупки: международные грузоперевозки"

BROKEN_DOCX_ERRORS = (PackageNotFoundError, zipfile.BadZipFile)


def make_docx_bytes(paragraphs=(), tables=()):
    """tables — список таблиц, таблица — список строк, строка — список ячеек."""
    document = Document()
    for text in paragraphs:
        document.add_paragraph(text)
    for rows in tables:
        table = document.add_table(rows=len(rows), cols=len(rows[0]))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                table.cell(r, c).text = value
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def make_zip(path, members):
    """members — список (имя, bytes); имя, оканчивающееся на '/', — директория."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        for name, data in members:
            zf.writestr(name, data)


class ExtractDocxTests(unittest.TestCase):
    def test_single_paragraph(self):
        result = extractor.extract_docx(make_docx_bytes(["Hello tender"]))
        self.assertEqual(result["file_type"], "docx")
        self.assertEqual(result["text"], "Hello tender")
        self.assertEqual(result["paragraph_count"], 1)
        self.assertEqual(result["table_count"], 0)

    def test_multiple_paragraphs(self):
        result = extractor.extract_docx(make_docx_bytes(["first", "second", "third"]))
        self.assertEqual(result["text"], "first\nsecond\nthird")
        self.assertEqual(result["paragraph_count"], 3)

    def test_empty_paragraphs_do_not_create_garbage(self):
        result = extractor.extract_docx(make_docx_bytes(["first", "", "   ", "second"]))
        self.assertEqual(result["text"], "first\nsecond")
        self.assertEqual(result["paragraph_count"], 2)

    def test_armenian_unicode_preserved(self):
        result = extractor.extract_docx(make_docx_bytes([ARMENIAN]))
        self.assertEqual(result["text"], ARMENIAN)

    def test_russian_unicode_preserved(self):
        result = extractor.extract_docx(make_docx_bytes([RUSSIAN]))
        self.assertEqual(result["text"], RUSSIAN)

    def test_inner_whitespace_not_normalized(self):
        result = extractor.extract_docx(make_docx_bytes(["  a  b   c  "]))
        self.assertEqual(result["text"], "a  b   c")

    def test_table_extracted(self):
        result = extractor.extract_docx(make_docx_bytes(tables=[[["A", "B", "C"]]]))
        self.assertEqual(result["text"], "A\tB\tC")
        self.assertEqual(result["table_count"], 1)
        self.assertEqual(result["paragraph_count"], 0)

    def test_table_multiple_rows(self):
        rows = [["Name", "Qty"], ["Bolt", "10"], [ARMENIAN, RUSSIAN]]
        result = extractor.extract_docx(make_docx_bytes(tables=[rows]))
        self.assertEqual(result["text"], f"Name\tQty\nBolt\t10\n{ARMENIAN}\t{RUSSIAN}")

    def test_paragraph_and_table_both_present(self):
        result = extractor.extract_docx(
            make_docx_bytes(["Intro paragraph"], tables=[[["k", "v"]]])
        )
        self.assertIn("Intro paragraph", result["text"])
        self.assertIn("k\tv", result["text"])

    def test_table_with_empty_row_skipped(self):
        result = extractor.extract_docx(make_docx_bytes(tables=[[["a", "b"], ["", ""]]]))
        self.assertEqual(result["text"], "a\tb")

    def test_char_count_matches_text_length(self):
        result = extractor.extract_docx(
            make_docx_bytes([ARMENIAN, RUSSIAN], tables=[[["x", "y"]]])
        )
        self.assertEqual(result["char_count"], len(result["text"]))
        self.assertGreater(result["char_count"], 0)

    def test_bytes_source(self):
        result = extractor.extract_docx(make_docx_bytes(["from bytes"]))
        self.assertEqual(result["text"], "from bytes")

    def test_file_like_source(self):
        stream = io.BytesIO(make_docx_bytes(["from stream"]))
        result = extractor.extract_docx(stream)
        self.assertEqual(result["text"], "from stream")

    def test_path_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "doc.docx"
            path.write_bytes(make_docx_bytes(["from path"]))
            self.assertEqual(extractor.extract_docx(path)["text"], "from path")
            self.assertEqual(extractor.extract_docx(str(path))["text"], "from path")

    def test_empty_docx(self):
        result = extractor.extract_docx(make_docx_bytes())
        self.assertEqual(result["text"], "")
        self.assertEqual(result["char_count"], 0)
        self.assertEqual(result["paragraph_count"], 0)
        self.assertEqual(result["table_count"], 0)

    def test_broken_docx_raises(self):
        with self.assertRaises(BROKEN_DOCX_ERRORS):
            extractor.extract_docx(b"this is not a docx file")


class CleanTextPieceTests(unittest.TestCase):
    def test_strip_and_newlines(self):
        self.assertEqual(extractor.clean_text_piece("  a\r\nb\rc  "), "a\nb\nc")

    def test_inner_spaces_and_unicode_untouched(self):
        self.assertEqual(extractor.clean_text_piece(f" {ARMENIAN}  x "), f"{ARMENIAN}  x")


class ExtractDocxFromZipTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.zip_path = self.tmp / "tender.zip"

    def test_two_docx(self):
        make_zip(self.zip_path, [
            ("hraver.docx", make_docx_bytes([ARMENIAN])),
            ("hraver_ru.docx", make_docx_bytes([RUSSIAN], tables=[[["a", "b"]]])),
        ])
        result = extractor.extract_docx_from_zip(self.zip_path)
        self.assertEqual(result["file_type"], "zip")
        self.assertEqual(result["member_count"], 2)
        self.assertEqual(result["docx_count"], 2)
        self.assertEqual(result["skipped_members"], [])
        self.assertEqual(result["failures"], [])

    def test_texts_of_both_docx_extracted(self):
        make_zip(self.zip_path, [
            ("hraver.docx", make_docx_bytes([ARMENIAN])),
            ("hraver_ru.docx", make_docx_bytes([RUSSIAN])),
        ])
        docs = extractor.extract_docx_from_zip(self.zip_path)["documents"]
        by_name = {d["member_name"]: d for d in docs}
        self.assertEqual(by_name["hraver.docx"]["text"], ARMENIAN)
        self.assertEqual(by_name["hraver_ru.docx"]["text"], RUSSIAN)
        self.assertEqual(by_name["hraver.docx"]["file_type"], "docx")
        self.assertEqual(by_name["hraver.docx"]["char_count"], len(ARMENIAN))

    def test_unsupported_member_skipped(self):
        make_zip(self.zip_path, [
            ("a.docx", make_docx_bytes(["a"])),
            ("b.pdf", b"%PDF-1.4"),
            ("c.txt", b"text"),
            ("d.doc", b"\xd0\xcf"),
        ])
        result = extractor.extract_docx_from_zip(self.zip_path)
        self.assertEqual(result["docx_count"], 1)
        self.assertEqual(result["skipped_members"], ["b.pdf", "c.txt", "d.doc"])

    def test_directory_member_ignored(self):
        make_zip(self.zip_path, [
            ("folder/", b""),
            ("folder/a.docx", make_docx_bytes(["a"])),
        ])
        result = extractor.extract_docx_from_zip(self.zip_path)
        self.assertEqual(result["member_count"], 1)
        self.assertEqual(result["docx_count"], 1)
        self.assertEqual(result["skipped_members"], [])

    def test_uppercase_extension_supported(self):
        make_zip(self.zip_path, [("HRAVER.DOCX", make_docx_bytes(["upper"]))])
        result = extractor.extract_docx_from_zip(self.zip_path)
        self.assertEqual(result["docx_count"], 1)
        self.assertEqual(result["documents"][0]["text"], "upper")

    def test_too_many_members(self):
        make_zip(self.zip_path, [(f"{i}.txt", b"x") for i in range(3)])
        with self.assertRaises(ValueError):
            extractor.extract_docx_from_zip(self.zip_path, max_members=2)

    def test_member_too_large(self):
        make_zip(self.zip_path, [("big.bin", b"x" * 100)])
        with self.assertRaises(ValueError):
            extractor.extract_docx_from_zip(self.zip_path, max_member_bytes=50)

    def test_total_uncompressed_too_large(self):
        make_zip(self.zip_path, [("a.bin", b"x" * 60), ("b.bin", b"x" * 60)])
        with self.assertRaises(ValueError):
            extractor.extract_docx_from_zip(
                self.zip_path, max_member_bytes=100, max_total_uncompressed_bytes=100
            )

    def test_broken_docx_goes_to_failures_others_extracted(self):
        make_zip(self.zip_path, [
            ("broken.docx", b"not a real docx"),
            ("good.docx", make_docx_bytes([RUSSIAN])),
        ])
        result = extractor.extract_docx_from_zip(self.zip_path)
        self.assertEqual(result["docx_count"], 1)
        self.assertEqual(result["documents"][0]["member_name"], "good.docx")
        self.assertEqual(result["documents"][0]["text"], RUSSIAN)
        self.assertEqual(len(result["failures"]), 1)
        failure = result["failures"][0]
        self.assertEqual(failure["member_name"], "broken.docx")
        self.assertTrue(failure["error_type"])
        self.assertIn("error_message", failure)

    def test_zip_writes_nothing_to_filesystem(self):
        work = self.tmp / "work"
        work.mkdir()
        zip_path = work / "tender.zip"
        make_zip(zip_path, [
            ("../evil.docx", make_docx_bytes(["evil"])),
            ("/abs_evil.docx", make_docx_bytes(["abs"])),
            ("ok.docx", make_docx_bytes(["ok"])),
        ])
        before_tree = sorted(str(p) for p in self.tmp.rglob("*"))
        cwd_before = sorted(os.listdir("."))

        result = extractor.extract_docx_from_zip(zip_path)

        self.assertEqual(result["docx_count"], 3)
        self.assertEqual(sorted(str(p) for p in self.tmp.rglob("*")), before_tree)
        self.assertFalse((self.tmp / "evil.docx").exists())
        self.assertEqual(sorted(os.listdir(".")), cwd_before)
        # member_name сохранён как метаданные без изменений
        names = {d["member_name"] for d in result["documents"]}
        self.assertIn("../evil.docx", names)


if __name__ == "__main__":
    unittest.main()
