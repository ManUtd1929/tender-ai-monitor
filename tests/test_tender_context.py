"""
Тесты src.ai.tender_context на временной SQLite БД (реальные repository, без сети).

Запуск из корня проекта:
    python -m unittest tests.test_tender_context -v
"""

import copy
import tempfile
import unittest
from pathlib import Path

from src.ai import tender_context as ctx
from src.database import announcement_repository, document_repository, enrichment_repository

RESOURCE_URL = "https://example.test/resource/1"


def make_announcement(**overrides) -> dict:
    announcement = {
        "title": "Организация международных грузоперевозок",
        "source_section": "open_competition",
        "source_section_name": "Открытый конкурс",
        "source_page_url": "https://gnumner.minfin.am/ru/page/test/",
        "resource_url": RESOURCE_URL,
        "resource_type": "armeps_documents_page",
        "published_at": "2026-09-01 10:00:00",
        "deadline_at": "2026-09-15 18:00:00",
        "tender_time_raw": "raw",
    }
    announcement.update(overrides)
    return announcement


def make_enrichment(**overrides) -> dict:
    enrichment = {
        "enrichment_status": "success",
        "procedure_code": "PC-1",
        "contracting_authority": "Министерство финансов",
        "detail_title": "Перевозка грузов",
        "detail_title_ru": "Перевозка грузов",
        "detail_title_en": "Cargo transport",
        "procurement_type": "goods",
        "procedure_type": "open",
        "description": "Международная перевозка автомобильным транспортом",
        "estimated_value_amd": "1000000",
        "published_at_detail": "2026-09-01 10:00:00",
        "deadline_at_detail": "2026-09-15 18:00:00",
        "number_of_lots": 1,
        "resource_id": "1",
        "document_url": None,
        "cpv_codes": [{"code": "60100000", "name": "Road transport services"}],
        "documents": [
            {"document_id": "1", "filename": "hraver.docx", "language": "RU", "title": None, "description": None},
        ],
    }
    enrichment.update(overrides)
    return enrichment


def make_download_result(filename="hraver.docx", sha256="a" * 64, **overrides) -> dict:
    result = {
        "source_url": f"https://example.test/files/{filename}",
        "saved_path": f"data/documents/test/{filename}",
        "filename": filename,
        "content_type": None,
        "size_bytes": 10,
        "sha256": sha256,
    }
    result.update(overrides)
    return result


def make_docx_extraction(text="Организация перевозки груза", **overrides) -> dict:
    result = {"file_type": "docx", "text": text, "char_count": len(text), "paragraph_count": 1, "table_count": 0}
    result.update(overrides)
    return result


def make_xlsx_extraction(text="[Sheet: Lot]\nItem\tQty", **overrides) -> dict:
    result = {"file_type": "xlsx", "text": text, "char_count": len(text), "sheet_count": 1, "row_count": 1, "cell_count": 2}
    result.update(overrides)
    return result


class TenderContextTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "test.db"
        document_repository.init_db(self.db_path)
        enrichment_repository.init_db(self.db_path)

    def add_announcement(self, **overrides) -> str:
        announcement = make_announcement(**overrides)
        announcement_repository.save_announcement(announcement, self.db_path)
        return announcement["resource_url"]

    def add_enrichment(self, resource_url=RESOURCE_URL, **overrides) -> None:
        enrichment_repository.save_enrichment(resource_url, make_enrichment(**overrides), self.db_path)

    def add_download(self, resource_url=RESOURCE_URL, source_kind="armeps_document", source_ref="1", **overrides) -> int:
        result = document_repository.save_download(
            resource_url, source_kind, source_ref, make_download_result(**overrides), db_path=self.db_path,
        )
        return result["download_id"]


class BuildTenderContextTests(TenderContextTestCase):
    def test_missing_announcement_raises(self):
        with self.assertRaises(ValueError):
            ctx.build_tender_context("https://example.test/none", db_path=self.db_path)

    def test_announcement_fields(self):
        self.add_announcement()
        self.add_enrichment()

        context = ctx.build_tender_context(RESOURCE_URL, db_path=self.db_path)

        self.assertEqual(context["announcement"], {
            "title": "Организация международных грузоперевозок",
            "section": "Открытый конкурс",
            "resource_type": "armeps_documents_page",
            "published_at": "2026-09-01 10:00:00",
            "deadline_at": "2026-09-15 18:00:00",
            "resource_url": RESOURCE_URL,
        })

    def test_enrichment_fields(self):
        self.add_announcement()
        self.add_enrichment()

        enrichment = ctx.build_tender_context(RESOURCE_URL, db_path=self.db_path)["enrichment"]

        self.assertEqual(enrichment["contracting_authority"], "Министерство финансов")
        self.assertEqual(enrichment["procurement_type"], "goods")
        self.assertEqual(enrichment["procedure_type"], "open")
        self.assertEqual(enrichment["description"], "Международная перевозка автомобильным транспортом")
        self.assertEqual(enrichment["estimated_value_amd"], "1000000")
        self.assertEqual(enrichment["number_of_lots"], 1)
        self.assertEqual(enrichment["detail_titles"], {
            "default": "Перевозка грузов", "ru": "Перевозка грузов", "en": "Cargo transport",
        })
        self.assertEqual(enrichment["dates"], {
            "published_at_detail": "2026-09-01 10:00:00", "deadline_at_detail": "2026-09-15 18:00:00",
        })

    def test_cpv_codes(self):
        self.add_announcement()
        self.add_enrichment()

        cpv = ctx.build_tender_context(RESOURCE_URL, db_path=self.db_path)["enrichment"]["cpv_codes"]

        self.assertEqual(cpv, [{"code": "60100000", "name": "Road transport services"}])

    def test_no_documents_is_empty_and_complete(self):
        self.add_announcement()
        self.add_enrichment(documents=[])

        context = ctx.build_tender_context(RESOURCE_URL, db_path=self.db_path)

        self.assertEqual(context["documents"], [])
        self.assertTrue(context["document_coverage"]["coverage_complete"])

    def test_latest_download_only(self):
        self.add_announcement()
        self.add_enrichment()
        first_id = self.add_download()
        document_repository.save_docx_extraction(first_id, make_docx_extraction("first version"), self.db_path)
        second_id = self.add_download(sha256="b" * 64)
        document_repository.save_docx_extraction(second_id, make_docx_extraction("second version"), self.db_path)

        documents = ctx.build_tender_context(RESOURCE_URL, db_path=self.db_path)["documents"]

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]["download_id"], second_id)
        self.assertEqual(documents[0]["text"], "second version")

    def test_docx_entry_fields_and_metrics(self):
        self.add_announcement()
        self.add_enrichment()
        download_id = self.add_download()
        document_repository.save_docx_extraction(
            download_id, make_docx_extraction("груз", paragraph_count=3, table_count=1), self.db_path,
        )

        [document] = ctx.build_tender_context(RESOURCE_URL, db_path=self.db_path)["documents"]

        self.assertEqual(document["download_id"], download_id)
        self.assertEqual(document["member_name"], "")
        self.assertEqual(document["file_type"], "docx")
        self.assertEqual(document["extraction_status"], "success")
        self.assertEqual(document["text"], "груз")
        self.assertEqual(document["metrics"], {"char_count": 4, "paragraph_count": 3, "table_count": 1})

    def test_xlsx_entry_metrics_have_no_docx_keys(self):
        self.add_announcement()
        self.add_enrichment(documents=[
            {"document_id": "1", "filename": "spec.xlsx", "language": "RU", "title": None, "description": None},
        ])
        download_id = self.add_download(source_ref="1", filename="spec.xlsx")
        document_repository.save_xlsx_extraction(download_id, make_xlsx_extraction(), self.db_path)

        [document] = ctx.build_tender_context(RESOURCE_URL, db_path=self.db_path)["documents"]

        self.assertEqual(document["file_type"], "xlsx")
        self.assertEqual(document["metrics"], {"char_count": 21, "sheet_count": 1, "row_count": 1, "cell_count": 2})
        self.assertNotIn("paragraph_count", document["metrics"])

    def test_nested_zip_extraction_rows(self):
        self.add_announcement(resource_type="eauction_tender_page")
        self.add_enrichment(document_url="https://example.test/files/tender.zip", documents=[])
        download_id = self.add_download(
            source_kind="eauction_document", source_ref="https://example.test/files/tender.zip",
            filename="tender.zip",
        )
        document_repository.save_zip_extraction(download_id, {
            "file_type": "zip",
            "documents": [{"member_name": "lot_1.zip!/spec.xlsx", **make_xlsx_extraction()}],
            "skipped_members": ["scan.pdf"],
            "failures": [{"member_name": "broken.docx", "error_type": "BadZipFile", "error_message": "x"}],
        }, self.db_path)

        documents = ctx.build_tender_context(RESOURCE_URL, db_path=self.db_path)["documents"]
        by_name = {d["member_name"]: d for d in documents}

        self.assertEqual(by_name["lot_1.zip!/spec.xlsx"]["extraction_status"], "success")
        self.assertEqual(by_name["scan.pdf"]["extraction_status"], "skipped")
        self.assertEqual(by_name["broken.docx"]["extraction_status"], "failed")

    def test_coverage_complete_with_single_successful_docx(self):
        self.add_announcement()
        self.add_enrichment()
        download_id = self.add_download()
        document_repository.save_docx_extraction(download_id, make_docx_extraction(), self.db_path)

        coverage = ctx.build_tender_context(RESOURCE_URL, db_path=self.db_path)["document_coverage"]

        self.assertTrue(coverage["coverage_complete"])
        self.assertEqual(coverage["successful_extractions"], 1)
        self.assertEqual(coverage["successful_docx"], 1)
        self.assertEqual(coverage["failed_extractions"], 0)
        self.assertEqual(coverage["unsupported_extensions"], [])

    def test_coverage_incomplete_when_no_document_processed_yet(self):
        # Известный (enrichment) документ, но document_pipeline ещё не запускался.
        self.add_announcement()
        self.add_enrichment()

        coverage = ctx.build_tender_context(RESOURCE_URL, db_path=self.db_path)["document_coverage"]

        self.assertFalse(coverage["coverage_complete"])
        self.assertEqual(coverage["successful_extractions"], 0)

    def test_coverage_incomplete_when_processing_failed(self):
        self.add_announcement()
        self.add_enrichment()
        document_repository.save_processing_state(
            RESOURCE_URL, "failed", error_type="OSError", error_message="timeout", db_path=self.db_path,
        )

        coverage = ctx.build_tender_context(RESOURCE_URL, db_path=self.db_path)["document_coverage"]

        self.assertFalse(coverage["coverage_complete"])

    def test_coverage_incomplete_with_failed_extraction(self):
        self.add_announcement()
        self.add_enrichment()
        download_id = self.add_download()
        document_repository.save_extraction(
            download_id, {"member_name": "", "extraction_status": "failed",
                          "error_type": "BadZipFile", "error_message": "x"}, self.db_path,
        )

        coverage = ctx.build_tender_context(RESOURCE_URL, db_path=self.db_path)["document_coverage"]

        self.assertFalse(coverage["coverage_complete"])
        self.assertEqual(coverage["failed_extractions"], 1)

    def test_skipped_doc_sibling_makes_coverage_incomplete(self):
        # Заявлены два документа ARMEPS: выбранный .docx скачан и извлечён успешно,
        # но .doc с техническими спецификациями document_pipeline никогда не скачивает.
        self.add_announcement()
        self.add_enrichment(documents=[
            {"document_id": "1", "filename": "hraver.docx", "language": "RU", "title": None, "description": None},
            {"document_id": "2", "filename": "spec.doc", "language": "RU", "title": None, "description": None},
        ])
        download_id = self.add_download()
        document_repository.save_docx_extraction(download_id, make_docx_extraction(), self.db_path)

        coverage = ctx.build_tender_context(RESOURCE_URL, db_path=self.db_path)["document_coverage"]

        self.assertFalse(coverage["coverage_complete"])
        self.assertIn(".doc", coverage["unsupported_extensions"])
        self.assertEqual(coverage["successful_extractions"], 1)

    def test_skipped_zip_member_extension_recorded(self):
        self.add_announcement(resource_type="eauction_tender_page")
        self.add_enrichment(document_url="https://example.test/files/tender.zip", documents=[])
        download_id = self.add_download(
            source_kind="eauction_document", source_ref="https://example.test/files/tender.zip",
            filename="tender.zip",
        )
        document_repository.save_zip_extraction(download_id, {
            "file_type": "zip", "documents": [], "skipped_members": ["scan.pdf"], "failures": [],
        }, self.db_path)

        coverage = ctx.build_tender_context(RESOURCE_URL, db_path=self.db_path)["document_coverage"]

        self.assertFalse(coverage["coverage_complete"])
        self.assertEqual(coverage["unsupported_extensions"], [".pdf"])

    def test_total_extracted_chars_sums_only_successful(self):
        self.add_announcement()
        self.add_enrichment()
        download_id = self.add_download()
        document_repository.save_docx_extraction(download_id, make_docx_extraction("1234567890"), self.db_path)

        coverage = ctx.build_tender_context(RESOURCE_URL, db_path=self.db_path)["document_coverage"]

        self.assertEqual(coverage["total_extracted_chars"], 10)


class BuildTriageContextTests(TenderContextTestCase):
    def build(self, char_budget=None):
        context = ctx.build_tender_context(RESOURCE_URL, db_path=self.db_path)
        kwargs = {} if char_budget is None else {"char_budget": char_budget}
        return context, ctx.build_triage_context(context, **kwargs)

    def test_compact_fields(self):
        self.add_announcement()
        self.add_enrichment()
        _, triage_context = self.build()

        self.assertEqual(triage_context["title"], "Организация международных грузоперевозок")
        self.assertEqual(triage_context["resource_type"], "armeps_documents_page")
        self.assertEqual(triage_context["cpv_codes"], [{"code": "60100000", "name": "Road transport services"}])
        self.assertIn("document_coverage", triage_context)

    def test_budget_enforced_across_documents(self):
        self.add_announcement()
        self.add_enrichment(documents=[
            {"document_id": "1", "filename": "a.docx", "language": "RU", "title": None, "description": None},
        ])
        download_id = self.add_download()
        document_repository.save_docx_extraction(download_id, make_docx_extraction("x" * 100), self.db_path)

        _, triage_context = self.build(char_budget=30)

        self.assertEqual(sum(len(p["preview_text"]) for p in triage_context["document_previews"]), 30)
        self.assertTrue(triage_context["document_previews"][0]["truncated"])

    def test_budget_spans_multiple_documents(self):
        self.add_announcement(resource_type="eauction_tender_page")
        self.add_enrichment(document_url="https://example.test/files/tender.zip", documents=[])
        download_id = self.add_download(
            source_kind="eauction_document", source_ref="https://example.test/files/tender.zip",
            filename="tender.zip",
        )
        document_repository.save_zip_extraction(download_id, {
            "file_type": "zip",
            "documents": [
                {"member_name": "a.docx", **make_docx_extraction("a" * 20)},
                {"member_name": "b.docx", **make_docx_extraction("b" * 20)},
            ],
            "skipped_members": [], "failures": [],
        }, self.db_path)

        _, triage_context = self.build(char_budget=25)

        previews = {p["member_name"]: p for p in triage_context["document_previews"]}
        self.assertEqual(len(previews["a.docx"]["preview_text"]), 20)
        self.assertFalse(previews["a.docx"]["truncated"])
        self.assertEqual(len(previews["b.docx"]["preview_text"]), 5)
        self.assertTrue(previews["b.docx"]["truncated"])

    def test_failed_and_skipped_documents_have_no_preview(self):
        self.add_announcement(resource_type="eauction_tender_page")
        self.add_enrichment(document_url="https://example.test/files/tender.zip", documents=[])
        download_id = self.add_download(
            source_kind="eauction_document", source_ref="https://example.test/files/tender.zip",
            filename="tender.zip",
        )
        document_repository.save_zip_extraction(download_id, {
            "file_type": "zip", "documents": [],
            "skipped_members": ["scan.pdf"],
            "failures": [{"member_name": "broken.docx", "error_type": "E", "error_message": "m"}],
        }, self.db_path)

        _, triage_context = self.build()

        self.assertEqual(triage_context["document_previews"], [])
        self.assertEqual(len(triage_context["documents"]), 2)

    def test_provenance_preserved(self):
        self.add_announcement()
        self.add_enrichment()
        download_id = self.add_download()
        document_repository.save_docx_extraction(download_id, make_docx_extraction("text"), self.db_path)

        _, triage_context = self.build()

        [preview] = triage_context["document_previews"]
        self.assertEqual(preview["download_id"], download_id)
        self.assertEqual(preview["member_name"], "")
        self.assertEqual(preview["file_type"], "docx")

    def test_original_tender_context_not_modified(self):
        self.add_announcement()
        self.add_enrichment()
        download_id = self.add_download()
        document_repository.save_docx_extraction(download_id, make_docx_extraction("text"), self.db_path)

        context, _ = self.build(char_budget=2)

        original = ctx.build_tender_context(RESOURCE_URL, db_path=self.db_path)
        self.assertEqual(context, original)

    def test_invalid_char_budget_raises(self):
        self.add_announcement()
        self.add_enrichment(documents=[])
        context = ctx.build_tender_context(RESOURCE_URL, db_path=self.db_path)
        for budget in (0, -1, True):
            with self.subTest(budget=budget):
                with self.assertRaises(ValueError):
                    ctx.build_triage_context(context, char_budget=budget)


class ChunkTests(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        self.assertEqual(ctx._chunk_text("hello", 100), ["hello"])

    def test_no_text_lost_and_no_extra_text(self):
        text = "line one\nline two\nline three\n" * 20
        chunks = ctx._chunk_text(text, 37)
        self.assertEqual("".join(chunks), text)

    def test_prefers_newline_boundary(self):
        text = "aaaa\nbbbb\ncccc\n"
        chunks = ctx._chunk_text(text, 6)
        for chunk in chunks[:-1]:
            self.assertTrue(chunk.endswith("\n"))

    def test_hard_split_when_no_newline_in_window(self):
        text = "x" * 50
        chunks = ctx._chunk_text(text, 10)
        self.assertEqual(chunks, ["x" * 10] * 5)

    def test_deterministic(self):
        text = "a\n" * 500 + "b" * 123
        self.assertEqual(ctx._chunk_text(text, 17), ctx._chunk_text(text, 17))

    def test_invalid_chunk_size_raises(self):
        for size in (0, -1, True):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    ctx._chunk_text("text", size)


class BuildDeepAnalysisContextTests(TenderContextTestCase):
    def test_stable_chunk_ids(self):
        self.add_announcement()
        self.add_enrichment()
        download_id = self.add_download()
        document_repository.save_docx_extraction(download_id, make_docx_extraction("x" * 50), self.db_path)
        context = ctx.build_tender_context(RESOURCE_URL, db_path=self.db_path)

        first = ctx.build_deep_analysis_context(context, chunk_size=10)
        second = ctx.build_deep_analysis_context(context, chunk_size=10)

        self.assertEqual(
            [c["chunk_id"] for c in first["chunks"]], [c["chunk_id"] for c in second["chunks"]],
        )
        self.assertEqual(first["chunks"][0]["chunk_id"], f"{download_id}::0")

    def test_provenance_on_each_chunk(self):
        self.add_announcement()
        self.add_enrichment()
        download_id = self.add_download()
        document_repository.save_docx_extraction(download_id, make_docx_extraction("x" * 30), self.db_path)
        context = ctx.build_tender_context(RESOURCE_URL, db_path=self.db_path)

        deep_context = ctx.build_deep_analysis_context(context, chunk_size=10)

        for chunk in deep_context["chunks"]:
            self.assertEqual(chunk["download_id"], download_id)
            self.assertEqual(chunk["member_name"], "")
            self.assertEqual(chunk["file_type"], "docx")

    def test_no_text_lost_across_full_document(self):
        self.add_announcement()
        self.add_enrichment()
        download_id = self.add_download()
        text = "Условие поставки.\n" * 50
        document_repository.save_docx_extraction(download_id, make_docx_extraction(text), self.db_path)
        context = ctx.build_tender_context(RESOURCE_URL, db_path=self.db_path)

        deep_context = ctx.build_deep_analysis_context(context, chunk_size=40)

        self.assertEqual("".join(c["text"] for c in deep_context["chunks"]), text)

    def test_failed_and_skipped_documents_produce_no_chunks(self):
        self.add_announcement(resource_type="eauction_tender_page")
        self.add_enrichment(document_url="https://example.test/files/tender.zip", documents=[])
        download_id = self.add_download(
            source_kind="eauction_document", source_ref="https://example.test/files/tender.zip",
            filename="tender.zip",
        )
        document_repository.save_zip_extraction(download_id, {
            "file_type": "zip", "documents": [],
            "skipped_members": ["scan.pdf"],
            "failures": [{"member_name": "broken.docx", "error_type": "E", "error_message": "m"}],
        }, self.db_path)
        context = ctx.build_tender_context(RESOURCE_URL, db_path=self.db_path)

        deep_context = ctx.build_deep_analysis_context(context)

        self.assertEqual(deep_context["chunks"], [])

    def test_empty_text_produces_no_chunks(self):
        self.add_announcement()
        self.add_enrichment()
        download_id = self.add_download()
        document_repository.save_docx_extraction(download_id, make_docx_extraction(""), self.db_path)
        context = ctx.build_tender_context(RESOURCE_URL, db_path=self.db_path)

        deep_context = ctx.build_deep_analysis_context(context)

        self.assertEqual(deep_context["chunks"], [])

    def test_multiple_documents_get_independent_chunk_sequences(self):
        self.add_announcement(resource_type="eauction_tender_page")
        self.add_enrichment(document_url="https://example.test/files/tender.zip", documents=[])
        download_id = self.add_download(
            source_kind="eauction_document", source_ref="https://example.test/files/tender.zip",
            filename="tender.zip",
        )
        document_repository.save_zip_extraction(download_id, {
            "file_type": "zip",
            "documents": [
                {"member_name": "a.docx", **make_docx_extraction("a" * 25)},
                {"member_name": "b.docx", **make_docx_extraction("b" * 25)},
            ],
            "skipped_members": [], "failures": [],
        }, self.db_path)
        context = ctx.build_tender_context(RESOURCE_URL, db_path=self.db_path)

        deep_context = ctx.build_deep_analysis_context(context, chunk_size=10)

        a_ids = [c["chunk_id"] for c in deep_context["chunks"] if c["member_name"] == "a.docx"]
        b_ids = [c["chunk_id"] for c in deep_context["chunks"] if c["member_name"] == "b.docx"]
        self.assertEqual(a_ids, [f"{download_id}:a.docx:{i}" for i in range(len(a_ids))])
        self.assertEqual(b_ids, [f"{download_id}:b.docx:{i}" for i in range(len(b_ids))])


class InputHashTests(TenderContextTestCase):
    def build_hash(self):
        context = ctx.build_tender_context(RESOURCE_URL, db_path=self.db_path)
        return ctx.compute_input_hash(context)

    def test_stable_for_identical_context(self):
        self.add_announcement()
        self.add_enrichment()
        self.assertEqual(self.build_hash(), self.build_hash())

    def test_changes_on_title(self):
        self.add_announcement()
        self.add_enrichment()
        before = self.build_hash()

        announcement_repository.save_announcement(make_announcement(title="Другое название"), self.db_path)

        self.assertNotEqual(before, self.build_hash())

    def test_changes_on_enrichment(self):
        self.add_announcement()
        self.add_enrichment()
        before = self.build_hash()

        self.add_enrichment(description="Изменённое описание")

        self.assertNotEqual(before, self.build_hash())

    def test_changes_on_extraction_text(self):
        self.add_announcement()
        self.add_enrichment()
        download_id = self.add_download()
        document_repository.save_docx_extraction(download_id, make_docx_extraction("первая версия"), self.db_path)
        before = self.build_hash()

        document_repository.save_docx_extraction(download_id, make_docx_extraction("вторая версия"), self.db_path)

        self.assertNotEqual(before, self.build_hash())

    def test_ignores_volatile_fields(self):
        # last_seen_at (announcements) обновляется при каждом save_announcement того же
        # содержимого; хэш не должен от него зависеть.
        self.add_announcement()
        self.add_enrichment()
        before = self.build_hash()

        announcement_repository.save_announcement(make_announcement(), self.db_path)  # touch, без изменений

        self.assertEqual(before, self.build_hash())

    def test_does_not_mutate_context(self):
        self.add_announcement()
        self.add_enrichment()
        context = ctx.build_tender_context(RESOURCE_URL, db_path=self.db_path)
        original = copy.deepcopy(context)

        ctx.compute_input_hash(context)

        self.assertEqual(context, original)


if __name__ == "__main__":
    unittest.main()
