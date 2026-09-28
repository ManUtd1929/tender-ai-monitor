"""
Тесты src.ai.evaluation_dataset на временной SQLite БД. Проверяют, что модуль не
пишет в БД (byte-for-byte сравнение файла до/после) и что попытка использовать
production data/tenders.db блокируется общей test-only защитой (tests/safety_guards.py).

Запуск из корня проекта:
    python -m unittest tests.test_evaluation_dataset -v
"""

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from src.ai import evaluation_dataset as evald
from src.ai import tender_context as tender_context_module
from src.database import announcement_repository, document_repository, enrichment_repository
from tests import safety_guards


def make_announcement(number, **overrides) -> dict:
    announcement = {
        "title": f"Тендер {number}",
        "source_section": "open_competition",
        "source_section_name": "Открытый конкурс",
        "source_page_url": "https://gnumner.minfin.am/ru/page/test/",
        "resource_url": f"https://example.test/resource/{number}",
        "resource_type": "armeps_documents_page",
        "published_at": "2026-09-01 10:00:00",
        "deadline_at": "2026-09-15 18:00:00",
    }
    announcement.update(overrides)
    return announcement


def make_enrichment(**overrides) -> dict:
    enrichment = {
        "enrichment_status": "success",
        "contracting_authority": "Министерство финансов",
        "procurement_type": "goods",
        "procedure_type": "open",
        "description": "Тендер",
        "cpv_codes": [],
        "documents": [],
    }
    enrichment.update(overrides)
    return enrichment


def make_docx_extraction(text="текст", **overrides) -> dict:
    result = {"file_type": "docx", "text": text, "char_count": len(text), "paragraph_count": 1, "table_count": 0}
    result.update(overrides)
    return result


class EvaluationDatasetTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "test.db"
        document_repository.init_db(self.db_path)
        enrichment_repository.init_db(self.db_path)

    def add(self, number, enrich=True, **enrichment_overrides) -> str:
        announcement = make_announcement(number)
        announcement_repository.save_announcement(announcement, self.db_path)
        if enrich:
            enrichment_repository.save_enrichment(
                announcement["resource_url"], make_enrichment(**enrichment_overrides), self.db_path,
            )
        return announcement["resource_url"]

    def file_hash(self) -> str:
        return hashlib.sha256(self.db_path.read_bytes()).hexdigest()


class CaseIdTests(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(
            evald._case_id("https://example.test/1"), evald._case_id("https://example.test/1"),
        )

    def test_different_url_different_id(self):
        self.assertNotEqual(evald._case_id("https://example.test/1"), evald._case_id("https://example.test/2"))


class BuildEvaluationCaseTests(EvaluationDatasetTestCase):
    def context_for(self, resource_url) -> dict:
        return tender_context_module.build_tender_context(resource_url, db_path=self.db_path)

    def test_same_tender_same_case_id(self):
        url = self.add(1)
        context = self.context_for(url)

        first = evald.build_evaluation_case(context)
        second = evald.build_evaluation_case(context)

        self.assertEqual(first["case_id"], second["case_id"])

    def test_case_id_survives_content_change(self):
        # case_id зависит только от resource_url, не от содержимого (за это отвечает input_hash).
        url = self.add(1)
        before = evald.build_evaluation_case(self.context_for(url))

        enrichment_repository.save_enrichment(url, make_enrichment(description="Другое описание"), self.db_path)
        after = evald.build_evaluation_case(self.context_for(url))

        self.assertEqual(before["case_id"], after["case_id"])
        self.assertNotEqual(before["input_hash"], after["input_hash"])

    def test_input_hash_matches_tender_context(self):
        url = self.add(1)
        context = self.context_for(url)

        case = evald.build_evaluation_case(context)

        self.assertEqual(case["input_hash"], tender_context_module.compute_input_hash(context))

    def test_expected_template_fields_are_null(self):
        url = self.add(1)
        case = evald.build_evaluation_case(self.context_for(url))

        self.assertEqual(case["expected"], {
            "relevance_status": None, "opportunity_type": None,
            "category": None, "expected_reason": None, "notes": None,
        })

    def test_top_level_fields(self):
        url = self.add(1, contracting_authority="Минфин", procurement_type="goods", procedure_type="open")
        case = evald.build_evaluation_case(self.context_for(url))

        self.assertEqual(case["resource_url"], url)
        self.assertEqual(case["title"], "Тендер 1")
        self.assertEqual(case["contracting_authority"], "Минфин")
        self.assertEqual(case["procurement_type"], "goods")
        self.assertEqual(case["procedure_type"], "open")
        self.assertEqual(case["deadline_at"], "2026-09-15 18:00:00")

    def test_cpv_codes_included(self):
        url = self.add(1, cpv_codes=[{"code": "60100000", "name": "Road transport"}])
        case = evald.build_evaluation_case(self.context_for(url))
        self.assertEqual(case["cpv_codes"], [{"code": "60100000", "name": "Road transport"}])

    def test_document_coverage_incomplete_is_preserved(self):
        # Известный (enrichment) документ с неподдерживаемым расширением -> incomplete.
        url = self.add(1, documents=[
            {"document_id": "1", "filename": "spec.doc", "language": "RU", "title": None, "description": None},
        ])
        case = evald.build_evaluation_case(self.context_for(url))

        self.assertFalse(case["document_coverage"]["coverage_complete"])
        self.assertIn(".doc", case["document_coverage"]["unsupported_extensions"])

    def test_document_coverage_complete_is_preserved(self):
        url = self.add(1)
        case = evald.build_evaluation_case(self.context_for(url))
        self.assertTrue(case["document_coverage"]["coverage_complete"])

    def test_triage_context_is_compact(self):
        url = self.add(1)
        download_result = document_repository.save_download(
            url, "armeps_document", "1",
            {"source_url": "https://example.test/f.docx", "saved_path": "p", "filename": "f.docx",
             "content_type": None, "size_bytes": 10, "sha256": "a" * 64},
            db_path=self.db_path,
        )
        document_repository.save_docx_extraction(
            download_result["download_id"], make_docx_extraction("x" * 50), self.db_path,
        )
        case = evald.build_evaluation_case(self.context_for(url))

        triage_context = case["triage_context"]
        self.assertIn("char_budget", triage_context)
        self.assertIn("document_previews", triage_context)
        # Компактно: полного текста документа (без обрезки) в triage_context нет как отдельного поля.
        self.assertNotIn("documents_full_text", triage_context)

    def test_does_not_modify_tender_context(self):
        import copy
        url = self.add(1)
        context = self.context_for(url)
        original = copy.deepcopy(context)

        evald.build_evaluation_case(context)

        self.assertEqual(context, original)


class GetEvaluationCandidatesTests(EvaluationDatasetTestCase):
    def test_requires_enrichment(self):
        self.add(1, enrich=False)
        enriched_url = self.add(2)

        candidates = evald.get_evaluation_candidates(db_path=self.db_path)

        self.assertEqual([c["resource_url"] for c in candidates], [enriched_url])

    def test_document_extraction_not_required(self):
        url = self.add(1, documents=[
            {"document_id": "1", "filename": "spec.docx", "language": "RU", "title": None, "description": None},
        ])
        candidates = evald.get_evaluation_candidates(db_path=self.db_path)
        self.assertEqual([c["resource_url"] for c in candidates], [url])

    def test_empty_db_returns_empty(self):
        self.assertEqual(evald.get_evaluation_candidates(db_path=self.db_path), [])

    def test_limit(self):
        for number in (1, 2, 3, 4):
            self.add(number)
        self.assertEqual(len(evald.get_evaluation_candidates(db_path=self.db_path, limit=2)), 2)
        self.assertEqual(len(evald.get_evaluation_candidates(db_path=self.db_path, limit=None)), 4)

    def test_invalid_limit_raises(self):
        for limit in (0, -1, True):
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    evald.get_evaluation_candidates(db_path=self.db_path, limit=limit)

    def test_diversity_spreads_across_resource_type(self):
        for number in (1, 2, 3):
            self.add(number)
        for number in (4, 5, 6):
            announcement_repository.save_announcement(
                make_announcement(number, resource_type="eauction_tender_page"), self.db_path,
            )
            enrichment_repository.save_enrichment(
                f"https://example.test/resource/{number}", make_enrichment(), self.db_path,
            )
        for number in (7, 8, 9):
            announcement_repository.save_announcement(
                make_announcement(number, resource_type="direct_file"), self.db_path,
            )
            enrichment_repository.save_enrichment(
                f"https://example.test/resource/{number}", make_enrichment(), self.db_path,
            )

        candidates = evald.get_evaluation_candidates(db_path=self.db_path, limit=3)

        resource_types = {c["announcement"]["resource_type"] for c in candidates}
        self.assertEqual(resource_types, {"armeps_documents_page", "eauction_tender_page", "direct_file"})

    def test_returns_tender_context_shaped_dicts(self):
        self.add(1)
        [candidate] = evald.get_evaluation_candidates(db_path=self.db_path)
        for key in ("resource_url", "announcement", "enrichment", "document_coverage", "documents"):
            self.assertIn(key, candidate)


class BuildEvaluationDatasetTests(EvaluationDatasetTestCase):
    def test_builds_one_case_per_candidate(self):
        self.add(1)
        self.add(2)
        cases = evald.build_evaluation_dataset(db_path=self.db_path)
        self.assertEqual(len(cases), 2)
        self.assertEqual({c["resource_url"] for c in cases}, {
            "https://example.test/resource/1", "https://example.test/resource/2",
        })


class ExportEvaluationDatasetTests(EvaluationDatasetTestCase):
    def test_json_round_trip(self):
        self.add(1)
        cases = evald.build_evaluation_dataset(db_path=self.db_path)
        export_path = Path(self._tmp.name) / "export.json"

        result_path = evald.export_evaluation_dataset(cases, export_path)

        self.assertEqual(result_path, export_path)
        loaded = json.loads(export_path.read_text(encoding="utf-8"))
        self.assertEqual(loaded, cases)

    def test_creates_parent_directories(self):
        self.add(1)
        cases = evald.build_evaluation_dataset(db_path=self.db_path)
        export_path = Path(self._tmp.name) / "nested" / "dir" / "export.json"

        evald.export_evaluation_dataset(cases, export_path)

        self.assertTrue(export_path.exists())


class NoMutationTests(EvaluationDatasetTestCase):
    def test_get_candidates_does_not_change_db_file(self):
        self.add(1, documents=[
            {"document_id": "1", "filename": "spec.docx", "language": "RU", "title": None, "description": None},
        ])
        before = self.file_hash()

        evald.get_evaluation_candidates(db_path=self.db_path)

        self.assertEqual(self.file_hash(), before)

    def test_full_cli_flow_does_not_change_db_file(self):
        self.add(1)
        self.add(2, documents=[
            {"document_id": "1", "filename": "spec.doc", "language": "RU", "title": None, "description": None},
        ])
        before = self.file_hash()
        export_path = Path(self._tmp.name) / "out.json"

        with redirect_stdout(io.StringIO()):
            evald.main(["--db-path", str(self.db_path)])
            evald.main(["--db-path", str(self.db_path), "--resource-url", "https://example.test/resource/1"])
            evald.main(["--db-path", str(self.db_path), "--export", str(export_path)])

        self.assertEqual(self.file_hash(), before)
        self.assertTrue(export_path.exists())

    def test_read_only_connection_cannot_write(self):
        self.add(1)
        with evald._connect_read_only(self.db_path) as conn:
            with self.assertRaises(Exception):
                conn.execute("INSERT INTO announcements (resource_url) VALUES ('x')")

    def test_read_only_connect_does_not_create_missing_file(self):
        missing = Path(self._tmp.name) / "does_not_exist.db"
        with self.assertRaises(Exception):
            with evald._connect_read_only(missing):
                pass
        self.assertFalse(missing.exists())


class ProductionDbBlockedTests(unittest.TestCase):
    """
    Общая test-only защита (tests/safety_guards.py) блокирует sqlite3.connect к
    production data/tenders.db даже в режиме read-only: попытка использовать модуль
    с db_path по умолчанию (production) должна упасть на этой защите, а не тихо
    прочитать реальные данные.
    """

    def test_default_db_path_is_production_path(self):
        self.assertTrue(safety_guards.is_production_db(evald.DEFAULT_DB_PATH))

    def test_get_candidates_against_production_db_is_blocked(self):
        with safety_guards.expect_violation(self, safety_guards.DATABASE_MESSAGE):
            evald.get_evaluation_candidates()

    def test_cli_against_production_db_is_blocked(self):
        with redirect_stdout(io.StringIO()):
            with safety_guards.expect_violation(self, safety_guards.DATABASE_MESSAGE):
                evald.main([])


class CliTests(EvaluationDatasetTestCase):
    def run_cli(self, argv) -> str:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            evald.main(["--db-path", str(self.db_path), *argv])
        return buffer.getvalue()

    def test_list_prints_table_with_resource_urls(self):
        self.add(1)
        self.add(2)
        output = self.run_cli([])
        self.assertIn("https://example.test/resource/1", output)
        self.assertIn("https://example.test/resource/2", output)
        self.assertIn("RESOURCE_URL", output)

    def test_empty_db_prints_message_not_error(self):
        output = self.run_cli([])
        self.assertIn("Кандидатов не найдено", output)

    def test_limit_option(self):
        for number in (1, 2, 3):
            self.add(number)
        output = self.run_cli(["--limit", "1"])
        self.assertEqual(output.count("https://example.test/resource/"), 1)

    def test_resource_url_option_prints_full_case_json(self):
        self.add(1)
        output = self.run_cli(["--resource-url", "https://example.test/resource/1"])
        case = json.loads(output)
        self.assertEqual(case["resource_url"], "https://example.test/resource/1")
        self.assertIn("triage_context", case)
        self.assertIn("expected", case)

    def test_export_option_writes_file_and_does_not_print_json(self):
        self.add(1)
        self.add(2)
        export_path = Path(self._tmp.name) / "export.json"

        output = self.run_cli(["--export", str(export_path)])

        self.assertIn("Экспортировано 2", output)
        cases = json.loads(export_path.read_text(encoding="utf-8"))
        self.assertEqual(len(cases), 2)

    def test_export_with_resource_url_exports_single_case(self):
        self.add(1)
        self.add(2)
        export_path = Path(self._tmp.name) / "export.json"

        self.run_cli(["--resource-url", "https://example.test/resource/1", "--export", str(export_path)])

        cases = json.loads(export_path.read_text(encoding="utf-8"))
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["resource_url"], "https://example.test/resource/1")

    def test_unknown_resource_url_raises(self):
        with self.assertRaises(ValueError):
            self.run_cli(["--resource-url", "https://example.test/does-not-exist"])


if __name__ == "__main__":
    unittest.main()
