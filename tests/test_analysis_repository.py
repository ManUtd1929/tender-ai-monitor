"""
Тесты src.database.analysis_repository на временной SQLite БД.

Запуск из корня проекта:
    python -m unittest tests.test_analysis_repository -v
"""

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from src.ai import tender_context as tender_context_module
from src.database import analysis_repository as repo
from src.database import announcement_repository, document_repository, enrichment_repository

RESOURCE_URL = "https://example.test/resource/1"


def make_announcement(number=1, **overrides) -> dict:
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
        "description": "Международная перевозка груза",
        "documents": [],
    }
    enrichment.update(overrides)
    return enrichment


def make_triage_result(**overrides) -> dict:
    result = {
        "relevance_status": "relevant",
        "opportunity_type": "logistics",
        "category": "international_freight",
        "confidence": "high",
        "reason": "Международная перевозка",
        "requires_deep_analysis": True,
        "evidence": [],
    }
    result.update(overrides)
    return result


def make_deep_result(**overrides) -> dict:
    result = {
        "summary": "Международная перевозка груза",
        "opportunity_type": "logistics",
        "category": "international_freight",
        "why_interesting": "Логистика",
        "participation_barriers": [],
        "missing_information": [],
        "manual_review_required": False,
        "evidence": [],
        "procurement": None,
        "logistics": {name: None for name in (
            "service", "cargo", "origin", "destination", "transport_mode", "weight",
            "volume", "frequency", "customs_requirements", "insurance_requirements",
            "special_conditions",
        )},
    }
    result.update(overrides)
    return result


class AnalysisRepositoryTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "test.db"
        repo.init_db(self.db_path)

    def add(self, number=1, enrich=True, **enrichment_overrides) -> str:
        announcement = make_announcement(number)
        announcement_repository.save_announcement(announcement, self.db_path)
        if enrich:
            enrichment_repository.save_enrichment(
                announcement["resource_url"], make_enrichment(**enrichment_overrides), self.db_path,
            )
        return announcement["resource_url"]

    def hash_for(self, resource_url) -> str:
        context = tender_context_module.build_tender_context(resource_url, db_path=self.db_path)
        return tender_context_module.compute_input_hash(context)


class InitDbTests(AnalysisRepositoryTestCase):
    def test_creates_all_layers(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        for table in ("announcements", "announcement_enrichment", "document_downloads", "tender_triage", "tender_deep_analysis"):
            self.assertIn(table, names)

    def test_idempotent(self):
        repo.init_db(self.db_path)
        repo.init_db(self.db_path)


class SaveTriageTests(AnalysisRepositoryTestCase):
    def test_unknown_resource_url_raises(self):
        with self.assertRaises(ValueError):
            repo.save_triage("https://example.test/none", "hash1", make_triage_result(), db_path=self.db_path)

    def test_first_save_is_new(self):
        url = self.add()
        status = repo.save_triage(url, "hash1", make_triage_result(), db_path=self.db_path)
        self.assertEqual(status, "new")

    def test_same_result_is_existing(self):
        url = self.add()
        repo.save_triage(url, "hash1", make_triage_result(), db_path=self.db_path)
        status = repo.save_triage(url, "hash1", make_triage_result(), db_path=self.db_path)
        self.assertEqual(status, "existing")

    def test_changed_hash_is_updated(self):
        url = self.add()
        repo.save_triage(url, "hash1", make_triage_result(), db_path=self.db_path)
        status = repo.save_triage(url, "hash2", make_triage_result(), db_path=self.db_path)
        self.assertEqual(status, "updated")

    def test_changed_result_is_updated(self):
        url = self.add()
        repo.save_triage(url, "hash1", make_triage_result(), db_path=self.db_path)
        status = repo.save_triage(
            url, "hash1", make_triage_result(confidence="low"), db_path=self.db_path,
        )
        self.assertEqual(status, "updated")

    def test_round_trip_fields(self):
        url = self.add()
        repo.save_triage(
            url, "hash1", make_triage_result(), provider="openai", model="gpt-x",
            prompt_version="v2", db_path=self.db_path,
        )

        row = repo.get_triage(url, db_path=self.db_path)

        self.assertEqual(row["resource_url"], url)
        self.assertEqual(row["input_hash"], "hash1")
        self.assertEqual(row["provider"], "openai")
        self.assertEqual(row["model"], "gpt-x")
        self.assertEqual(row["prompt_version"], "v2")
        self.assertEqual(row["relevance_status"], "relevant")
        self.assertTrue(row["requires_deep_analysis"])
        self.assertEqual(row["result"], make_triage_result())

    def test_default_provider_and_model_are_fake(self):
        url = self.add()
        repo.save_triage(url, "hash1", make_triage_result(), db_path=self.db_path)
        row = repo.get_triage(url, db_path=self.db_path)
        self.assertEqual(row["provider"], "fake")
        self.assertEqual(row["model"], "fake")

    def test_get_missing_is_none(self):
        self.assertIsNone(repo.get_triage(RESOURCE_URL, db_path=self.db_path))

    def test_count_triage(self):
        self.assertEqual(repo.count_triage(self.db_path), 0)
        url1 = self.add(1)
        url2 = self.add(2)
        repo.save_triage(url1, "h1", make_triage_result(), db_path=self.db_path)
        repo.save_triage(url2, "h2", make_triage_result(), db_path=self.db_path)
        self.assertEqual(repo.count_triage(self.db_path), 2)


class SaveDeepAnalysisTests(AnalysisRepositoryTestCase):
    def test_unknown_resource_url_raises(self):
        with self.assertRaises(ValueError):
            repo.save_deep_analysis("https://example.test/none", "hash1", make_deep_result(), db_path=self.db_path)

    def test_new_then_existing_then_updated(self):
        url = self.add()
        self.assertEqual(repo.save_deep_analysis(url, "hash1", make_deep_result(), db_path=self.db_path), "new")
        self.assertEqual(repo.save_deep_analysis(url, "hash1", make_deep_result(), db_path=self.db_path), "existing")
        self.assertEqual(repo.save_deep_analysis(url, "hash2", make_deep_result(), db_path=self.db_path), "updated")

    def test_round_trip(self):
        url = self.add()
        repo.save_deep_analysis(url, "hash1", make_deep_result(), db_path=self.db_path)

        row = repo.get_deep_analysis(url, db_path=self.db_path)

        self.assertEqual(row["input_hash"], "hash1")
        self.assertEqual(row["result"], make_deep_result())

    def test_get_missing_is_none(self):
        self.assertIsNone(repo.get_deep_analysis(RESOURCE_URL, db_path=self.db_path))

    def test_count_deep_analysis(self):
        url = self.add()
        self.assertEqual(repo.count_deep_analysis(self.db_path), 0)
        repo.save_deep_analysis(url, "hash1", make_deep_result(), db_path=self.db_path)
        self.assertEqual(repo.count_deep_analysis(self.db_path), 1)


class TriageCandidatesTests(AnalysisRepositoryTestCase):
    def test_enriched_without_triage_is_candidate(self):
        url = self.add()
        candidates = repo.get_triage_candidates(db_path=self.db_path)
        self.assertEqual([c["resource_url"] for c in candidates], [url])
        self.assertIn("tender_context", candidates[0])
        self.assertIn("input_hash", candidates[0])

    def test_without_enrichment_is_not_candidate(self):
        self.add(enrich=False)
        self.assertEqual(repo.get_triage_candidates(db_path=self.db_path), [])

    def test_missing_document_processing_still_a_candidate(self):
        # Раздел 13: успешная обработка документов не обязательна для triage.
        url = self.add(documents=[
            {"document_id": "1", "filename": "spec.docx", "language": "RU", "title": None, "description": None},
        ])
        candidates = repo.get_triage_candidates(db_path=self.db_path)
        self.assertEqual([c["resource_url"] for c in candidates], [url])

    def test_matching_hash_is_not_a_candidate(self):
        url = self.add()
        input_hash = self.hash_for(url)
        repo.save_triage(url, input_hash, make_triage_result(), db_path=self.db_path)

        self.assertEqual(repo.get_triage_candidates(db_path=self.db_path), [])

    def test_changed_hash_is_a_candidate_again(self):
        url = self.add()
        input_hash = self.hash_for(url)
        repo.save_triage(url, input_hash, make_triage_result(), db_path=self.db_path)

        enrichment_repository.save_enrichment(
            url, make_enrichment(description="Изменённое описание"), self.db_path,
        )

        candidates = repo.get_triage_candidates(db_path=self.db_path)
        self.assertEqual([c["resource_url"] for c in candidates], [url])
        self.assertNotEqual(candidates[0]["input_hash"], input_hash)

    def test_limit(self):
        for number in (1, 2, 3):
            self.add(number)
        candidates = repo.get_triage_candidates(db_path=self.db_path, limit=2)
        self.assertEqual(len(candidates), 2)

    def test_invalid_limit_raises(self):
        with self.assertRaises(ValueError):
            repo.get_triage_candidates(db_path=self.db_path, limit=0)

    def test_count_matches_candidates(self):
        self.add(1)
        self.add(2)
        self.assertEqual(repo.count_triage_candidates(self.db_path), 2)


class DeepAnalysisCandidatesTests(AnalysisRepositoryTestCase):
    def test_no_triage_is_not_a_candidate(self):
        self.add()
        self.assertEqual(repo.get_deep_analysis_candidates(db_path=self.db_path), [])

    def test_not_relevant_triage_is_not_a_candidate(self):
        url = self.add()
        repo.save_triage(
            url, self.hash_for(url),
            make_triage_result(relevance_status="not_relevant", requires_deep_analysis=False,
                                opportunity_type="unrelated"),
            db_path=self.db_path,
        )
        self.assertEqual(repo.get_deep_analysis_candidates(db_path=self.db_path), [])

    def test_relevant_triage_is_a_candidate(self):
        url = self.add()
        repo.save_triage(url, self.hash_for(url), make_triage_result(), db_path=self.db_path)

        candidates = repo.get_deep_analysis_candidates(db_path=self.db_path)

        self.assertEqual([c["resource_url"] for c in candidates], [url])
        self.assertEqual(candidates[0]["triage"], make_triage_result())

    def test_maybe_triage_is_a_candidate(self):
        url = self.add()
        repo.save_triage(
            url, self.hash_for(url), make_triage_result(relevance_status="maybe"), db_path=self.db_path,
        )
        candidates = repo.get_deep_analysis_candidates(db_path=self.db_path)
        self.assertEqual([c["resource_url"] for c in candidates], [url])

    def test_matching_hash_is_not_a_candidate(self):
        url = self.add()
        input_hash = self.hash_for(url)
        repo.save_triage(url, input_hash, make_triage_result(), db_path=self.db_path)
        repo.save_deep_analysis(url, input_hash, make_deep_result(), db_path=self.db_path)

        self.assertEqual(repo.get_deep_analysis_candidates(db_path=self.db_path), [])

    def test_changed_hash_is_a_candidate_again(self):
        url = self.add()
        input_hash = self.hash_for(url)
        repo.save_triage(url, input_hash, make_triage_result(), db_path=self.db_path)
        repo.save_deep_analysis(url, input_hash, make_deep_result(), db_path=self.db_path)

        enrichment_repository.save_enrichment(
            url, make_enrichment(description="Изменённое описание"), self.db_path,
        )

        candidates = repo.get_deep_analysis_candidates(db_path=self.db_path)
        self.assertEqual([c["resource_url"] for c in candidates], [url])

    def test_limit(self):
        for number in (1, 2, 3):
            url = self.add(number)
            repo.save_triage(url, self.hash_for(url), make_triage_result(), db_path=self.db_path)

        candidates = repo.get_deep_analysis_candidates(db_path=self.db_path, limit=2)
        self.assertEqual(len(candidates), 2)

    def test_invalid_limit_raises(self):
        with self.assertRaises(ValueError):
            repo.get_deep_analysis_candidates(db_path=self.db_path, limit=-1)

    def test_count_matches_candidates(self):
        url = self.add()
        repo.save_triage(url, self.hash_for(url), make_triage_result(), db_path=self.db_path)
        self.assertEqual(repo.count_deep_analysis_candidates(self.db_path), 1)


if __name__ == "__main__":
    unittest.main()
