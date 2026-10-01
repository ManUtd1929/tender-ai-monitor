"""
Регрессия первого live one-shot: модель вернула ev_doc_40_fdab79d939c6153e_0010 (16-символьный хеш
из content_group_id) вместо реального ev_doc_40_fdab79_0010 (6 символов). Структура фикстуры повторяет
live-каталог (один docx, download_id=40, member_name="", 0010 — 11-я строка), текст синтетический.
OpenAI client не создаётся.

    python -m unittest tests.test_live_evidence_id_regression -v
"""

import unittest

from src.ai import analysis_pipeline, deep_prompt, evidence_catalog
from src.database import pipeline_state_repository as state_repo
from tests.test_deep_dedup import make_document, make_tender_context
from tests.test_deep_prompt import make_triage_result
from tests.test_openai_deep_analysis import UNCLEAR, make_analyzer, make_response, payload
from src.ai import openai_deep_analysis, tender_context as ctx

LIVE_BAD_ID = "ev_doc_40_fdab79d939c6153e_0010"
TEXT = "\n".join(f"Строка документа {n}" for n in range(40))
TENDER = make_tender_context([make_document("", TEXT, download_id=40, file_type="docx")])
TRIAGE = make_triage_result()


def build_context():
    deep = ctx.build_deep_analysis_context(TENDER)
    return deep, deep_prompt.build_deep_context(TENDER, deep, TRIAGE)


DEEP, DEEP_CONTEXT = build_context()
CATALOG = DEEP_CONTEXT["evidence_catalog"]
GROUP_ID = DEEP_CONTEXT["content_groups"][0]["content_group_id"]
UNIT_10 = next(u for u in CATALOG if u["download_id"] == 40 and u["exact_text"] == "Строка документа 10")
VALID_ID = UNIT_10["evidence_id"]
FRAGMENT = VALID_ID.split("_")[3]
LIVE_SHAPED_BAD_ID = f"ev_doc_40_{GROUP_ID.removeprefix('content-')}_0010"  # то, что собрала модель


def barrier(evidence_ids):
    return {"type": "financial_requirement", "description": "d", "severity": "low", "evidence_ids": evidence_ids}


class LiveIdRegressionTests(unittest.TestCase):
    def test_live_shape_matches_real_failure(self):
        self.assertEqual(len(FRAGMENT), 6)
        self.assertEqual(len(GROUP_ID.removeprefix("content-")), 16)
        self.assertTrue(GROUP_ID.removeprefix("content-").startswith(FRAGMENT))
        self.assertNotEqual(LIVE_SHAPED_BAD_ID, VALID_ID)
        self.assertEqual(LIVE_SHAPED_BAD_ID.rsplit("_", 1)[1], VALID_ID.rsplit("_", 1)[1])

    def test_catalog_ids_are_deterministic(self):
        again = build_context()[1]["evidence_catalog"]
        self.assertEqual([u["evidence_id"] for u in CATALOG], [u["evidence_id"] for u in again])

    def test_live_malformed_id_is_unknown_and_valid_id_is_known(self):
        index = evidence_catalog.index_catalog(CATALOG)
        for bad in (LIVE_BAD_ID, LIVE_SHAPED_BAD_ID):
            with self.assertRaisesRegex(ValueError, "неизвестный evidence_id"):
                evidence_catalog.materialize_evidence([bad], index, "participation_barriers[0]")
        (item,) = evidence_catalog.materialize_evidence([VALID_ID], index, "x")
        self.assertEqual(item["text"], "Строка документа 10")

    def test_prompt_exposes_only_catalog_ids_and_forbids_construction(self):
        text = deep_prompt.build_user_input(DEEP_CONTEXT)
        self.assertIn(f"[{VALID_ID}] Строка документа 10", text)
        self.assertNotIn(LIVE_SHAPED_BAD_ID, text)
        for needle in ("COPY each evidence_id EXACTLY", "never construct", "never modify its hash fragment",
                       "content_group_id", "NOT parts of an ID"):
            self.assertIn(needle.lower(), deep_prompt.SYSTEM_PROMPT.lower())

    def test_prompt_version_bumped_to_v6(self):
        self.assertEqual(deep_prompt.DEEP_PROMPT_VERSION, "procurement-deep-v6")

    def test_malformed_id_in_barrier_fails_validation_and_keeps_raw_output(self):
        # analyzer строит каталог из своего контекста; проверяем само поведение на каталоге теста
        raw = payload(**UNCLEAR, participation_barriers=[barrier([LIVE_BAD_ID])])
        analyzer, client, _ = make_analyzer(make_response(raw))
        with self.assertRaises(openai_deep_analysis.DeepAnalysisError) as raised:
            analyzer.deep_analyze_with_metadata(TENDER, DEEP, TRIAGE)
        error = raised.exception
        self.assertEqual(error.kind, openai_deep_analysis.KIND_VALIDATION)
        self.assertIn("неизвестный evidence_id", str(error))
        self.assertEqual(error.raw_model_output["participation_barriers"][0]["evidence_ids"], [LIVE_BAD_ID])
        self.assertEqual(len(client.responses.calls), 1)  # без повторных вызовов

    def test_materialization_uses_the_same_catalog_as_prompt(self):
        analyzer, _, _ = make_analyzer()
        self.assertIn(f"[{VALID_ID}] Строка документа 10", analyzer.build_request(DEEP_CONTEXT)["input"])
        response = make_response(payload(**UNCLEAR, evidence_ids=[VALID_ID]))
        raw, result = analyzer._parse_response(response, DEEP_CONTEXT, {})
        self.assertEqual(result["evidence"][0]["text"], "Строка документа 10")
        self.assertEqual(result["evidence"][0]["evidence_id"], VALID_ID)

    def test_no_openai_client_created(self):
        analyzer = openai_deep_analysis.OpenAIDeepAnalysisAnalyzer(environ={})
        self.assertIsNone(analyzer._client)


class RetryEligibilityTests(unittest.TestCase):
    PRIOR = {"state": state_repo.STATE_ESCALATION_CANDIDATE, "input_hash": "h", "model": "gpt-5.6-luna",
             "prompt_version": "procurement-deep-v5", "error_kind": "validation"}

    class Analyzer:
        model = "gpt-5.6-luna"

        def __init__(self, version):
            self.prompt_version = version

    STATES = (state_repo.STATE_ESCALATION_CANDIDATE, state_repo.STATE_DEEP_ERROR)

    def test_same_prompt_version_stays_blocked(self):
        self.assertTrue(analysis_pipeline._known_failure(self.PRIOR, "h", self.Analyzer("procurement-deep-v5"), self.STATES))

    def test_v6_makes_failed_live_tender_eligible_once(self):
        self.assertFalse(analysis_pipeline._known_failure(
            self.PRIOR, "h", self.Analyzer(deep_prompt.DEEP_PROMPT_VERSION), self.STATES))
        failed_v6 = {**self.PRIOR, "prompt_version": deep_prompt.DEEP_PROMPT_VERSION}
        self.assertTrue(analysis_pipeline._known_failure(
            failed_v6, "h", self.Analyzer(deep_prompt.DEEP_PROMPT_VERSION), self.STATES))


if __name__ == "__main__":
    unittest.main()
