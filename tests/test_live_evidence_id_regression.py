"""
Регрессия live-сбоев Deep v5/v6: модель строила строковый ID из видимого content_group_id
(ev_doc_40_fdab79d939c6153e_0010 / _0004 вместо ev_doc_40_fdab79_0010). procurement-deep-v7: модель
возвращает request-local целочисленные evidence_refs (1..N), канонические ID остаются внутренними.
Фикстура повторяет структуру live-каталога (один docx, download_id=40, member_name="", 0010 — 11-я
строка документа), текст синтетический. OpenAI client не создаётся.

    python -m unittest tests.test_live_evidence_id_regression -v
"""

import json
import unittest

from src.ai import analysis_pipeline, deep_prompt, evidence_catalog, openai_deep_analysis
from src.ai import tender_context as ctx
from src.database import pipeline_state_repository as state_repo
from tests.test_deep_dedup import make_document, make_tender_context
from tests.test_deep_prompt import make_triage_result
from tests.test_openai_deep_analysis import UNCLEAR, make_analyzer, make_response, payload

LIVE_BAD_IDS = ("ev_doc_40_fdab79d939c6153e_0010", "ev_doc_40_fdab79d939c6153e_0004")
TEXT = "\n".join(f"Строка документа {n}" for n in range(40))
TRIAGE = make_triage_result()


def build_for(tender):
    deep = ctx.build_deep_analysis_context(tender)
    return deep, deep_prompt.build_deep_context(tender, deep, TRIAGE)


TENDER = make_tender_context([make_document("", TEXT, download_id=40, file_type="docx")])
DEEP, DEEP_CONTEXT = build_for(TENDER)
CATALOG = DEEP_CONTEXT["evidence_catalog"]
REFS = evidence_catalog.evidence_ref_map(CATALOG)
GROUP = DEEP_CONTEXT["content_groups"][0]
UNIT_10 = next(u for u in CATALOG if u["download_id"] == 40 and u["exact_text"] == "Строка документа 10")
VALID_ID = UNIT_10["evidence_id"]
REF_10 = REFS[VALID_ID]


def barrier(refs):
    return {"type": "financial_requirement", "description": "d", "severity": "low", "evidence_refs": refs}


def all_ref_schemas(node):
    """Все схемы значений evidence_refs в JSON schema."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "evidence_refs":
                yield value
            else:
                yield from all_ref_schemas(value)
    elif isinstance(node, list):
        for value in node:
            yield from all_ref_schemas(value)


class NumericRefProtocolTests(unittest.TestCase):
    def test_refs_are_deterministic_one_based_positions(self):
        again = build_for(TENDER)[1]["evidence_catalog"]
        self.assertEqual(REFS, evidence_catalog.evidence_ref_map(again))
        self.assertEqual(sorted(REFS.values()), list(range(1, len(CATALOG) + 1)))
        self.assertEqual(REFS[CATALOG[0]["evidence_id"]], 1)
        self.assertEqual(REFS[CATALOG[-1]["evidence_id"]], len(CATALOG))

    def test_strict_schema_uses_bounded_integers_without_enum(self):
        size = len(CATALOG)
        schema = deep_prompt.build_deep_output_schema(size)
        found = list(all_ref_schemas(schema))
        self.assertEqual(len(found), 4)  # top-level, barrier, item, lot
        for ref_schema in found:
            self.assertEqual(
                ref_schema, {"type": "array", "items": {"type": "integer", "minimum": 1, "maximum": size}},
            )
        self.assertNotIn("enum", json.dumps(found))
        self.assertNotIn("evidence_ids", json.dumps(schema))

    def test_empty_catalog_schema_is_safe(self):
        schema = deep_prompt.build_deep_output_schema(0)
        for ref_schema in all_ref_schemas(schema):
            self.assertEqual(ref_schema["maxItems"], 0)
            self.assertNotIn("minimum", ref_schema["items"])
            self.assertNotIn("maximum", ref_schema["items"])

    def test_numeric_ref_materializes_canonical_evidence(self):
        index = evidence_catalog.index_catalog(CATALOG)
        (evidence,) = evidence_catalog.materialize_evidence([REF_10], CATALOG, "x")
        self.assertEqual(evidence["evidence_id"], VALID_ID)
        self.assertEqual(evidence["text"], "Строка документа 10")
        self.assertEqual((evidence["download_id"], evidence["member_name"]), (40, ""))
        self.assertIs(index[VALID_ID], UNIT_10)

    def test_live_malformed_strings_are_not_part_of_the_protocol(self):
        for bad in LIVE_BAD_IDS + (VALID_ID, str(REF_10)):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "целым числом"):
                evidence_catalog.materialize_evidence([bad], CATALOG, "participation_barriers[0]")

    def test_bounds_and_duplicates_are_rejected_without_repair(self):
        size = len(CATALOG)
        evidence_catalog.materialize_evidence([1, size], CATALOG, "x")  # границы валидны
        for bad in ([0], [size + 1], [-1], [REF_10, REF_10], [True], [1.0]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                evidence_catalog.materialize_evidence(bad, CATALOG, "x")

    def test_model_visible_input_has_numbered_units_and_no_internal_identifiers(self):
        text = deep_prompt.build_user_input(DEEP_CONTEXT)
        self.assertRegex(text, rf"(?m)^\[{REF_10}\] Строка документа 10$")
        self.assertIn(f"EVIDENCE_CATALOG ({len(CATALOG)} units)", text)
        digest = GROUP["content_sha256"]
        for forbidden in (VALID_ID, "ev_doc", "ev_ann", "content_group_id", "content_sha256", GROUP["content_group_id"],
                          digest, digest[:6], digest[:16]):
            self.assertNotIn(forbidden, text)
        # internal catalog/context still carry provenance
        self.assertEqual(UNIT_10["content_group_id"], GROUP["content_group_id"])

    def test_prompt_never_mentions_canonical_ids(self):
        for forbidden in ("ev_doc", "ev_ann", "ev_enr", "evidence_id", "content_group_id", "content_sha256"):
            self.assertNotIn(forbidden, deep_prompt.SYSTEM_PROMPT)
        self.assertIn("Cite evidence using the provided numeric evidence references only", deep_prompt.SYSTEM_PROMPT)
        self.assertIn("from 1 through N", deep_prompt.SYSTEM_PROMPT)
        self.assertIn("Do not create references", deep_prompt.SYSTEM_PROMPT)

    def test_prompt_version_is_v7(self):
        self.assertEqual(deep_prompt.DEEP_PROMPT_VERSION, "procurement-deep-v7")


class SameCatalogTests(unittest.TestCase):
    def test_request_input_schema_and_materialization_use_one_catalog(self):
        analyzer, client, _ = make_analyzer()
        request = analyzer.build_request(DEEP_CONTEXT)
        size = len(CATALOG)
        schema = request["text"]["format"]["schema"]
        for ref_schema in all_ref_schemas(schema):
            self.assertEqual(ref_schema["items"]["maximum"], size)
        self.assertRegex(request["input"], rf"(?m)^\[{size}\] ")
        self.assertNotRegex(request["input"], rf"(?m)^\[{size + 1}\] ")
        response = make_response(payload(**UNCLEAR, evidence_refs=[REF_10]))
        _, result = analyzer._parse_response(response, DEEP_CONTEXT, {})
        self.assertEqual(result["evidence"][0]["evidence_id"], VALID_ID)
        self.assertEqual(result["evidence"][0]["text"], "Строка документа 10")
        self.assertEqual(client.responses.calls, [])  # fake client не вызывался

    def test_live_malformed_id_fails_validation_and_keeps_raw_output(self):
        for bad in LIVE_BAD_IDS:
            with self.subTest(bad=bad):
                raw = payload(**UNCLEAR, participation_barriers=[barrier([bad])])
                analyzer, client, _ = make_analyzer(make_response(raw))
                with self.assertRaises(openai_deep_analysis.DeepAnalysisError) as raised:
                    analyzer.deep_analyze_with_metadata(TENDER, DEEP, TRIAGE)
                error = raised.exception
                self.assertEqual(error.kind, openai_deep_analysis.KIND_VALIDATION)
                self.assertEqual(error.raw_model_output["participation_barriers"][0]["evidence_refs"], [bad])
                self.assertEqual(len(client.responses.calls), 1)  # без повторных вызовов

    def test_out_of_range_ref_fails_validation(self):
        raw = payload(**UNCLEAR, evidence_refs=[len(CATALOG) + 1])
        analyzer, _, _ = make_analyzer(make_response(raw))
        with self.assertRaises(openai_deep_analysis.DeepAnalysisError) as raised:
            analyzer.deep_analyze_with_metadata(TENDER, DEEP, TRIAGE)
        self.assertEqual(raised.exception.kind, openai_deep_analysis.KIND_VALIDATION)

    def test_no_openai_client_created(self):
        analyzer = openai_deep_analysis.OpenAIDeepAnalysisAnalyzer(environ={})
        self.assertIsNone(analyzer._client)


class MedicalScaleTests(unittest.TestCase):
    UNITS = 1699

    @classmethod
    def setUpClass(cls):
        text = "\n".join(f"Медицинская позиция {n}" for n in range(1, cls.UNITS + 1))
        tender = make_tender_context([make_document("spec.docx", text, download_id=7, file_type="docx")])
        deep = ctx.build_deep_analysis_context(tender)
        cls.context = deep_prompt.build_deep_context(tender, deep, TRIAGE)
        cls.catalog = cls.context["evidence_catalog"]
        cls.documents = [u for u in cls.catalog if u["source_type"] == "document"]

    def test_catalog_has_at_least_1699_units(self):
        self.assertGreaterEqual(len(self.catalog), self.UNITS)

    def test_schema_scales_without_enum(self):
        size = len(self.catalog)
        schema = deep_prompt.build_deep_output_schema(size)
        refs = list(all_ref_schemas(schema))
        self.assertTrue(refs)
        for ref_schema in refs:
            self.assertEqual(ref_schema["items"], {"type": "integer", "minimum": 1, "maximum": size})
        serialized = json.dumps(schema)
        self.assertNotIn('"enum": ["ev_', serialized)
        self.assertLess(len(serialized), 20_000)  # размер схемы не зависит от размера каталога
        text_format = deep_prompt.build_text_format(size)
        self.assertEqual(text_format["format"]["schema"], schema)

    def test_boundaries_materialize_canonical_units(self):
        size = len(self.catalog)
        last = evidence_catalog.materialize_evidence([size], self.catalog, "x")[0]
        self.assertEqual(last["evidence_id"], self.catalog[-1]["evidence_id"])
        self.assertEqual(last["text"], f"Медицинская позиция {self.UNITS}")
        first = evidence_catalog.materialize_evidence([1], self.catalog, "x")[0]
        self.assertEqual(first["evidence_id"], self.catalog[0]["evidence_id"])
        with self.assertRaisesRegex(ValueError, "неизвестная ссылка"):
            evidence_catalog.materialize_evidence([size + 1], self.catalog, "x")

    def test_max_ref_equals_exactly_1699_when_catalog_is_documents_only(self):
        # announcement/enrichment units присутствуют всегда; проверяем явную границу 1699 на каталоге из 1699 units
        catalog = self.catalog[:self.UNITS]
        self.assertEqual(len(catalog), 1699)
        schema = deep_prompt.build_deep_output_schema(len(catalog))
        for ref_schema in all_ref_schemas(schema):
            self.assertEqual(ref_schema["items"]["maximum"], 1699)
        evidence_catalog.materialize_evidence([1699], catalog, "x")
        with self.assertRaises(ValueError):
            evidence_catalog.materialize_evidence([1700], catalog, "x")


class RetryEligibilityTests(unittest.TestCase):
    STATES = (state_repo.STATE_ESCALATION_CANDIDATE, state_repo.STATE_DEEP_ERROR)

    class Analyzer:
        model = "gpt-5.6-luna"

        def __init__(self, version):
            self.prompt_version = version

    def prior(self, version):
        return {"state": state_repo.STATE_ESCALATION_CANDIDATE, "input_hash": "h", "model": "gpt-5.6-luna",
                "prompt_version": version, "error_kind": "validation"}

    def known(self, prior_version, current_version):
        return analysis_pipeline._known_failure(
            self.prior(prior_version), "h", self.Analyzer(current_version), self.STATES,
        )

    def test_v6_failure_does_not_block_v7_retry(self):
        self.assertFalse(self.known("procurement-deep-v6", deep_prompt.DEEP_PROMPT_VERSION))

    def test_v5_failure_does_not_block_v7_retry(self):
        self.assertFalse(self.known("procurement-deep-v5", deep_prompt.DEEP_PROMPT_VERSION))

    def test_v7_failure_blocks_same_v7_retry(self):
        self.assertTrue(self.known(deep_prompt.DEEP_PROMPT_VERSION, deep_prompt.DEEP_PROMPT_VERSION))


if __name__ == "__main__":
    unittest.main()
