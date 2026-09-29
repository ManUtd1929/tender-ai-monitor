"""
Тесты src.ai.deep_prompt: strict JSON schema (procurement deep-MVP), детерминированная
сборка DEEP_CONTEXT и сериализация. Сети и OpenAI здесь нет.

Запуск из корня проекта:
    python -m unittest tests.test_deep_prompt -v
"""

import json
import unittest

from src.ai import deep_prompt, relevance_schema


def make_tender_context(**overrides) -> dict:
    context = {
        "resource_url": "https://example.test/resource/1",
        "announcement": {
            "title": "Поставка компьютерной техники",
            "section": "Открытый конкурс",
            "resource_type": "armeps_documents_page",
            "published_at": "2026-01-10",
            "deadline_at": "2026-02-01",
            "resource_url": "https://example.test/resource/1",
        },
        "enrichment": {
            "enrichment_status": "success",
            "procedure_code": "ATCPO-2026-0001",
            "contracting_authority": "Министерство обороны",
            "detail_titles": {"default": "Компьютеры", "ru": None, "en": None},
            "procurement_type": "goods",
            "procedure_type": "open",
            "description": "Закупка ноутбуков для нужд министерства",
            "estimated_value_amd": 5000000,
            "dates": {"published_at_detail": "2026-01-10", "deadline_at_detail": "2026-02-01"},
            "number_of_lots": 3,
            "cpv_codes": [{"code": "30213100", "name": "Портативные компьютеры"}],
        },
        "document_coverage": {
            "successful_extractions": 1, "failed_extractions": 0, "skipped_extractions": 0,
            "successful_docx": 1, "successful_xlsx": 0, "unsupported_extensions": [],
            "total_extracted_chars": 40, "coverage_complete": True,
        },
        "documents": [
            {
                "download_id": 7, "member_name": "spec.docx", "file_type": "docx",
                "extraction_status": "success", "text": "Ноутбук, 10 шт.\nГарантия 24 месяца.",
                "metrics": {},
            },
        ],
    }
    context.update(overrides)
    return context


def make_deep_analysis_context(**overrides) -> dict:
    context = {
        "resource_url": "https://example.test/resource/1",
        "chunk_size": 4000,
        "chunks": [
            {
                "chunk_id": "7:spec.docx:0", "download_id": 7, "member_name": "spec.docx",
                "file_type": "docx", "text": "Ноутбук, 10 шт.\nГарантия 24 месяца.",
            },
        ],
    }
    context.update(overrides)
    return context


def make_triage_result(**overrides) -> dict:
    result = {
        "relevance_status": "relevant", "opportunity_type": "procurement", "category": "computer_equipment",
        "confidence": "high", "reason": "Закупка ноутбуков", "requires_deep_analysis": True, "evidence": [],
    }
    result.update(overrides)
    return result


class BuildDeepContextTests(unittest.TestCase):
    def test_flattens_announcement_and_enrichment(self):
        context = deep_prompt.build_deep_context(
            make_tender_context(), make_deep_analysis_context(), make_triage_result(),
        )
        self.assertEqual(context["title"], "Поставка компьютерной техники")
        self.assertEqual(context["contracting_authority"], "Министерство обороны")
        self.assertEqual(context["procedure_type"], "open")
        self.assertEqual(context["number_of_lots"], 3)
        self.assertEqual(context["cpv_codes"], [{"code": "30213100", "name": "Портативные компьютеры"}])

    def test_documents_are_summarized_without_text(self):
        context = deep_prompt.build_deep_context(
            make_tender_context(), make_deep_analysis_context(), make_triage_result(),
        )
        self.assertEqual(context["documents"], [
            {"download_id": 7, "member_name": "spec.docx", "file_type": "docx", "extraction_status": "success"},
        ])

    def test_chunks_are_carried_over(self):
        context = deep_prompt.build_deep_context(
            make_tender_context(), make_deep_analysis_context(), make_triage_result(),
        )
        self.assertEqual(context["chunks"], make_deep_analysis_context()["chunks"])

    def test_triage_result_is_summarized(self):
        context = deep_prompt.build_deep_context(
            make_tender_context(), make_deep_analysis_context(), make_triage_result(),
        )
        self.assertEqual(context["triage"], {
            "relevance_status": "relevant", "opportunity_type": "procurement",
            "category": "computer_equipment", "confidence": "high", "reason": "Закупка ноутбуков",
        })

    def test_mismatched_resource_url_rejected(self):
        with self.assertRaises(ValueError):
            deep_prompt.build_deep_context(
                make_tender_context(), make_deep_analysis_context(resource_url="other"), make_triage_result(),
            )

    def test_inputs_are_not_mutated(self):
        tender_context = make_tender_context()
        deep_analysis_context = make_deep_analysis_context()
        triage_result = make_triage_result()
        before = (
            json.dumps(tender_context, sort_keys=True),
            json.dumps(deep_analysis_context, sort_keys=True),
            json.dumps(triage_result, sort_keys=True),
        )
        deep_prompt.build_deep_context(tender_context, deep_analysis_context, triage_result)
        after = (
            json.dumps(tender_context, sort_keys=True),
            json.dumps(deep_analysis_context, sort_keys=True),
            json.dumps(triage_result, sort_keys=True),
        )
        self.assertEqual(before, after)

    def test_is_deterministic(self):
        first = deep_prompt.build_deep_context(
            make_tender_context(), make_deep_analysis_context(), make_triage_result(),
        )
        second = deep_prompt.build_deep_context(
            make_tender_context(), make_deep_analysis_context(), make_triage_result(),
        )
        self.assertEqual(deep_prompt.serialize_deep_context(first), deep_prompt.serialize_deep_context(second))


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.schema = deep_prompt.build_deep_output_schema()

    def test_top_level_fields_match_relevance_schema(self):
        self.assertEqual(set(self.schema["properties"]), set(relevance_schema.DEEP_ANALYSIS_COMMON_FIELDS))
        self.assertEqual(set(self.schema["required"]), set(relevance_schema.DEEP_ANALYSIS_COMMON_FIELDS))
        self.assertIs(self.schema["additionalProperties"], False)

    def test_opportunity_type_is_restricted_to_mvp(self):
        enum = self.schema["properties"]["opportunity_type"]["enum"]
        self.assertEqual(set(enum), {"procurement", "unclear"})
        self.assertNotIn("logistics", enum)
        self.assertNotIn("logistics_and_procurement", enum)
        self.assertNotIn("other_service", enum)
        self.assertNotIn("unrelated", enum)

    def test_logistics_field_is_forced_null(self):
        self.assertEqual(self.schema["properties"]["logistics"]["type"], "null")

    def test_procurement_field_is_nullable_object(self):
        procurement = self.schema["properties"]["procurement"]
        self.assertEqual(set(procurement["type"]), {"object", "null"})
        self.assertEqual(set(procurement["properties"]), set(relevance_schema.PROCUREMENT_FIELDS))
        self.assertEqual(set(procurement["required"]), set(relevance_schema.PROCUREMENT_FIELDS))

    def test_procurement_item_schema_matches_relevance_schema(self):
        item_schema = self.schema["properties"]["procurement"]["properties"]["items"]["items"]
        self.assertEqual(set(item_schema["properties"]), set(relevance_schema.PROCUREMENT_ITEM_FIELDS))
        self.assertEqual(set(item_schema["required"]), set(relevance_schema.PROCUREMENT_ITEM_FIELDS))

    def test_procurement_lot_schema_matches_relevance_schema(self):
        lot_schema = self.schema["properties"]["procurement"]["properties"]["lots"]["items"]
        self.assertEqual(set(lot_schema["properties"]), set(relevance_schema.PROCUREMENT_LOT_FIELDS))
        self.assertEqual(set(lot_schema["required"]), set(relevance_schema.PROCUREMENT_LOT_FIELDS))

    def test_brand_or_equivalent_schema_matches_relevance_schema(self):
        brand_schema = self.schema["properties"]["procurement"]["properties"]["brand_or_equivalent"]
        self.assertEqual(set(brand_schema["properties"]), set(relevance_schema.BRAND_OR_EQUIVALENT_FIELDS))
        self.assertEqual(set(brand_schema["type"]), {"object", "null"})

    def test_barrier_schema_matches_relevance_schema(self):
        barrier_schema = self.schema["properties"]["participation_barriers"]["items"]
        self.assertEqual(set(barrier_schema["properties"]), set(relevance_schema.PARTICIPATION_BARRIER_FIELDS))
        self.assertEqual(
            barrier_schema["properties"]["type"]["enum"], list(relevance_schema.PARTICIPATION_BARRIER_TYPES),
        )
        self.assertEqual(
            barrier_schema["properties"]["severity"]["enum"], list(relevance_schema.CONFIDENCE_LEVELS),
        )

    def test_source_conflict_schema_matches_relevance_schema(self):
        conflict_schema = self.schema["properties"]["source_conflicts"]["items"]
        self.assertEqual(set(conflict_schema["properties"]), set(relevance_schema.SOURCE_CONFLICT_FIELDS))

    def test_evidence_item_matches_relevance_schema(self):
        item = self.schema["properties"]["evidence"]["items"]
        self.assertEqual(set(item["properties"]), set(relevance_schema.EVIDENCE_FIELDS))
        self.assertEqual(set(item["required"]), set(relevance_schema.EVIDENCE_FIELDS))

    def test_confidence_enum_comes_from_relevance_schema(self):
        self.assertEqual(
            self.schema["properties"]["confidence"]["enum"], list(relevance_schema.CONFIDENCE_LEVELS),
        )

    def test_text_format_is_strict_json_schema(self):
        text_format = deep_prompt.build_text_format()
        self.assertEqual(text_format["format"]["type"], "json_schema")
        self.assertIs(text_format["format"]["strict"], True)
        self.assertEqual(text_format["format"]["schema"], self.schema)

    def test_each_call_returns_fresh_dict(self):
        first = deep_prompt.build_deep_output_schema()
        first["properties"]["category"]["type"] = "mutated"
        self.assertNotEqual(deep_prompt.build_deep_output_schema()["properties"]["category"]["type"], "mutated")


class PromptContentTests(unittest.TestCase):
    def test_prompt_version_is_set(self):
        self.assertEqual(deep_prompt.DEEP_PROMPT_VERSION, "procurement-deep-v1")

    def test_prompt_has_no_unformatted_placeholders(self):
        self.assertNotIn("{", deep_prompt.SYSTEM_PROMPT)
        self.assertNotIn("}", deep_prompt.SYSTEM_PROMPT)

    def test_prompt_states_no_commercial_analysis(self):
        for term in ("profit", "margin", "ROI", "sourcing/logistics/customs cost", "landed cost"):
            self.assertIn(term, deep_prompt.SYSTEM_PROMPT)

    def test_prompt_demands_verbatim_evidence(self):
        self.assertIn("verbatim", deep_prompt.SYSTEM_PROMPT)
        self.assertIn("Never translate", deep_prompt.SYSTEM_PROMPT)

    def test_prompt_lists_document_evidence_field(self):
        from src.ai import evidence_grounding
        self.assertIn(evidence_grounding.DEEP_DOCUMENT_EVIDENCE_FIELD, deep_prompt.SYSTEM_PROMPT)

    def test_prompt_states_absence_of_evidence_is_not_absence_of_fact(self):
        self.assertIn("ABSENCE OF EVIDENCE IS NOT EVIDENCE OF ABSENCE", deep_prompt.SYSTEM_PROMPT)
        self.assertIn("missing_information", deep_prompt.SYSTEM_PROMPT)

    def test_prompt_forbids_computing_estimated_value(self):
        self.assertIn("never\ncompute or infer a value yourself", deep_prompt.SYSTEM_PROMPT)

    def test_prompt_does_not_leak_pilot_case_ids(self):
        for case_id in ("case-119b5668646a58ec", "case-7e57fcaf39c956fa", "case-43670738adb7ac08"):
            self.assertNotIn(case_id, deep_prompt.SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
