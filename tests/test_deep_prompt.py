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

    def test_chunks_are_replaced_by_evidence_catalog(self):
        context = deep_prompt.build_deep_context(
            make_tender_context(), make_deep_analysis_context(), make_triage_result(),
        )
        self.assertNotIn("chunks", context)
        texts = [unit["exact_text"] for unit in context["evidence_catalog"] if unit["source_type"] == "document"]
        self.assertEqual(texts, ["Ноутбук, 10 шт.", "Гарантия 24 месяца."])

    def test_user_input_shows_catalog_ids_and_exact_text_once(self):
        context = deep_prompt.build_deep_context(
            make_tender_context(), make_deep_analysis_context(), make_triage_result(),
        )
        user_input = deep_prompt.build_user_input(context)
        self.assertIn("EVIDENCE_CATALOG", user_input)
        for unit in context["evidence_catalog"]:
            self.assertIn(f"[{unit['evidence_id']}", user_input)
        self.assertEqual(user_input.count("Поставка компьютерной техники"), 1)
        self.assertEqual(user_input.count("Ноутбук, 10 шт."), 1)
        self.assertNotIn('"chunks"', user_input)

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


def model_fields(fields) -> set:
    """Поля model output v4/v5: evidence -> evidence_ids."""
    return {"evidence_ids" if name == "evidence" else name for name in fields}


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.schema = deep_prompt.build_deep_output_schema()

    def test_top_level_fields_match_relevance_schema(self):
        expected = model_fields(relevance_schema.DEEP_ANALYSIS_COMMON_FIELDS)
        self.assertEqual(set(self.schema["properties"]), expected)
        self.assertEqual(set(self.schema["required"]), expected)
        self.assertNotIn("evidence", self.schema["properties"])
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
        expected = model_fields(relevance_schema.PROCUREMENT_ITEM_FIELDS)
        self.assertEqual(set(item_schema["properties"]), expected)
        self.assertEqual(set(item_schema["required"]), expected)

    def test_procurement_lot_schema_matches_relevance_schema(self):
        lot_schema = self.schema["properties"]["procurement"]["properties"]["lots"]["items"]
        expected = model_fields(relevance_schema.PROCUREMENT_LOT_FIELDS)
        self.assertEqual(set(lot_schema["properties"]), expected)
        self.assertEqual(set(lot_schema["required"]), expected)

    def test_brand_or_equivalent_schema_matches_relevance_schema(self):
        brand_schema = self.schema["properties"]["procurement"]["properties"]["brand_or_equivalent"]
        self.assertEqual(set(brand_schema["properties"]), set(relevance_schema.BRAND_OR_EQUIVALENT_FIELDS))
        self.assertEqual(set(brand_schema["type"]), {"object", "null"})

    def test_barrier_schema_matches_relevance_schema(self):
        barrier_schema = self.schema["properties"]["participation_barriers"]["items"]
        self.assertEqual(set(barrier_schema["properties"]), model_fields(relevance_schema.PARTICIPATION_BARRIER_FIELDS))
        self.assertEqual(
            barrier_schema["properties"]["type"]["enum"], list(relevance_schema.PARTICIPATION_BARRIER_TYPES),
        )
        self.assertEqual(
            barrier_schema["properties"]["severity"]["enum"], list(relevance_schema.CONFIDENCE_LEVELS),
        )

    def test_source_conflict_schema_matches_relevance_schema(self):
        conflict_schema = self.schema["properties"]["source_conflicts"]["items"]
        self.assertEqual(set(conflict_schema["properties"]), set(relevance_schema.SOURCE_CONFLICT_FIELDS))

    def test_every_evidence_field_is_a_list_of_id_strings(self):
        procurement_item = self.schema["properties"]["procurement"]["properties"]["items"]["items"]
        procurement_lot = self.schema["properties"]["procurement"]["properties"]["lots"]["items"]
        barrier = self.schema["properties"]["participation_barriers"]["items"]
        for container in (self.schema, procurement_item, procurement_lot, barrier):
            self.assertEqual(
                container["properties"]["evidence_ids"], {"type": "array", "items": {"type": "string"}},
            )

    def test_schema_has_no_place_for_model_written_evidence_text(self):
        def keys(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    yield key
                    yield from keys(value)
            elif isinstance(node, list):
                for value in node:
                    yield from keys(value)

        names = set(keys(self.schema))
        for forbidden in ("text", "evidence", "member_name", "download_id", "source_type", "field"):
            self.assertNotIn(forbidden, names)

    def test_over_max_length_evidence_is_rejected_by_validation(self):
        evidence = {
            "source_type": "announcement", "field": "title", "download_id": None, "member_name": None,
            "text": "x" * (relevance_schema.EVIDENCE_TEXT_MAX_CHARS + 1),
        }
        with self.assertRaises(ValueError):
            relevance_schema.validate_evidence_item(evidence)

    def test_triage_evidence_schema_is_unchanged(self):
        from src.ai import triage_prompt
        text_schema = triage_prompt.build_evidence_item_schema()["properties"]["text"]
        self.assertEqual(text_schema, {"type": "string"})

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
        self.assertEqual(deep_prompt.DEEP_PROMPT_VERSION, "procurement-deep-v5")

    def test_prompt_has_no_unformatted_placeholders(self):
        self.assertNotIn("{", deep_prompt.SYSTEM_PROMPT)
        self.assertNotIn("}", deep_prompt.SYSTEM_PROMPT)

    def test_prompt_states_no_commercial_analysis(self):
        for term in ("profit", "margin", "ROI", "sourcing/logistics/customs cost", "landed cost"):
            self.assertIn(term, deep_prompt.SYSTEM_PROMPT)

    def test_prompt_selects_evidence_by_id_only(self):
        for phrase in (
            "You NEVER write, copy, quote, translate, paraphrase, shorten or reconstruct source text",
            "Use only IDs that appear literally in EVIDENCE_CATALOG",
            "Never invent, guess, edit, extend or\n  combine an ID",
            "One or several IDs may support one conclusion",
            "smallest sufficient set",
            "the application materializes the exact source text",
            "may be paraphrased in your own words; evidence is selected only by ID",
            "evidence_ids",
        ):
            self.assertIn(phrase, deep_prompt.SYSTEM_PROMPT)

    def test_prompt_no_longer_asks_for_verbatim_quotes(self):
        for phrase in ("verbatim, contiguous", "SHORTEST sufficient verbatim", "evidence.text"):
            self.assertNotIn(phrase, deep_prompt.SYSTEM_PROMPT)

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
