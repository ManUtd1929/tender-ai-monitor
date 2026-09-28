"""
Тесты src.ai.relevance_schema. Чистая логика валидации, без БД и сети.

Запуск из корня проекта:
    python -m unittest tests.test_relevance_schema -v
"""

import copy
import unittest

from src.ai import relevance_schema as schema


def make_evidence(**overrides) -> dict:
    item = {
        "source_type": "document",
        "field": None,
        "download_id": 1,
        "member_name": "hraver.docx",
        "text": "Организация международных грузоперевозок автомобильным транспортом",
    }
    item.update(overrides)
    return item


def make_triage(**overrides) -> dict:
    result = {
        "relevance_status": "relevant",
        "opportunity_type": "logistics",
        "category": "international_freight",
        "confidence": "high",
        "reason": "Тендер на организацию международных автомобильных грузоперевозок",
        "requires_deep_analysis": True,
        "evidence": [make_evidence()],
    }
    result.update(overrides)
    return result


def make_procurement_block(**overrides) -> dict:
    block = {name: None for name in schema.PROCUREMENT_SCALAR_FIELDS}
    for name in schema.PROCUREMENT_LIST_FIELDS:
        block[name] = []
    block["subject"] = "Поставка компьютерной техники"
    block.update(overrides)
    return block


def make_logistics_block(**overrides) -> dict:
    block = {name: None for name in schema.LOGISTICS_SCALAR_FIELDS}
    block["service"] = "Международная перевозка"
    block.update(overrides)
    return block


def make_deep_result(opportunity_type="logistics", **overrides) -> dict:
    result = {
        "summary": "Тендер на международную перевозку груза",
        "opportunity_type": opportunity_type,
        "category": "international_freight",
        "why_interesting": "Логистическая услуга, релевантно CIO",
        "participation_barriers": [],
        "missing_information": [],
        "manual_review_required": False,
        "evidence": [make_evidence()],
        "procurement": None,
        "logistics": None,
    }
    if opportunity_type in ("procurement", "logistics_and_procurement"):
        result["procurement"] = make_procurement_block()
    if opportunity_type in ("logistics", "logistics_and_procurement"):
        result["logistics"] = make_logistics_block()
    result.update(overrides)
    return result


class EvidenceTests(unittest.TestCase):
    def test_valid_evidence_round_trips(self):
        item = make_evidence()
        validated = schema.validate_evidence_item(item)
        self.assertEqual(validated, item)

    def test_input_not_modified(self):
        item = make_evidence()
        original = copy.deepcopy(item)
        schema.validate_evidence_item(item)
        self.assertEqual(item, original)

    def test_unknown_source_type_rejected(self):
        with self.assertRaises(ValueError):
            schema.validate_evidence_item(make_evidence(source_type="webpage"))

    def test_blank_text_rejected(self):
        for text in ("", "   ", None):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    schema.validate_evidence_item(make_evidence(text=text))

    def test_too_long_text_rejected(self):
        long_text = "x" * (schema.EVIDENCE_TEXT_MAX_CHARS + 1)
        with self.assertRaises(ValueError):
            schema.validate_evidence_item(make_evidence(text=long_text))

    def test_max_length_text_allowed(self):
        text = "x" * schema.EVIDENCE_TEXT_MAX_CHARS
        schema.validate_evidence_item(make_evidence(text=text))

    def test_unknown_key_rejected(self):
        item = make_evidence()
        item["extra"] = "x"
        with self.assertRaises(ValueError):
            schema.validate_evidence_item(item)

    def test_missing_key_rejected(self):
        item = make_evidence()
        del item["field"]
        with self.assertRaises(ValueError):
            schema.validate_evidence_item(item)

    def test_bad_download_id_type_rejected(self):
        with self.assertRaises(ValueError):
            schema.validate_evidence_item(make_evidence(download_id="1"))

    def test_download_id_none_allowed(self):
        schema.validate_evidence_item(make_evidence(download_id=None, source_type="announcement"))

    def test_not_a_dict_rejected(self):
        with self.assertRaises(ValueError):
            schema.validate_evidence_item("not a dict")


class TriageValidationTests(unittest.TestCase):
    def test_valid_relevant_result(self):
        validated = schema.validate_triage_result(make_triage())
        self.assertEqual(validated["relevance_status"], "relevant")
        self.assertTrue(validated["requires_deep_analysis"])

    def test_input_not_modified(self):
        result = make_triage()
        original = copy.deepcopy(result)
        schema.validate_triage_result(result)
        self.assertEqual(result, original)

    def test_not_relevant_requires_deep_false(self):
        validated = schema.validate_triage_result(make_triage(
            relevance_status="not_relevant", requires_deep_analysis=False,
            opportunity_type="unrelated", evidence=[],
        ))
        self.assertFalse(validated["requires_deep_analysis"])

    def test_not_relevant_with_deep_true_rejected(self):
        with self.assertRaises(ValueError):
            schema.validate_triage_result(make_triage(
                relevance_status="not_relevant", requires_deep_analysis=True,
            ))

    def test_relevant_with_deep_false_rejected(self):
        with self.assertRaises(ValueError):
            schema.validate_triage_result(make_triage(requires_deep_analysis=False))

    def test_maybe_requires_deep_true(self):
        validated = schema.validate_triage_result(make_triage(relevance_status="maybe"))
        self.assertTrue(validated["requires_deep_analysis"])

    def test_maybe_with_deep_false_rejected(self):
        with self.assertRaises(ValueError):
            schema.validate_triage_result(make_triage(
                relevance_status="maybe", requires_deep_analysis=False,
            ))

    def test_unknown_relevance_status_rejected(self):
        with self.assertRaises(ValueError):
            schema.validate_triage_result(make_triage(relevance_status="probably"))

    def test_unknown_opportunity_type_rejected(self):
        with self.assertRaises(ValueError):
            schema.validate_triage_result(make_triage(opportunity_type="construction"))

    def test_unknown_confidence_rejected(self):
        with self.assertRaises(ValueError):
            schema.validate_triage_result(make_triage(confidence="very_high"))

    def test_blank_reason_rejected(self):
        with self.assertRaises(ValueError):
            schema.validate_triage_result(make_triage(reason=""))

    def test_category_none_allowed(self):
        validated = schema.validate_triage_result(make_triage(category=None))
        self.assertIsNone(validated["category"])

    def test_non_bool_requires_deep_analysis_rejected(self):
        with self.assertRaises(ValueError):
            schema.validate_triage_result(make_triage(requires_deep_analysis="true"))

    def test_unknown_top_level_key_rejected(self):
        result = make_triage()
        result["extra_field"] = "x"
        with self.assertRaises(ValueError):
            schema.validate_triage_result(result)

    def test_missing_top_level_key_rejected(self):
        result = make_triage()
        del result["reason"]
        with self.assertRaises(ValueError):
            schema.validate_triage_result(result)

    def test_invalid_evidence_item_rejected(self):
        with self.assertRaises(ValueError):
            schema.validate_triage_result(make_triage(evidence=[{"bad": "item"}]))

    def test_not_a_dict_rejected(self):
        with self.assertRaises(ValueError):
            schema.validate_triage_result(["not", "a", "dict"])

    def test_all_opportunity_types_accepted_when_consistent(self):
        # "unclear" допустим только с maybe, поэтому для него отдельные тесты ниже.
        for opportunity_type in schema.OPPORTUNITY_TYPES:
            if opportunity_type == "unclear":
                continue
            with self.subTest(opportunity_type=opportunity_type):
                schema.validate_triage_result(make_triage(opportunity_type=opportunity_type))

    def test_existing_opportunity_types_still_valid(self):
        for opportunity_type in (
            "logistics", "procurement", "logistics_and_procurement", "other_service", "unrelated",
        ):
            self.assertIn(opportunity_type, schema.OPPORTUNITY_TYPES)
            with self.subTest(opportunity_type=opportunity_type):
                schema.validate_triage_result(make_triage(opportunity_type=opportunity_type))

    def test_maybe_with_other_opportunity_types_still_valid(self):
        for opportunity_type in schema.OPPORTUNITY_TYPES:
            with self.subTest(opportunity_type=opportunity_type):
                validated = schema.validate_triage_result(make_triage(
                    relevance_status="maybe", opportunity_type=opportunity_type,
                ))
                self.assertEqual(validated["opportunity_type"], opportunity_type)

    def test_maybe_with_unclear_valid(self):
        validated = schema.validate_triage_result(make_triage(
            relevance_status="maybe", opportunity_type="unclear", requires_deep_analysis=True,
        ))
        self.assertEqual(validated["relevance_status"], "maybe")
        self.assertEqual(validated["opportunity_type"], "unclear")
        self.assertTrue(validated["requires_deep_analysis"])

    def test_relevant_with_unclear_rejected(self):
        with self.assertRaises(ValueError):
            schema.validate_triage_result(make_triage(
                relevance_status="relevant", opportunity_type="unclear",
            ))

    def test_not_relevant_with_unclear_rejected(self):
        with self.assertRaises(ValueError):
            schema.validate_triage_result(make_triage(
                relevance_status="not_relevant", opportunity_type="unclear",
                requires_deep_analysis=False, evidence=[],
            ))


class ProcurementBlockTests(unittest.TestCase):
    def test_valid_block_round_trips(self):
        block = make_procurement_block(items=[{"name": "Laptop", "quantity": "10"}])
        validated = schema.validate_procurement_block(block)
        self.assertEqual(validated["subject"], "Поставка компьютерной техники")
        self.assertEqual(validated["items"], [{"name": "Laptop", "quantity": "10"}])

    def test_unknown_key_rejected(self):
        block = make_procurement_block()
        block["extra"] = "x"
        with self.assertRaises(ValueError):
            schema.validate_procurement_block(block)

    def test_missing_key_rejected(self):
        block = make_procurement_block()
        del block["subject"]
        with self.assertRaises(ValueError):
            schema.validate_procurement_block(block)

    def test_none_list_field_becomes_empty_list(self):
        validated = schema.validate_procurement_block(make_procurement_block(items=None))
        self.assertEqual(validated["items"], [])

    def test_non_list_list_field_rejected(self):
        with self.assertRaises(ValueError):
            schema.validate_procurement_block(make_procurement_block(items="a laptop"))

    def test_non_str_scalar_field_rejected(self):
        with self.assertRaises(ValueError):
            schema.validate_procurement_block(make_procurement_block(subject=123))


class LogisticsBlockTests(unittest.TestCase):
    def test_valid_block_round_trips(self):
        block = make_logistics_block(origin="Yerevan", destination="Moscow")
        validated = schema.validate_logistics_block(block)
        self.assertEqual(validated["origin"], "Yerevan")
        self.assertEqual(validated["destination"], "Moscow")

    def test_unknown_key_rejected(self):
        block = make_logistics_block()
        block["extra"] = "x"
        with self.assertRaises(ValueError):
            schema.validate_logistics_block(block)

    def test_non_str_scalar_field_rejected(self):
        with self.assertRaises(ValueError):
            schema.validate_logistics_block(make_logistics_block(weight=100))


class ParticipationBarrierTests(unittest.TestCase):
    def test_known_barriers_accepted(self):
        validated = schema.validate_participation_barriers(list(schema.PARTICIPATION_BARRIER_TYPES))
        self.assertEqual(validated, list(schema.PARTICIPATION_BARRIER_TYPES))

    def test_unknown_barrier_rejected(self):
        with self.assertRaises(ValueError):
            schema.validate_participation_barriers(["needs_a_helicopter"])

    def test_not_a_list_rejected(self):
        with self.assertRaises(ValueError):
            schema.validate_participation_barriers("certification")


class DeepAnalysisValidationTests(unittest.TestCase):
    def test_valid_logistics_result(self):
        validated = schema.validate_deep_analysis_result(make_deep_result("logistics"))
        self.assertIsNotNone(validated["logistics"])
        self.assertIsNone(validated["procurement"])

    def test_valid_procurement_result(self):
        validated = schema.validate_deep_analysis_result(make_deep_result("procurement"))
        self.assertIsNotNone(validated["procurement"])
        self.assertIsNone(validated["logistics"])

    def test_valid_logistics_and_procurement_result(self):
        validated = schema.validate_deep_analysis_result(make_deep_result("logistics_and_procurement"))
        self.assertIsNotNone(validated["procurement"])
        self.assertIsNotNone(validated["logistics"])

    def test_valid_other_service_result_has_no_blocks(self):
        validated = schema.validate_deep_analysis_result(make_deep_result("other_service"))
        self.assertIsNone(validated["procurement"])
        self.assertIsNone(validated["logistics"])

    def test_valid_unrelated_result_has_no_blocks(self):
        validated = schema.validate_deep_analysis_result(make_deep_result("unrelated"))
        self.assertIsNone(validated["procurement"])
        self.assertIsNone(validated["logistics"])

    def test_valid_unclear_result_has_no_blocks(self):
        validated = schema.validate_deep_analysis_result(make_deep_result("unclear"))
        self.assertEqual(validated["opportunity_type"], "unclear")
        self.assertIsNone(validated["procurement"])
        self.assertIsNone(validated["logistics"])

    def test_procurement_present_for_unclear_type_rejected(self):
        with self.assertRaises(ValueError):
            schema.validate_deep_analysis_result(
                make_deep_result("unclear", procurement=make_procurement_block())
            )

    def test_input_not_modified(self):
        result = make_deep_result("logistics_and_procurement")
        original = copy.deepcopy(result)
        schema.validate_deep_analysis_result(result)
        self.assertEqual(result, original)

    def test_procurement_missing_for_procurement_type_rejected(self):
        with self.assertRaises(ValueError):
            schema.validate_deep_analysis_result(make_deep_result("procurement", procurement=None))

    def test_logistics_present_for_procurement_type_rejected(self):
        with self.assertRaises(ValueError):
            schema.validate_deep_analysis_result(
                make_deep_result("procurement", logistics=make_logistics_block())
            )

    def test_logistics_missing_for_logistics_type_rejected(self):
        with self.assertRaises(ValueError):
            schema.validate_deep_analysis_result(make_deep_result("logistics", logistics=None))

    def test_procurement_present_for_logistics_type_rejected(self):
        with self.assertRaises(ValueError):
            schema.validate_deep_analysis_result(
                make_deep_result("logistics", procurement=make_procurement_block())
            )

    def test_procurement_present_for_unrelated_type_rejected(self):
        with self.assertRaises(ValueError):
            schema.validate_deep_analysis_result(
                make_deep_result("unrelated", procurement=make_procurement_block())
            )

    def test_logistics_and_procurement_missing_one_block_rejected(self):
        with self.assertRaises(ValueError):
            schema.validate_deep_analysis_result(
                make_deep_result("logistics_and_procurement", logistics=None)
            )

    def test_unknown_opportunity_type_rejected(self):
        with self.assertRaises(ValueError):
            schema.validate_deep_analysis_result(make_deep_result("construction"))

    def test_blank_summary_rejected(self):
        with self.assertRaises(ValueError):
            schema.validate_deep_analysis_result(make_deep_result("logistics", summary=""))

    def test_blank_why_interesting_rejected(self):
        with self.assertRaises(ValueError):
            schema.validate_deep_analysis_result(make_deep_result("logistics", why_interesting="  "))

    def test_non_bool_manual_review_required_rejected(self):
        with self.assertRaises(ValueError):
            schema.validate_deep_analysis_result(
                make_deep_result("logistics", manual_review_required="no")
            )

    def test_unknown_participation_barrier_rejected(self):
        with self.assertRaises(ValueError):
            schema.validate_deep_analysis_result(
                make_deep_result("logistics", participation_barriers=["needs_a_wizard"])
            )

    def test_known_participation_barriers_accepted(self):
        validated = schema.validate_deep_analysis_result(make_deep_result(
            "procurement", participation_barriers=["official_dealer_required", "certification"],
        ))
        self.assertEqual(validated["participation_barriers"], ["official_dealer_required", "certification"])

    def test_missing_information_must_be_strings(self):
        with self.assertRaises(ValueError):
            schema.validate_deep_analysis_result(
                make_deep_result("logistics", missing_information=[None])
            )

    def test_missing_information_round_trips(self):
        validated = schema.validate_deep_analysis_result(make_deep_result(
            "logistics", missing_information=["estimated_value_amd", "delivery_deadline"],
        ))
        self.assertEqual(validated["missing_information"], ["estimated_value_amd", "delivery_deadline"])

    def test_unknown_top_level_key_rejected(self):
        result = make_deep_result("logistics")
        result["extra_field"] = "x"
        with self.assertRaises(ValueError):
            schema.validate_deep_analysis_result(result)

    def test_missing_top_level_key_rejected(self):
        result = make_deep_result("logistics")
        del result["summary"]
        with self.assertRaises(ValueError):
            schema.validate_deep_analysis_result(result)

    def test_invalid_evidence_item_rejected(self):
        with self.assertRaises(ValueError):
            schema.validate_deep_analysis_result(
                make_deep_result("logistics", evidence=[{"source_type": "document"}])
            )


if __name__ == "__main__":
    unittest.main()
