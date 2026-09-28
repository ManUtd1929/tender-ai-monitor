"""
Тесты src.ai.triage_prompt: strict JSON schema (procurement-MVP), детерминированная
сериализация и отсутствие golden set в prompt. Сети и OpenAI здесь нет.

Запуск из корня проекта:
    python -m unittest tests.test_triage_prompt -v
"""

import json
import unittest

from src.ai import golden_set, relevance_schema, triage_prompt


def make_context(**overrides) -> dict:
    context = {
        "resource_url": "https://example.test/resource/1",
        "title": "Поставка мебели",
        "description": "Закупка офисной мебели",
        "cpv_codes": [{"code": "39100000", "name": "Мебель"}],
        "documents": [],
        "document_previews": [],
    }
    context.update(overrides)
    return context


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.schema = triage_prompt.build_triage_output_schema()

    def test_top_level_fields_match_relevance_schema(self):
        self.assertEqual(set(self.schema["properties"]), set(relevance_schema.TRIAGE_FIELDS))
        self.assertEqual(set(self.schema["required"]), set(relevance_schema.TRIAGE_FIELDS))
        self.assertIs(self.schema["additionalProperties"], False)

    def test_evidence_item_matches_relevance_schema(self):
        item = self.schema["properties"]["evidence"]["items"]
        self.assertEqual(set(item["properties"]), set(relevance_schema.EVIDENCE_FIELDS))
        self.assertEqual(set(item["required"]), set(relevance_schema.EVIDENCE_FIELDS))
        self.assertIs(item["additionalProperties"], False)
        self.assertEqual(
            item["properties"]["source_type"]["enum"], list(relevance_schema.EVIDENCE_SOURCE_TYPES),
        )

    def test_enums_come_from_relevance_schema(self):
        properties = self.schema["properties"]
        self.assertEqual(properties["relevance_status"]["enum"], list(relevance_schema.RELEVANCE_STATUSES))
        self.assertEqual(properties["confidence"]["enum"], list(relevance_schema.CONFIDENCE_LEVELS))

    def test_logistics_values_are_not_available_to_model(self):
        enum = self.schema["properties"]["opportunity_type"]["enum"]
        self.assertEqual(set(enum), {"procurement", "other_service", "unrelated", "unclear"})
        self.assertNotIn("logistics", enum)
        self.assertNotIn("logistics_and_procurement", enum)

    def test_global_schema_keeps_logistics_values(self):
        self.assertIn("logistics", relevance_schema.OPPORTUNITY_TYPES)
        self.assertIn("logistics_and_procurement", relevance_schema.OPPORTUNITY_TYPES)

    def test_nullable_fields(self):
        properties = self.schema["properties"]
        self.assertIn("null", properties["category"]["type"])

    def test_text_format_is_strict_json_schema(self):
        text = triage_prompt.build_text_format()
        self.assertEqual(text["format"]["type"], "json_schema")
        self.assertIs(text["format"]["strict"], True)
        self.assertEqual(text["format"]["name"], triage_prompt.TRIAGE_OUTPUT_NAME)
        self.assertEqual(text["format"]["schema"], self.schema)

    def test_each_call_returns_fresh_dict(self):
        first = triage_prompt.build_triage_output_schema()
        first["properties"]["category"]["type"] = "mutated"
        self.assertNotEqual(triage_prompt.build_triage_output_schema()["properties"]["category"]["type"], "mutated")


class SerializationTests(unittest.TestCase):
    def test_key_order_does_not_change_output(self):
        first = {"a": 1, "b": {"x": 1, "y": 2}}
        second = {"b": {"y": 2, "x": 1}, "a": 1}
        self.assertEqual(
            triage_prompt.serialize_triage_context(first), triage_prompt.serialize_triage_context(second),
        )

    def test_non_ascii_text_is_preserved(self):
        text = triage_prompt.serialize_triage_context({"title": "Համակարգիչ / Поставка"})
        self.assertIn("Համակարգիչ / Поставка", text)

    def test_user_input_round_trips_context(self):
        context = make_context()
        user_input = triage_prompt.build_user_input(context)
        self.assertTrue(user_input.startswith("TENDER_CONTEXT (JSON):\n"))
        self.assertEqual(json.loads(user_input.split("\n", 1)[1]), context)


class PromptContentTests(unittest.TestCase):
    def test_prompt_version_is_set(self):
        self.assertEqual(triage_prompt.TRIAGE_PROMPT_VERSION, "procurement-v1")

    def test_prompt_states_key_policy(self):
        prompt = triage_prompt.SYSTEM_PROMPT
        for fragment in ("participation barriers", "NOT by itself a reason", "contradict", "maybe", "unclear"):
            self.assertIn(fragment, prompt)

    def test_prompt_does_not_leak_golden_set(self):
        cases = golden_set.load_golden_set()
        self.assertGreaterEqual(len(cases), 1)
        prompt = triage_prompt.SYSTEM_PROMPT
        for case in cases:
            with self.subTest(case=case["case_id"]):
                self.assertNotIn(case["case_id"], prompt)
                self.assertNotIn(case["resource_url"], prompt)
                self.assertNotIn(case["title"][:40], prompt)
                self.assertNotIn(case["expected"]["expected_reason"], prompt)
                if case["expected"]["category"]:
                    self.assertNotIn(case["expected"]["category"], prompt)


if __name__ == "__main__":
    unittest.main()
