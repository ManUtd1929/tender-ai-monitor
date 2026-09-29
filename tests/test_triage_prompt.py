"""
Тесты src.ai.triage_prompt: strict JSON schema (procurement-MVP), детерминированная
сериализация и отсутствие golden set в prompt. Сети и OpenAI здесь нет.

Запуск из корня проекта:
    python -m unittest tests.test_triage_prompt -v
"""

import importlib
import inspect
import json
import sys
import unittest
from unittest import mock

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
        self.assertEqual(triage_prompt.TRIAGE_PROMPT_VERSION, "procurement-v2")

    def test_prompt_states_key_policy(self):
        prompt = triage_prompt.SYSTEM_PROMPT
        for fragment in ("participation barriers", "NOT by itself a reason", "contradict", "maybe", "unclear"):
            self.assertIn(fragment, prompt)

    def test_prompt_has_no_unformatted_placeholders(self):
        self.assertNotRegex(triage_prompt.SYSTEM_PROMPT, r"\{[a-z_]+\}")


class OpportunityTypePolicyTests(unittest.TestCase):
    def setUp(self):
        self.prompt = " ".join(triage_prompt.SYSTEM_PROMPT.split())

    def test_construction_works_are_unrelated(self):
        self.assertIn(
            '"unrelated": the primary subject is construction, reconstruction, repair, installation '
            "or other civil works",
            self.prompt,
        )
        self.assertIn("not_relevant + unrelated for construction, reconstruction, repair and civil works", self.prompt)

    def test_materials_installed_in_works_do_not_make_procurement(self):
        self.assertIn("consumed or installed as part of works do NOT turn a works tender", self.prompt)

    def test_professional_services_are_other_service(self):
        self.assertIn('"other_service": the primary subject is a non-physical service', self.prompt)
        for service in (
            "technical supervision", "design", "consulting", "expert review", "archiving",
            "event organization", "software licence", "audit", "training",
        ):
            with self.subTest(service=service):
                self.assertIn(service, self.prompt)
        self.assertIn("not_relevant + other_service for professional or non-goods services", self.prompt)

    def test_goods_are_procurement(self):
        self.assertIn('"procurement": the primary subject is the delivery, supply or acquisition of physical', self.prompt)

    def test_works_are_not_described_as_other_service(self):
        self.assertNotIn("a service or works, not goods", self.prompt)


class CanonicalCategoryTests(unittest.TestCase):
    GOODS = {
        "aviation_fuel", "blinds", "computer_equipment", "drinking_water", "household_goods",
        "infrastructure_goods", "laboratory_supplies", "medical_equipment", "modular_buildings",
        "pharmaceuticals_and_lab_supplies", "plants", "tires",
    }
    SERVICES = {
        "archiving_services", "design_and_cost_estimation", "expertise_services", "software_license",
        "sports_event_services", "technical_supervision",
    }

    def test_canonical_vocabulary_is_the_static_business_taxonomy(self):
        self.assertEqual(set(triage_prompt.GOODS_CATEGORIES), self.GOODS)
        self.assertEqual(set(triage_prompt.SERVICE_CATEGORIES), self.SERVICES)
        self.assertEqual(len(triage_prompt.GOODS_CATEGORIES), len(self.GOODS))
        self.assertEqual(len(triage_prompt.SERVICE_CATEGORIES), len(self.SERVICES))

    def test_canonical_values_are_valid_snake_case_categories(self):
        for category in triage_prompt.GOODS_CATEGORIES + triage_prompt.SERVICE_CATEGORIES:
            with self.subTest(category=category):
                self.assertRegex(category, r"[a-z][a-z0-9]*(_[a-z0-9]+)*")

    def test_prompt_lists_every_canonical_category_and_requires_exact_use(self):
        prompt = " ".join(triage_prompt.SYSTEM_PROMPT.split())
        self.assertIn("MUST use exactly this value", prompt)
        for category in self.GOODS | self.SERVICES:
            with self.subTest(category=category):
                self.assertIn(category, prompt)
        self.assertIn("goods: aviation_fuel,", prompt)
        self.assertIn("services: archiving_services,", prompt)

    def test_new_category_outside_vocabulary_remains_possible(self):
        prompt = " ".join(triage_prompt.SYSTEM_PROMPT.split())
        self.assertIn("none of the canonical categories fits", prompt)
        self.assertIn("create a concise normalized snake_case label", prompt)
        category_schema = triage_prompt.build_triage_output_schema()["properties"]["category"]
        self.assertNotIn("enum", category_schema)
        self.assertEqual(category_schema["type"], ["string", "null"])

    def test_categories_are_not_loaded_from_golden_set(self):
        source = inspect.getsource(triage_prompt)
        for forbidden in ("load_golden_set", "golden_set_procurement", "import golden_set", "evaluation/"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)
        self.assertNotIn("golden_set", " ".join(sys.modules["src.ai.triage_prompt"].__dict__.keys()))

    def test_prompt_is_independent_of_golden_set_contents(self):
        with mock.patch.object(golden_set, "load_golden_set", side_effect=AssertionError("golden set read")):
            reloaded = importlib.reload(triage_prompt)
        self.assertEqual(reloaded.SYSTEM_PROMPT, triage_prompt.SYSTEM_PROMPT)
        self.assertEqual(set(reloaded.GOODS_CATEGORIES), self.GOODS)


class EvidencePolicyTests(unittest.TestCase):
    def test_prompt_demands_verbatim_evidence(self):
        prompt = " ".join(triage_prompt.SYSTEM_PROMPT.split())
        for fragment in (
            "verbatim, contiguous fragment", "Never translate it, paraphrase it",
            "append an explanation or a translation", "ellipsis", "belong in reason, not in evidence",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, prompt)


class GoldenSetLeakTests(unittest.TestCase):
    def test_prompt_does_not_leak_golden_set(self):
        cases = golden_set.load_golden_set()
        self.assertGreaterEqual(len(cases), 1)
        prompt = triage_prompt.SYSTEM_PROMPT
        for case in cases:
            with self.subTest(case=case["case_id"]):
                self.assertNotIn(case["case_id"], prompt)
                self.assertNotIn(case["resource_url"], prompt)
                self.assertNotIn(case["title"], prompt)
                self.assertNotIn(case["title"][:40], prompt)
                self.assertNotIn(case["expected"]["expected_reason"], prompt)
                # Категории golden set намеренно могут совпадать с каноническим словарём
                # (это продуктовая таксономия), поэтому проверяется только остальное.


if __name__ == "__main__":
    unittest.main()
