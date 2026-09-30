"""
Тесты src.ai.openai_deep_analysis на fake client: реального OpenAI/сети нет (сеть блокирует
tests/safety_guards.py). Запрос, Structured Output, разбор ответа, application validation,
классы ошибок и bounded retry (переиспользованный из src.ai.openai_triage).

Запуск из корня проекта:
    python -m unittest tests.test_openai_deep_analysis -v
"""

import json
import unittest
from types import SimpleNamespace
from unittest import mock

import openai

try:
    import httpx2 as httpx  # openai SDK 3.x
except ImportError:  # pragma: no cover
    import httpx

from src.ai import deep_prompt, openai_deep_analysis, openai_triage, relevance_schema
from tests.test_deep_prompt import make_deep_analysis_context, make_tender_context, make_triage_result

API_KEY = "sk-test-secret-key-123"


TENDER_CONTEXT = make_tender_context()
DEEP_ANALYSIS_CONTEXT = make_deep_analysis_context()
TRIAGE_RESULT = make_triage_result()
DEEP_CONTEXT = deep_prompt.build_deep_context(TENDER_CONTEXT, DEEP_ANALYSIS_CONTEXT, TRIAGE_RESULT)


def unit_id(text: str) -> str:
    """ID unit'а каталога с точным текстом (ID генерирует Python, в тестах не хардкодятся)."""
    (unit,) = [u for u in DEEP_CONTEXT["evidence_catalog"] if u["exact_text"] == text]
    return unit["evidence_id"]


TITLE_ID = unit_id("Поставка компьютерной техники")
NOTEBOOK_ID = unit_id("Ноутбук, 10 шт.")
WARRANTY_ID = unit_id("Гарантия 24 месяца.")


def materialized(*evidence_ids) -> list:
    """Ожидаемый materialized evidence: текст берётся из каталога."""
    index = {u["evidence_id"]: u for u in DEEP_CONTEXT["evidence_catalog"]}
    return [
        {"evidence_id": i, "source_type": index[i]["source_type"], "field": index[i]["field"],
         "download_id": index[i]["download_id"], "member_name": index[i]["member_name"],
         "text": index[i]["exact_text"]}
        for i in evidence_ids
    ]


def procurement_block(**overrides) -> dict:
    block = {name: None for name in relevance_schema.PROCUREMENT_SCALAR_FIELDS}
    for name in relevance_schema.PROCUREMENT_OBJECT_FIELDS:
        block[name] = None
    for name in relevance_schema.PROCUREMENT_LIST_FIELDS:
        block[name] = []
    block["subject"] = "Поставка ноутбуков"
    block["items"] = [{
        "item_name": "Ноутбук", "lot_number": None, "quantity": "10", "unit": "шт.",
        "key_specifications": [], "brand_or_equivalent": None, "evidence_ids": [NOTEBOOK_ID],
    }]
    block.update(overrides)
    return block


def payload(**overrides) -> dict:
    result = {
        "summary": "Закупка ноутбуков для министерства.",
        "opportunity_type": "procurement",
        "category": "computer_equipment",
        "why_interesting": "Физический товар, потенциально можно импортировать.",
        "contracting_authority": "Министерство обороны",
        "procedure_code": None,
        "confidence": "high",
        "participation_barriers": [],
        "missing_information": [],
        "source_conflicts": [],
        "manual_review_required": False,
        "evidence_ids": [TITLE_ID],
        "procurement": procurement_block(),
        "logistics": None,
    }
    result.update(overrides)
    return result


UNCLEAR = dict(
    opportunity_type="unclear", category=None, procurement=None,
    summary="Недостаточно данных, чтобы определить предмет закупки.",
)


def make_response(result=None, text=None, status="completed", refusal=None, incomplete_reason=None):
    if text is None and result is not None:
        text = json.dumps(result, ensure_ascii=False)
    content = []
    if refusal is not None:
        content.append(SimpleNamespace(type="refusal", refusal=refusal))
    if text is not None:
        content.append(SimpleNamespace(type="output_text", text=text))
    return SimpleNamespace(
        id="resp_deep_1",
        status=status,
        incomplete_details=SimpleNamespace(reason=incomplete_reason) if incomplete_reason else None,
        output=[
            SimpleNamespace(type="reasoning", content=None),
            SimpleNamespace(type="message", content=content),
        ],
        usage=SimpleNamespace(
            input_tokens=1000, output_tokens=500, total_tokens=1500,
            input_tokens_details=SimpleNamespace(cached_tokens=100),
            output_tokens_details=SimpleNamespace(reasoning_tokens=300),
        ),
    )


def api_request():
    return httpx.Request("POST", "https://api.openai.com/v1/responses")


def status_error(error_class, status_code, code=None, message="error"):
    response = httpx.Response(status_code, request=api_request())
    return error_class(message, response=response, body={"code": code} if code else None)


class FakeResponses:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, *outcomes):
        self.responses = FakeResponses(outcomes)


def make_analyzer(*outcomes, **kwargs):
    client = FakeClient(*outcomes)
    sleeps = []
    kwargs.setdefault("environ", {})
    analyzer = openai_deep_analysis.OpenAIDeepAnalysisAnalyzer(client=client, sleep=sleeps.append, **kwargs)
    return analyzer, client, sleeps


class SettingsTests(unittest.TestCase):
    def test_defaults(self):
        analyzer = openai_deep_analysis.OpenAIDeepAnalysisAnalyzer(client=FakeClient(), environ={})
        self.assertEqual(analyzer.model, "gpt-5.6-luna")
        self.assertEqual(analyzer.reasoning_effort, "high")
        self.assertEqual(analyzer.prompt_version, "procurement-deep-v5")

    def test_shared_openai_model_env_var_is_used(self):
        analyzer = openai_deep_analysis.OpenAIDeepAnalysisAnalyzer(
            client=FakeClient(), environ={"OPENAI_MODEL": "shared-model"},
        )
        self.assertEqual(analyzer.model, "shared-model")

    def test_deep_specific_reasoning_env_var_is_separate_from_triage(self):
        analyzer = openai_deep_analysis.OpenAIDeepAnalysisAnalyzer(
            client=FakeClient(),
            environ={"OPENAI_TRIAGE_REASONING_EFFORT": "low", "OPENAI_DEEP_REASONING_EFFORT": "xhigh"},
        )
        self.assertEqual(analyzer.reasoning_effort, "xhigh")

    def test_explicit_arguments_beat_environment(self):
        analyzer = openai_deep_analysis.OpenAIDeepAnalysisAnalyzer(
            client=FakeClient(), model="m", reasoning_effort="low",
            environ={"OPENAI_MODEL": "other", "OPENAI_DEEP_REASONING_EFFORT": "high"},
        )
        self.assertEqual((analyzer.model, analyzer.reasoning_effort), ("m", "low"))

    def test_invalid_reasoning_effort_is_config_error(self):
        with self.assertRaises(openai_deep_analysis.DeepAnalysisError) as context:
            openai_deep_analysis.OpenAIDeepAnalysisAnalyzer(client=FakeClient(), reasoning_effort="turbo", environ={})
        self.assertEqual(context.exception.kind, openai_deep_analysis.KIND_CONFIG)

    def test_invalid_max_attempts_rejected(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                openai_deep_analysis.OpenAIDeepAnalysisAnalyzer(client=FakeClient(), max_attempts=value, environ={})


class MissingApiKeyTests(unittest.TestCase):
    def test_no_key_does_not_fail_at_construction(self):
        analyzer = openai_deep_analysis.OpenAIDeepAnalysisAnalyzer(environ={})
        self.assertFalse(analyzer.has_api_key)

    def test_no_key_fails_only_on_request_and_creates_no_client(self):
        analyzer = openai_deep_analysis.OpenAIDeepAnalysisAnalyzer(environ={})
        with mock.patch.object(openai, "OpenAI") as client_class:
            with self.assertRaises(openai_deep_analysis.DeepAnalysisError) as context:
                analyzer.deep_analyze(TENDER_CONTEXT, DEEP_ANALYSIS_CONTEXT, TRIAGE_RESULT)
        self.assertEqual(context.exception.kind, openai_deep_analysis.KIND_CONFIG)
        client_class.assert_not_called()

    def test_key_creates_client_without_sdk_retries(self):
        analyzer = openai_deep_analysis.OpenAIDeepAnalysisAnalyzer(
            environ={"OPENAI_API_KEY": API_KEY}, timeout=42.0,
        )
        with mock.patch.object(openai, "OpenAI") as client_class:
            analyzer.ensure_client()
        client_class.assert_called_once_with(api_key=API_KEY, timeout=42.0, max_retries=0)


class BuildRequestTests(unittest.TestCase):
    def setUp(self):
        self.analyzer, self.client, _ = make_analyzer()
        self.deep_context = DEEP_CONTEXT
        self.request = self.analyzer.build_request(self.deep_context)

    def test_model_and_reasoning(self):
        self.assertEqual(self.request["model"], "gpt-5.6-luna")
        self.assertEqual(self.request["reasoning"], {"effort": "high"})

    def test_structured_output_schema_is_used(self):
        self.assertEqual(self.request["text"], deep_prompt.build_text_format())
        self.assertIs(self.request["text"]["format"]["strict"], True)

    def test_no_tools(self):
        for name in ("tools", "tool_choice", "previous_response_id"):
            self.assertNotIn(name, self.request)
        self.assertIs(self.request["store"], False)

    def test_instructions_and_input(self):
        self.assertEqual(self.request["instructions"], deep_prompt.SYSTEM_PROMPT)
        self.assertEqual(self.request["input"], deep_prompt.build_user_input(self.deep_context))


class ValidResponseTests(unittest.TestCase):
    def assertAccepted(self, overrides):
        analyzer, client, _ = make_analyzer(make_response(payload(**overrides)))
        outcome = analyzer.deep_analyze_with_metadata(TENDER_CONTEXT, DEEP_ANALYSIS_CONTEXT, TRIAGE_RESULT)
        self.assertEqual(outcome["attempts"], 1)
        self.assertEqual(len(client.responses.calls), 1)
        return outcome

    def test_procurement_result(self):
        outcome = self.assertAccepted({})
        self.assertEqual(outcome["result"]["opportunity_type"], "procurement")
        self.assertIsNotNone(outcome["result"]["procurement"])

    def test_unclear_result_has_no_procurement_block(self):
        outcome = self.assertAccepted(UNCLEAR)
        self.assertIsNone(outcome["result"]["procurement"])

    def test_multiple_items_and_lots(self):
        block = procurement_block(
            total_lots=2,
            items=[
                {"item_name": "Ноутбук", "lot_number": "1", "quantity": "10", "unit": "шт.",
                 "key_specifications": [], "brand_or_equivalent": None, "evidence_ids": [NOTEBOOK_ID]},
                {"item_name": "Монитор", "lot_number": "2", "quantity": None, "unit": None,
                 "key_specifications": [], "brand_or_equivalent": None, "evidence_ids": [NOTEBOOK_ID, WARRANTY_ID]},
            ],
            lots=[
                {"lot_number": "1", "description": "Ноутбуки", "item_count": 1, "evidence_ids": [NOTEBOOK_ID]},
                {"lot_number": "2", "description": "Мониторы", "item_count": 1, "evidence_ids": []},
            ],
        )
        outcome = self.assertAccepted({"procurement": block})
        self.assertEqual(len(outcome["result"]["procurement"]["items"]), 2)
        self.assertIsNone(outcome["result"]["procurement"]["items"][1]["quantity"])
        procurement = outcome["result"]["procurement"]
        self.assertEqual(procurement["items"][1]["evidence"], materialized(NOTEBOOK_ID, WARRANTY_ID))
        self.assertEqual(procurement["lots"][0]["evidence"], materialized(NOTEBOOK_ID))
        self.assertEqual(procurement["lots"][1]["evidence"], [])

    def test_barrier_with_evidence_accepted(self):
        barrier = {
            "type": "official_dealer_required", "description": "Требуется официальный дилер",
            "severity": "medium", "evidence_ids": [WARRANTY_ID],
        }
        outcome = self.assertAccepted({"participation_barriers": [barrier]})
        expected = {**{k: v for k, v in barrier.items() if k != "evidence_ids"}, "evidence": materialized(WARRANTY_ID)}
        self.assertEqual(outcome["result"]["participation_barriers"], [expected])

    def test_source_conflict_round_trips(self):
        conflict = {
            "sources": ["announcement", "document"],
            "conflict_description": "Announcement описывает строительство, документ — товар",
            "impact": "Невозможно однозначно определить предмет закупки",
        }
        outcome = self.assertAccepted({"source_conflicts": [conflict], "manual_review_required": True})
        self.assertEqual(outcome["result"]["source_conflicts"], [conflict])

    def test_document_evidence_is_materialized_from_catalog(self):
        outcome = self.assertAccepted({"evidence_ids": [NOTEBOOK_ID]})
        self.assertEqual(outcome["result"]["evidence"], materialized(NOTEBOOK_ID))
        self.assertEqual(outcome["result"]["evidence"][0]["text"], "Ноутбук, 10 шт.")
        self.assertEqual(outcome["result"]["evidence"][0]["download_id"], 7)

    def test_several_ids_support_one_conclusion_in_given_order(self):
        outcome = self.assertAccepted({"evidence_ids": [WARRANTY_ID, TITLE_ID, NOTEBOOK_ID]})
        self.assertEqual(
            [e["text"] for e in outcome["result"]["evidence"]],
            ["Гарантия 24 месяца.", "Поставка компьютерной техники", "Ноутбук, 10 шт."],
        )

    def test_raw_model_output_is_kept_separately_and_holds_only_ids(self):
        outcome = self.assertAccepted({})
        self.assertEqual(outcome["raw_model_output"], payload())
        self.assertNotIn("evidence", outcome["raw_model_output"])
        self.assertNotIn("Ноутбук, 10 шт.", json.dumps(outcome["raw_model_output"], ensure_ascii=False))

    def test_usage_metadata(self):
        outcome = self.assertAccepted({})
        self.assertEqual(outcome["usage"], {
            "input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500,
            "cached_tokens": 100, "cache_write_tokens": 0, "reasoning_tokens": 300,
        })
        self.assertEqual(outcome["response_id"], "resp_deep_1")

    def test_deep_analyze_returns_result_only(self):
        analyzer, _, _ = make_analyzer(make_response(payload()))
        result = analyzer.deep_analyze(TENDER_CONTEXT, DEEP_ANALYSIS_CONTEXT, TRIAGE_RESULT)
        self.assertEqual(result["evidence"], materialized(TITLE_ID))
        self.assertEqual(result["procurement"]["items"][0]["evidence"], materialized(NOTEBOOK_ID))
        self.assertNotIn("evidence_ids", json.dumps(result))


class ApplicationValidationTests(unittest.TestCase):
    def assertRejected(self, response, kind=openai_deep_analysis.KIND_VALIDATION):
        analyzer, client, sleeps = make_analyzer(response)
        with self.assertRaises(openai_deep_analysis.DeepAnalysisError) as context:
            analyzer.deep_analyze(TENDER_CONTEXT, DEEP_ANALYSIS_CONTEXT, TRIAGE_RESULT)
        self.assertEqual(context.exception.kind, kind)
        self.assertEqual(len(client.responses.calls), 1, "validation/business ошибки не повторяются")
        self.assertEqual(sleeps, [])
        return context.exception

    def test_relevance_validation_is_called(self):
        analyzer, _, _ = make_analyzer(make_response(payload()))
        with mock.patch.object(
            relevance_schema, "validate_deep_analysis_result", wraps=relevance_schema.validate_deep_analysis_result,
        ) as validate:
            analyzer.deep_analyze(TENDER_CONTEXT, DEEP_ANALYSIS_CONTEXT, TRIAGE_RESULT)
        validate.assert_called_once()

    def test_logistics_opportunity_type_rejected(self):
        self.assertRejected(make_response(payload(opportunity_type="logistics")))

    def test_other_service_opportunity_type_rejected(self):
        self.assertRejected(make_response(payload(opportunity_type="other_service", category=None)))

    def test_missing_evidence_on_item_rejected(self):
        block = procurement_block(items=[{
            "item_name": "Ноутбук", "lot_number": None, "quantity": None, "unit": None,
            "key_specifications": [], "brand_or_equivalent": None, "evidence_ids": [],
        }])
        self.assertRejected(make_response(payload(procurement=block)))

    def test_missing_evidence_on_barrier_rejected(self):
        barrier = {
            "type": "certification", "description": "Требуется сертификат",
            "severity": "high", "evidence_ids": [],
        }
        self.assertRejected(make_response(payload(participation_barriers=[barrier])))

    def test_unsupported_barrier_type_rejected(self):
        barrier = {
            "type": "needs_a_wizard", "description": "x", "severity": "low",
            "evidence_ids": [NOTEBOOK_ID],
        }
        self.assertRejected(make_response(payload(participation_barriers=[barrier])))

    def test_category_required_for_procurement(self):
        self.assertRejected(make_response(payload(category=None)))

    def test_category_forbidden_for_unclear(self):
        self.assertRejected(make_response(payload(**{**UNCLEAR, "category": "computer_equipment"})))

    def test_unknown_evidence_id_rejected(self):
        error = self.assertRejected(make_response(payload(evidence_ids=["ev_doc_7_ffffff_0000"])))
        self.assertIn("неизвестный evidence_id", str(error))

    def test_id_with_altered_suffix_is_not_fuzzy_matched(self):
        self.assertRejected(make_response(payload(evidence_ids=[NOTEBOOK_ID[:-1] + "9"])))
        self.assertRejected(make_response(payload(evidence_ids=[NOTEBOOK_ID.upper()])))
        self.assertRejected(make_response(payload(evidence_ids=[" " + NOTEBOOK_ID])))

    def test_model_supplied_evidence_text_is_rejected(self):
        # Старый формат v3 (объекты с text) больше не принимается: у модели нет поля evidence.
        legacy = payload()
        legacy["evidence"] = [{"source_type": "announcement", "field": "title", "download_id": None,
                               "member_name": None, "text": "Поставка компьютерной техники"}]
        self.assertRejected(make_response(legacy))
        without_ids = {k: v for k, v in payload().items() if k != "evidence_ids"}
        self.assertRejected(make_response({**without_ids, "evidence": legacy["evidence"]}))

    def test_evidence_object_instead_of_id_rejected(self):
        self.assertRejected(make_response(payload(evidence_ids=[{"text": "Ноутбук, 10 шт."}])))

    def test_duplicate_id_in_one_list_rejected(self):
        error = self.assertRejected(make_response(payload(evidence_ids=[TITLE_ID, TITLE_ID])))
        self.assertIn("повторяется", str(error))

    def test_missing_evidence_ids_on_item_rejected(self):
        item = procurement_block()["items"][0]
        del item["evidence_ids"]
        self.assertRejected(make_response(payload(procurement=procurement_block(items=[item]))))

    def test_unknown_field_rejected(self):
        self.assertRejected(make_response({**payload(), "extra": 1}))

    def test_missing_field_rejected(self):
        broken = payload()
        del broken["summary"]
        self.assertRejected(make_response(broken))

    def test_not_json_rejected(self):
        self.assertRejected(make_response(text="not json"), openai_deep_analysis.KIND_INVALID_OUTPUT)

    def test_refusal(self):
        error = self.assertRejected(make_response(refusal="I cannot help"), openai_deep_analysis.KIND_REFUSAL)
        self.assertIn("I cannot help", str(error))

    def test_incomplete_response(self):
        self.assertRejected(
            make_response(text='{"summary', status="incomplete", incomplete_reason="max_output_tokens"),
            openai_deep_analysis.KIND_INCOMPLETE,
        )


class RetryTests(unittest.TestCase):
    def timeout_error(self):
        return openai.APITimeoutError(request=api_request())

    def test_timeout_is_retried_then_succeeds(self):
        analyzer, client, sleeps = make_analyzer(self.timeout_error(), make_response(payload()))
        outcome = analyzer.deep_analyze_with_metadata(TENDER_CONTEXT, DEEP_ANALYSIS_CONTEXT, TRIAGE_RESULT)
        self.assertEqual(outcome["attempts"], 2)
        self.assertEqual(sleeps, [analyzer.retry_base_delay])

    def test_retries_are_bounded(self):
        analyzer, client, sleeps = make_analyzer(*[self.timeout_error() for _ in range(5)])
        with self.assertRaises(openai_deep_analysis.DeepAnalysisError) as context:
            analyzer.deep_analyze(TENDER_CONTEXT, DEEP_ANALYSIS_CONTEXT, TRIAGE_RESULT)
        self.assertEqual(context.exception.kind, openai_deep_analysis.KIND_TIMEOUT)
        self.assertEqual(len(client.responses.calls), 3)

    def test_rate_limit_is_retried(self):
        analyzer, client, _ = make_analyzer(
            status_error(openai.RateLimitError, 429, code="rate_limit_exceeded"), make_response(payload()),
        )
        analyzer.deep_analyze(TENDER_CONTEXT, DEEP_ANALYSIS_CONTEXT, TRIAGE_RESULT)
        self.assertEqual(len(client.responses.calls), 2)

    def assertNotRetried(self, error, kind):
        analyzer, client, sleeps = make_analyzer(error, make_response(payload()))
        with self.assertRaises(openai_deep_analysis.DeepAnalysisError) as context:
            analyzer.deep_analyze(TENDER_CONTEXT, DEEP_ANALYSIS_CONTEXT, TRIAGE_RESULT)
        self.assertEqual(context.exception.kind, kind)
        self.assertEqual(len(client.responses.calls), 1)
        self.assertEqual(sleeps, [])

    def test_authentication_error_not_retried_and_message_has_no_key(self):
        error = status_error(openai.AuthenticationError, 401, message=f"Incorrect API key provided: {API_KEY}")
        analyzer, client, sleeps = make_analyzer(error, environ={"OPENAI_API_KEY": API_KEY})
        with self.assertRaises(openai_deep_analysis.DeepAnalysisError) as context:
            analyzer.deep_analyze(TENDER_CONTEXT, DEEP_ANALYSIS_CONTEXT, TRIAGE_RESULT)
        self.assertNotIn(API_KEY, str(context.exception))

    def test_insufficient_quota_not_retried(self):
        error = status_error(openai.RateLimitError, 429, code="insufficient_quota")
        self.assertNotRetried(error, openai_deep_analysis.KIND_RATE_LIMIT)


class TriageNotImplementedTests(unittest.TestCase):
    def test_triage_is_not_implemented(self):
        analyzer, client, _ = make_analyzer()
        with self.assertRaises(NotImplementedError):
            analyzer.triage({})
        self.assertEqual(client.responses.calls, [])


if __name__ == "__main__":
    unittest.main()
