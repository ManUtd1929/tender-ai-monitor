"""
Тесты src.ai.openai_triage на fake client: реального OpenAI/сети нет (сеть блокирует
tests/safety_guards.py). Запрос, Structured Output, разбор ответа, application validation,
классы ошибок и bounded retry.

Запуск из корня проекта:
    python -m unittest tests.test_openai_triage -v
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import openai

try:
    import httpx2 as httpx  # openai SDK 3.x
except ImportError:  # pragma: no cover
    import httpx

from src.ai import openai_triage, relevance_pipeline, relevance_schema, triage_prompt
from src.database import announcement_repository, document_repository, enrichment_repository
from src.database import analysis_repository
from tests.test_evaluation_dataset import make_announcement, make_enrichment

API_KEY = "sk-test-secret-key-123"

TRIAGE_CONTEXT = {
    "resource_url": "https://example.test/resource/1",
    "title": "Поставка мебели",
    "description": "Закупка офисной мебели",
    "cpv_codes": [],
    "documents": [{"download_id": 7, "member_name": "spec.docx", "file_type": "docx", "extraction_status": "success"}],
    "document_previews": [
        {"download_id": 7, "member_name": "spec.docx", "file_type": "docx", "preview_text": "Стол, стул", "truncated": False},
    ],
}


def evidence_item(**overrides) -> dict:
    item = {
        "source_type": "announcement", "field": "title", "download_id": None,
        "member_name": None, "text": "Поставка мебели",
    }
    item.update(overrides)
    return item


def payload(**overrides) -> dict:
    result = {
        "relevance_status": "relevant",
        "opportunity_type": "procurement",
        "category": "office_furniture",
        "confidence": "high",
        "reason": "Закупка физического товара.",
        "requires_deep_analysis": True,
        "evidence": [evidence_item()],
    }
    result.update(overrides)
    return result


OTHER_SERVICE = dict(
    relevance_status="not_relevant", opportunity_type="other_service", category="cleaning_services",
    requires_deep_analysis=False,
)
UNRELATED = dict(
    relevance_status="not_relevant", opportunity_type="unrelated", category=None, requires_deep_analysis=False,
)
MAYBE_UNCLEAR = dict(
    relevance_status="maybe", opportunity_type="unclear", category=None, confidence="low",
    requires_deep_analysis=True,
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
        id="resp_test_1",
        status=status,
        incomplete_details=SimpleNamespace(reason=incomplete_reason) if incomplete_reason else None,
        output=[
            SimpleNamespace(type="reasoning", content=None),
            SimpleNamespace(type="message", content=content),
        ],
        usage=SimpleNamespace(
            input_tokens=100, output_tokens=50, total_tokens=150,
            input_tokens_details=SimpleNamespace(cached_tokens=10),
            output_tokens_details=SimpleNamespace(reasoning_tokens=30),
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
    analyzer = openai_triage.OpenAITriageAnalyzer(client=client, sleep=sleeps.append, **kwargs)
    return analyzer, client, sleeps


class SettingsTests(unittest.TestCase):
    def test_defaults(self):
        analyzer = openai_triage.OpenAITriageAnalyzer(client=FakeClient(), environ={})
        self.assertEqual(analyzer.model, "gpt-5.6-terra")
        self.assertEqual(analyzer.reasoning_effort, "medium")
        self.assertEqual(analyzer.prompt_version, "procurement-v1")

    def test_environment_overrides(self):
        analyzer = openai_triage.OpenAITriageAnalyzer(
            client=FakeClient(),
            environ={"OPENAI_MODEL": "other-model", "OPENAI_TRIAGE_REASONING_EFFORT": "high"},
        )
        self.assertEqual((analyzer.model, analyzer.reasoning_effort), ("other-model", "high"))

    def test_explicit_arguments_beat_environment(self):
        analyzer = openai_triage.OpenAITriageAnalyzer(
            client=FakeClient(), model="m", reasoning_effort="low",
            environ={"OPENAI_MODEL": "other", "OPENAI_TRIAGE_REASONING_EFFORT": "high"},
        )
        self.assertEqual((analyzer.model, analyzer.reasoning_effort), ("m", "low"))

    def test_blank_environment_values_use_defaults(self):
        analyzer = openai_triage.OpenAITriageAnalyzer(
            client=FakeClient(), environ={"OPENAI_MODEL": "  ", "OPENAI_TRIAGE_REASONING_EFFORT": ""},
        )
        self.assertEqual((analyzer.model, analyzer.reasoning_effort), ("gpt-5.6-terra", "medium"))

    def test_invalid_reasoning_effort_is_config_error(self):
        with self.assertRaises(openai_triage.TriageError) as context:
            openai_triage.OpenAITriageAnalyzer(client=FakeClient(), reasoning_effort="turbo", environ={})
        self.assertEqual(context.exception.kind, openai_triage.KIND_CONFIG)

    def test_invalid_max_attempts_rejected(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                openai_triage.OpenAITriageAnalyzer(client=FakeClient(), max_attempts=value, environ={})


class MissingApiKeyTests(unittest.TestCase):
    def test_no_key_does_not_fail_at_construction(self):
        analyzer = openai_triage.OpenAITriageAnalyzer(environ={})
        self.assertFalse(analyzer.has_api_key)

    def test_no_key_fails_only_on_request_and_creates_no_client(self):
        analyzer = openai_triage.OpenAITriageAnalyzer(environ={})
        with mock.patch.object(openai, "OpenAI") as client_class:
            with self.assertRaises(openai_triage.TriageError) as context:
                analyzer.triage(TRIAGE_CONTEXT)
        self.assertEqual(context.exception.kind, openai_triage.KIND_CONFIG)
        self.assertIn("OPENAI_API_KEY", str(context.exception))
        client_class.assert_not_called()

    def test_key_creates_client_without_sdk_retries(self):
        analyzer = openai_triage.OpenAITriageAnalyzer(environ={"OPENAI_API_KEY": API_KEY}, timeout=42.0)
        self.assertTrue(analyzer.has_api_key)
        with mock.patch.object(openai, "OpenAI") as client_class:
            client = analyzer.ensure_client()
            self.assertIs(analyzer.ensure_client(), client)
        client_class.assert_called_once_with(api_key=API_KEY, timeout=42.0, max_retries=0)


class BuildRequestTests(unittest.TestCase):
    def setUp(self):
        self.analyzer, self.client, _ = make_analyzer()
        self.request = self.analyzer.build_request(TRIAGE_CONTEXT)

    def test_model_and_reasoning(self):
        self.assertEqual(self.request["model"], "gpt-5.6-terra")
        self.assertEqual(self.request["reasoning"], {"effort": "medium"})

    def test_structured_output_schema_is_used(self):
        self.assertEqual(self.request["text"], triage_prompt.build_text_format())
        self.assertIs(self.request["text"]["format"]["strict"], True)

    def test_no_tools(self):
        for name in ("tools", "tool_choice", "previous_response_id"):
            self.assertNotIn(name, self.request)
        self.assertIs(self.request["store"], False)

    def test_instructions_and_input(self):
        self.assertEqual(self.request["instructions"], triage_prompt.SYSTEM_PROMPT)
        self.assertEqual(self.request["input"], triage_prompt.build_user_input(TRIAGE_CONTEXT))

    def test_request_is_deterministic(self):
        again = self.analyzer.build_request(json.loads(json.dumps(TRIAGE_CONTEXT)))
        self.assertEqual(again, self.request)

    def test_context_is_not_mutated(self):
        before = json.dumps(TRIAGE_CONTEXT, sort_keys=True)
        self.analyzer.build_request(TRIAGE_CONTEXT)
        self.assertEqual(json.dumps(TRIAGE_CONTEXT, sort_keys=True), before)

    def test_logistics_values_not_possible_in_request_schema(self):
        enum = self.request["text"]["format"]["schema"]["properties"]["opportunity_type"]["enum"]
        self.assertNotIn("logistics", enum)
        self.assertNotIn("logistics_and_procurement", enum)

    def test_configured_model_and_effort_are_sent(self):
        analyzer, client, _ = make_analyzer(make_response(payload()), model="m2", reasoning_effort="high")
        analyzer.triage(TRIAGE_CONTEXT)
        sent = client.responses.calls[0]
        self.assertEqual(sent["model"], "m2")
        self.assertEqual(sent["reasoning"], {"effort": "high"})

    def test_call_passes_request_and_timeout(self):
        analyzer, client, _ = make_analyzer(make_response(payload()), timeout=33.0)
        analyzer.triage(TRIAGE_CONTEXT)
        sent = client.responses.calls[0]
        self.assertEqual(sent["timeout"], 33.0)
        self.assertEqual(sent["text"], triage_prompt.build_text_format())


class ValidResponseTests(unittest.TestCase):
    def assertAccepted(self, overrides):
        analyzer, client, _ = make_analyzer(make_response(payload(**overrides)))
        outcome = analyzer.triage_with_metadata(TRIAGE_CONTEXT)
        expected = payload(**overrides)
        self.assertEqual(outcome["result"], expected)
        self.assertEqual(outcome["attempts"], 1)
        self.assertEqual(len(client.responses.calls), 1)
        return outcome

    def test_procurement(self):
        outcome = self.assertAccepted({})
        self.assertEqual(outcome["result"]["opportunity_type"], "procurement")

    def test_other_service(self):
        self.assertAccepted(OTHER_SERVICE)

    def test_unrelated(self):
        self.assertAccepted(UNRELATED)

    def test_maybe_unclear(self):
        self.assertAccepted(MAYBE_UNCLEAR)

    def test_document_evidence_with_real_reference(self):
        evidence = [evidence_item(source_type="document", field="preview_text", download_id=7, member_name="spec.docx")]
        self.assertAccepted({"evidence": evidence})

    def test_usage_metadata(self):
        outcome = self.assertAccepted({})
        self.assertEqual(outcome["usage"], {
            "input_tokens": 100, "output_tokens": 50, "total_tokens": 150,
            "cached_tokens": 10, "reasoning_tokens": 30,
        })
        self.assertEqual(outcome["response_id"], "resp_test_1")

    def test_triage_returns_result_only(self):
        analyzer, _, _ = make_analyzer(make_response(payload()))
        self.assertEqual(analyzer.triage(TRIAGE_CONTEXT), payload())

    def test_usage_missing_is_none(self):
        response = make_response(payload())
        response.usage = None
        analyzer, _, _ = make_analyzer(response)
        self.assertIsNone(analyzer.triage_with_metadata(TRIAGE_CONTEXT)["usage"])


class ApplicationValidationTests(unittest.TestCase):
    def assertRejected(self, response, kind=openai_triage.KIND_VALIDATION):
        analyzer, client, sleeps = make_analyzer(response)
        with self.assertRaises(openai_triage.TriageError) as context:
            analyzer.triage(TRIAGE_CONTEXT)
        self.assertEqual(context.exception.kind, kind)
        self.assertEqual(len(client.responses.calls), 1, "validation/business ошибки не повторяются")
        self.assertEqual(sleeps, [])
        return context.exception

    def test_relevance_validation_is_called(self):
        analyzer, _, _ = make_analyzer(make_response(payload()))
        with mock.patch.object(
            relevance_schema, "validate_triage_result", wraps=relevance_schema.validate_triage_result,
        ) as validate:
            analyzer.triage(TRIAGE_CONTEXT)
        validate.assert_called_once()
        self.assertEqual(validate.call_args.args[0], payload())

    def test_schema_rule_violation_rejected(self):
        error = self.assertRejected(make_response(payload(relevance_status="not_relevant", opportunity_type="unrelated", category=None)))
        self.assertIn("requires_deep_analysis", str(error))

    def test_unknown_field_rejected(self):
        self.assertRejected(make_response({**payload(), "extra": 1}))

    def test_missing_field_rejected(self):
        broken = payload()
        del broken["reason"]
        self.assertRejected(make_response(broken))

    def test_logistics_opportunity_type_rejected(self):
        for value in ("logistics", "logistics_and_procurement"):
            with self.subTest(value=value):
                self.assertRejected(make_response(payload(opportunity_type=value)))

    def test_unknown_enum_rejected(self):
        self.assertRejected(make_response(payload(confidence="certain")))

    def test_relevant_must_be_procurement(self):
        self.assertRejected(make_response(payload(opportunity_type="other_service")))

    def test_maybe_must_be_unclear(self):
        self.assertRejected(make_response(payload(relevance_status="maybe", opportunity_type="procurement")))

    def test_not_relevant_cannot_be_procurement(self):
        self.assertRejected(make_response(payload(relevance_status="not_relevant", requires_deep_analysis=False)))

    def test_category_rules(self):
        cases = {
            "procurement without category": payload(category=None),
            "procurement with blank-like category": payload(category="Office Furniture"),
            "camel case category": payload(category="officeFurniture"),
            "unrelated with category": payload(**{**UNRELATED, "category": "x_y"}),
            "unclear with category": payload(**{**MAYBE_UNCLEAR, "category": "x_y"}),
            "other_service without category": payload(**{**OTHER_SERVICE, "category": None}),
        }
        for name, result in cases.items():
            with self.subTest(name):
                self.assertRejected(make_response(result))

    def test_new_snake_case_category_is_accepted(self):
        analyzer, _, _ = make_analyzer(make_response(payload(category="brand_new_goods_2")))
        self.assertEqual(analyzer.triage(TRIAGE_CONTEXT)["category"], "brand_new_goods_2")

    def test_invented_document_evidence_rejected(self):
        for item in (
            evidence_item(source_type="document", download_id=999, member_name="spec.docx"),
            evidence_item(source_type="document", download_id=7, member_name="other.docx"),
            evidence_item(source_type="document", download_id=None, member_name=None),
            evidence_item(source_type="announcement", download_id=7),
        ):
            with self.subTest(item=item):
                self.assertRejected(make_response(payload(evidence=[item])))

    def test_overlong_evidence_rejected(self):
        self.assertRejected(make_response(payload(evidence=[evidence_item(text="x" * 501)])))

    def test_not_json_rejected(self):
        error = self.assertRejected(make_response(text="not json"), openai_triage.KIND_INVALID_OUTPUT)
        self.assertEqual(error.usage["total_tokens"], 150)

    def test_code_fence_is_not_parsed(self):
        text = "```json\n" + json.dumps(payload()) + "\n```"
        self.assertRejected(make_response(text=text), openai_triage.KIND_INVALID_OUTPUT)

    def test_non_object_json_rejected(self):
        self.assertRejected(make_response(text="[1, 2]"), openai_triage.KIND_INVALID_OUTPUT)

    def test_empty_output_rejected(self):
        response = make_response(text="")
        self.assertRejected(response, openai_triage.KIND_INVALID_OUTPUT)

    def test_refusal(self):
        error = self.assertRejected(make_response(refusal="I cannot help"), openai_triage.KIND_REFUSAL)
        self.assertIn("I cannot help", str(error))
        self.assertEqual(error.response_id, "resp_test_1")

    def test_incomplete_response(self):
        error = self.assertRejected(
            make_response(text='{"relevance', status="incomplete", incomplete_reason="max_output_tokens"),
            openai_triage.KIND_INCOMPLETE,
        )
        self.assertIn("max_output_tokens", str(error))


class RetryTests(unittest.TestCase):
    def timeout_error(self):
        return openai.APITimeoutError(request=api_request())

    def test_timeout_is_retried_then_succeeds(self):
        analyzer, client, sleeps = make_analyzer(self.timeout_error(), make_response(payload()))
        outcome = analyzer.triage_with_metadata(TRIAGE_CONTEXT)
        self.assertEqual(outcome["attempts"], 2)
        self.assertEqual(len(client.responses.calls), 2)
        self.assertEqual(sleeps, [analyzer.retry_base_delay])

    def test_retries_are_bounded_for_timeout(self):
        analyzer, client, sleeps = make_analyzer(*[self.timeout_error() for _ in range(5)])
        with self.assertRaises(openai_triage.TriageError) as context:
            analyzer.triage(TRIAGE_CONTEXT)
        self.assertEqual(context.exception.kind, openai_triage.KIND_TIMEOUT)
        self.assertEqual(context.exception.attempts, 3)
        self.assertEqual(len(client.responses.calls), 3)
        self.assertEqual(sleeps, [2.0, 4.0])

    def test_max_attempts_is_configurable(self):
        analyzer, client, sleeps = make_analyzer(self.timeout_error(), self.timeout_error(), max_attempts=1)
        with self.assertRaises(openai_triage.TriageError):
            analyzer.triage(TRIAGE_CONTEXT)
        self.assertEqual(len(client.responses.calls), 1)
        self.assertEqual(sleeps, [])

    def test_rate_limit_is_retried(self):
        analyzer, client, _ = make_analyzer(
            status_error(openai.RateLimitError, 429, code="rate_limit_exceeded"), make_response(payload()),
        )
        analyzer.triage(TRIAGE_CONTEXT)
        self.assertEqual(len(client.responses.calls), 2)

    def test_rate_limit_after_retries_gives_rate_limit_kind(self):
        errors = [status_error(openai.RateLimitError, 429) for _ in range(3)]
        analyzer, client, _ = make_analyzer(*errors)
        with self.assertRaises(openai_triage.TriageError) as context:
            analyzer.triage(TRIAGE_CONTEXT)
        self.assertEqual(context.exception.kind, openai_triage.KIND_RATE_LIMIT)
        self.assertEqual(len(client.responses.calls), 3)

    def test_server_error_is_retried(self):
        analyzer, client, _ = make_analyzer(
            status_error(openai.InternalServerError, 500), status_error(openai.InternalServerError, 503),
            make_response(payload()),
        )
        outcome = analyzer.triage_with_metadata(TRIAGE_CONTEXT)
        self.assertEqual(outcome["attempts"], 3)

    def test_connection_error_is_retried(self):
        analyzer, client, _ = make_analyzer(
            openai.APIConnectionError(request=api_request()), make_response(payload()),
        )
        analyzer.triage(TRIAGE_CONTEXT)
        self.assertEqual(len(client.responses.calls), 2)

    def assertNotRetried(self, error, kind):
        analyzer, client, sleeps = make_analyzer(error, make_response(payload()))
        with self.assertRaises(openai_triage.TriageError) as context:
            analyzer.triage(TRIAGE_CONTEXT)
        self.assertEqual(context.exception.kind, kind)
        self.assertEqual(len(client.responses.calls), 1)
        self.assertEqual(sleeps, [])
        return context.exception

    def test_authentication_error_not_retried_and_message_has_no_key(self):
        error = status_error(openai.AuthenticationError, 401, message=f"Incorrect API key provided: {API_KEY}")
        raised = self.assertNotRetried(error, openai_triage.KIND_AUTHENTICATION)
        self.assertNotIn(API_KEY, str(raised))

    def test_permission_denied_not_retried(self):
        self.assertNotRetried(status_error(openai.PermissionDeniedError, 403), openai_triage.KIND_AUTHENTICATION)

    def test_bad_request_not_retried(self):
        self.assertNotRetried(status_error(openai.BadRequestError, 400), openai_triage.KIND_API_ERROR)

    def test_not_found_not_retried(self):
        self.assertNotRetried(status_error(openai.NotFoundError, 404), openai_triage.KIND_API_ERROR)

    def test_insufficient_quota_not_retried(self):
        error = status_error(openai.RateLimitError, 429, code="insufficient_quota")
        self.assertNotRetried(error, openai_triage.KIND_RATE_LIMIT)

    def test_api_key_is_redacted_from_error_messages(self):
        analyzer, _, _ = make_analyzer(
            status_error(openai.BadRequestError, 400, message=f"bad {API_KEY}"),
            environ={"OPENAI_API_KEY": API_KEY},
        )
        with self.assertRaises(openai_triage.TriageError) as context:
            analyzer.triage(TRIAGE_CONTEXT)
        self.assertNotIn(API_KEY, str(context.exception))

    def test_classify_api_error(self):
        self.assertEqual(openai_triage.classify_api_error(self.timeout_error()), ("timeout", True))
        self.assertEqual(openai_triage.classify_api_error(ValueError("x")), ("api_error", False))
        self.assertEqual(
            openai_triage.classify_api_error(status_error(openai.APIStatusError, 408)), ("api_error", True),
        )


class DeepAnalysisNotImplementedTests(unittest.TestCase):
    def test_deep_analyze_is_not_implemented(self):
        analyzer, client, _ = make_analyzer()
        with self.assertRaises(NotImplementedError):
            analyzer.deep_analyze({}, {}, {})
        self.assertEqual(client.responses.calls, [])


class PipelineCompatibilityTests(unittest.TestCase):
    """Analyzer подходит под interface relevance_pipeline.run_triage (временная БД, fake client)."""

    def test_run_triage_with_openai_analyzer(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            document_repository.init_db(db_path)
            enrichment_repository.init_db(db_path)
            announcement = make_announcement(1)
            announcement_repository.save_announcement(announcement, db_path)
            enrichment_repository.save_enrichment(announcement["resource_url"], make_enrichment(), db_path)

            analyzer, client, _ = make_analyzer(make_response(payload()))
            summary = relevance_pipeline.run_triage(
                analyzer, db_path=db_path, provider=analyzer.provider, model=analyzer.model,
                prompt_version=analyzer.prompt_version,
            )

            self.assertEqual(summary["success_count"], 1, summary["failures"])
            self.assertEqual(summary["relevant_count"], 1)
            stored = analysis_repository.get_triage(announcement["resource_url"], db_path=db_path)
            self.assertEqual(stored["provider"], "openai")
            self.assertEqual(stored["prompt_version"], "procurement-v1")
            self.assertEqual(len(client.responses.calls), 1)


if __name__ == "__main__":
    unittest.main()
