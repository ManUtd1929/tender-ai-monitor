"""
OpenAI deep analysis analyzer (procurement-only MVP, STAGE 2): Responses API + Structured
Outputs. Mirrors src.ai.openai_triage (STAGE 1), reusing its transport/retry/error-class
building blocks instead of duplicating them (retry_create, classify_api_error, usage_dict,
output_parts, TriageError) — see module docstring of openai_triage for why that layer is
provider-generic.

Реализует analyzer interface из src.ai.relevance_pipeline: analyzer.deep_analyze(tender_context,
deep_analysis_context, triage_result) -> dict. triage() здесь намеренно не реализован (это
STAGE 1, отдельный модуль).

Ответ разбирается json.loads без чистки markdown/code fences, затем проходит
relevance_schema.validate_deep_analysis_result и MVP-правила этого модуля (opportunity_type
ограничен procurement/unclear, category snake_case для procurement и null для unclear,
evidence grounding по deep_context — src.ai.evidence_grounding.validate_deep_evidence_grounding).

Никаких tools/web search/file search/MCP в запросе нет: модель решает только по переданному
deep_context. Модуль не пишет в БД, не вызывает Telegram и не подключён к monitor.py/
relevance_pipeline (это отдельный явный шаг — см. src.ai.deep_analysis_evaluator).

Import безопасен без OPENAI_API_KEY: ошибка конфигурации (DeepAnalysisError kind="config")
возникает только при попытке создать client / выполнить запрос.
"""

import json
import logging
import os
import time

import openai

from src.ai import deep_prompt
from src.ai import evidence_grounding
from src.ai import openai_triage
from src.ai import relevance_schema

logger = logging.getLogger(__name__)

PROVIDER = openai_triage.PROVIDER
DEFAULT_MODEL = "gpt-5.6-terra"
DEFAULT_REASONING_EFFORT = "high"
REASONING_EFFORTS = openai_triage.REASONING_EFFORTS

DEFAULT_TIMEOUT_SECONDS = 180.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_BASE_DELAY_SECONDS = 2.0
# Deep context (полный текст документов) больше triage: разумный запас для structured output.
DEFAULT_MAX_OUTPUT_TOKENS = 16000

# Классы ошибок — те же имена, что и у STAGE 1 (openai_triage.KIND_*), переиспользуются как есть.
KIND_CONFIG = openai_triage.KIND_CONFIG
KIND_AUTHENTICATION = openai_triage.KIND_AUTHENTICATION
KIND_TIMEOUT = openai_triage.KIND_TIMEOUT
KIND_RATE_LIMIT = openai_triage.KIND_RATE_LIMIT
KIND_API_ERROR = openai_triage.KIND_API_ERROR
KIND_REFUSAL = openai_triage.KIND_REFUSAL
KIND_INCOMPLETE = openai_triage.KIND_INCOMPLETE
KIND_INVALID_OUTPUT = openai_triage.KIND_INVALID_OUTPUT
KIND_VALIDATION = openai_triage.KIND_VALIDATION

classify_api_error = openai_triage.classify_api_error

CATEGORY_PATTERN = openai_triage.CATEGORY_PATTERN


class DeepAnalysisError(openai_triage.TriageError):
    """Ошибка deep-analysis вызова. Та же форма, что и TriageError (kind/usage/response_id/attempts)."""


# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------

def load_settings(environ=None, model: str | None = None, reasoning_effort: str | None = None) -> dict:
    """
    {api_key, model, reasoning_effort}: явные аргументы > переменные окружения (OPENAI_API_KEY,
    OPENAI_MODEL — общая с triage, OPENAI_DEEP_REASONING_EFFORT — отдельная от
    OPENAI_TRIAGE_REASONING_EFFORT) > defaults (модель gpt-5.6-terra, effort high). Пустая
    строка считается "не задано". DeepAnalysisError(config) — недопустимый reasoning effort.
    """
    environ = os.environ if environ is None else environ

    def pick(explicit, name, default):
        for value in (explicit, environ.get(name)):
            if isinstance(value, str) and value.strip():
                return value.strip()
        return default

    effort = pick(reasoning_effort, "OPENAI_DEEP_REASONING_EFFORT", DEFAULT_REASONING_EFFORT)
    if effort not in REASONING_EFFORTS:
        raise DeepAnalysisError(
            KIND_CONFIG, f"Недопустимый reasoning effort {effort!r} (ожидается одно из {REASONING_EFFORTS})",
        )
    return {
        "api_key": pick(None, "OPENAI_API_KEY", None),
        "model": pick(model, "OPENAI_MODEL", DEFAULT_MODEL),
        "reasoning_effort": effort,
    }


# --------------------------------------------------------------------------
# application validation (после validate_deep_analysis_result)
# --------------------------------------------------------------------------

def _evidence_lists(result: dict) -> list:
    """Все evidence-списки validated deep result: top-level + barrier + procurement item/lot."""
    lists = [result["evidence"]]
    for barrier in result["participation_barriers"]:
        lists.append(barrier["evidence"])
    procurement = result["procurement"]
    if procurement is not None:
        for item in procurement["items"]:
            lists.append(item["evidence"])
        for lot in procurement["lots"]:
            lists.append(lot["evidence"])
    return lists


def validate_mvp_deep_analysis_result(result: dict, deep_context: dict | None = None) -> dict:
    """
    validate_deep_analysis_result + правила procurement deep-MVP. ValueError при нарушении:
        - opportunity_type только из deep_prompt.DEEP_MVP_OPPORTUNITY_TYPES (procurement/unclear);
          logistics деталей на этом этапе не бывает — сама relevance_schema уже требует
          logistics=None для этих двух типов;
        - category: snake_case, не null для procurement; null для unclear;
        - если передан deep_context: каждый evidence item (top-level, каждого participation
          barrier, каждой procurement item/lot) проходит
          evidence_grounding.validate_deep_evidence_grounding — источник существует и
          evidence.text — дословный фрагмент значения источника/восстановленного из chunks
          текста документа.
    Возвращает нормализованный результат validate_deep_analysis_result.
    """
    validated = relevance_schema.validate_deep_analysis_result(result)
    opportunity_type = validated["opportunity_type"]
    category = validated["category"]

    if opportunity_type not in deep_prompt.DEEP_MVP_OPPORTUNITY_TYPES:
        raise ValueError(f"opportunity_type={opportunity_type!r} не допускается в procurement deep-MVP")

    if opportunity_type == "procurement":
        if category is None or not CATEGORY_PATTERN.fullmatch(category):
            raise ValueError(f"MVP: category должна быть snake_case строкой, получено {category!r}")
    elif category is not None:
        raise ValueError(f"MVP: для opportunity_type={opportunity_type!r} category должна быть null: {category!r}")

    if deep_context is not None:
        for evidence_list in _evidence_lists(validated):
            evidence_grounding.validate_deep_evidence_grounding(evidence_list, deep_context)
    return validated


# --------------------------------------------------------------------------
# analyzer
# --------------------------------------------------------------------------

class OpenAIDeepAnalysisAnalyzer:
    prompt_version = deep_prompt.DEEP_PROMPT_VERSION
    provider = PROVIDER

    def __init__(
        self,
        client=None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY_SECONDS,
        max_output_tokens: int | None = DEFAULT_MAX_OUTPUT_TOKENS,
        sleep=time.sleep,
        environ=None,
    ):
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError(f"max_attempts должен быть целым >= 1: {max_attempts!r}")
        settings = load_settings(environ, model=model, reasoning_effort=reasoning_effort)
        self.model = settings["model"]
        self.reasoning_effort = settings["reasoning_effort"]
        self._api_key = settings["api_key"]
        self._client = client
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.retry_base_delay = retry_base_delay
        self.max_output_tokens = max_output_tokens
        self._sleep = sleep

    # -- client ------------------------------------------------------------

    @property
    def has_api_key(self) -> bool:
        """Можно ли выполнять запросы (ключ задан или client передан извне); key не раскрывается."""
        return bool(self._api_key) or self._client is not None

    def ensure_client(self):
        """Возвращает client; создаёт его при первом вызове. DeepAnalysisError(config) — нет API key."""
        if self._client is None:
            if not self._api_key:
                raise DeepAnalysisError(KIND_CONFIG, "OPENAI_API_KEY не задан: нельзя создать OpenAI client")
            self._client = openai.OpenAI(api_key=self._api_key, timeout=self.timeout, max_retries=0)
        return self._client

    # -- request -----------------------------------------------------------

    def build_request(self, deep_context: dict) -> dict:
        """Kwargs для client.responses.create; без tools, web/file search и MCP."""
        request = {
            "model": self.model,
            "instructions": deep_prompt.SYSTEM_PROMPT,
            "input": deep_prompt.build_user_input(deep_context),
            "reasoning": {"effort": self.reasoning_effort},
            "text": deep_prompt.build_text_format(),
            "store": False,
        }
        if self.max_output_tokens is not None:
            request["max_output_tokens"] = self.max_output_tokens
        return request

    def _redact(self, text: str) -> str:
        return text.replace(self._api_key, "[redacted]") if self._api_key else text

    def _create_with_retry(self, client, request: dict):
        """(response, attempts). Повторяет только transient ошибки, не более max_attempts."""
        return openai_triage.retry_create(
            client, request, timeout=self.timeout, max_attempts=self.max_attempts,
            retry_base_delay=self.retry_base_delay, sleep=self._sleep,
            error_class=DeepAnalysisError, redact=self._redact,
        )

    # -- public API --------------------------------------------------------

    def deep_analyze_with_metadata(self, tender_context: dict, deep_analysis_context: dict, triage_result: dict) -> dict:
        """
        {"result": validated deep-analysis dict, "usage": dict | None, "response_id": str | None,
        "attempts": int}. DeepAnalysisError (см. KIND_*) — любая ошибка; usage/response_id/
        attempts прикладываются, если известны.
        """
        client = self.ensure_client()
        deep_context = deep_prompt.build_deep_context(tender_context, deep_analysis_context, triage_result)
        request = self.build_request(deep_context)
        logger.info(
            "OpenAI deep analysis: model=%s effort=%s prompt=%s",
            self.model, self.reasoning_effort, self.prompt_version,
        )

        response, attempts = self._create_with_retry(client, request)
        usage = openai_triage.usage_dict(response)
        response_id = getattr(response, "id", None)
        details = {"usage": usage, "response_id": response_id, "attempts": attempts}

        result = self._parse_response(response, deep_context, details)
        logger.info(
            "OpenAI deep analysis результат: %s/%s manual_review_required=%s usage=%s",
            result["opportunity_type"], result["category"], result["manual_review_required"], usage,
        )
        return {"result": result, **details}

    def deep_analyze(self, tender_context: dict, deep_analysis_context: dict, triage_result: dict) -> dict:
        """Analyzer interface relevance_pipeline: validated deep-analysis result."""
        return self.deep_analyze_with_metadata(tender_context, deep_analysis_context, triage_result)["result"]

    def triage(self, triage_context):
        raise NotImplementedError("Triage не реализован здесь (это STAGE 2, см. src.ai.openai_triage)")

    # -- parsing -----------------------------------------------------------

    def _parse_response(self, response, deep_context: dict, details: dict) -> dict:
        status = getattr(response, "status", None)
        text, refusal = openai_triage.output_parts(response)

        if refusal is not None:
            raise DeepAnalysisError(KIND_REFUSAL, f"Модель отказалась отвечать: {refusal}", **details)
        if status == "incomplete":
            reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
            raise DeepAnalysisError(KIND_INCOMPLETE, f"Ответ неполный (incomplete: {reason})", **details)
        if status == "failed":
            code = getattr(getattr(response, "error", None), "code", None)
            raise DeepAnalysisError(KIND_API_ERROR, f"Response failed (code={code})", **details)
        if not text.strip():
            raise DeepAnalysisError(KIND_INVALID_OUTPUT, "Пустой structured output", **details)

        try:
            raw = json.loads(text)
        except json.JSONDecodeError as error:
            raise DeepAnalysisError(KIND_INVALID_OUTPUT, f"Ответ не является JSON: {error}", **details) from error
        if not isinstance(raw, dict):
            raise DeepAnalysisError(
                KIND_INVALID_OUTPUT, f"Ответ должен быть JSON-объектом: {type(raw).__name__}", **details
            )

        try:
            return validate_mvp_deep_analysis_result(raw, deep_context)
        except ValueError as error:
            raise DeepAnalysisError(KIND_VALIDATION, f"Application validation: {error}", **details) from error
