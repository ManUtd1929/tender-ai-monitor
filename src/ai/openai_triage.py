"""
OpenAI triage analyzer (procurement-only MVP): Responses API + Structured Outputs.

Реализует analyzer interface из src.ai.relevance_pipeline: analyzer.triage(triage_context)
-> dict. Дополнительно triage_with_metadata() возвращает usage/response_id/attempts
(нужны evaluator'у). deep_analyze здесь намеренно не реализован (отдельный этап).

Структура ответа обеспечивается strict JSON schema на стороне API (triage_prompt);
ответ разбирается json.loads без чистки markdown/code fences. После этого результат
всё равно проходит relevance_schema.validate_triage_result и MVP-правила этого модуля —
application validation остаётся последней защитой.

Никаких tools/web search/file search/MCP в запросе нет: модель решает только по
переданному triage_context. Модуль не пишет в БД, не вызывает Telegram и не подключён
к monitor.py.

Import безопасен без OPENAI_API_KEY: ошибка конфигурации (TriageError kind="config")
возникает только при попытке создать client / выполнить запрос.

Retry: SDK-retry отключён (max_retries=0), вместо него небольшой bounded retry здесь —
только для transient ошибок (timeout, connection, rate limit, 5xx). Validation,
refusal, auth и bad request не повторяются.
"""

import json
import logging
import os
import re
import time

import openai

from src.ai import relevance_schema
from src.ai import triage_prompt

logger = logging.getLogger(__name__)

PROVIDER = "openai"
DEFAULT_MODEL = "gpt-5.6-terra"
DEFAULT_REASONING_EFFORT = "medium"
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_BASE_DELAY_SECONDS = 2.0
MAX_RETRY_DELAY_SECONDS = 20.0
DEFAULT_MAX_OUTPUT_TOKENS = 8000

# Классы ошибок (TriageError.kind).
KIND_CONFIG = "config"
KIND_AUTHENTICATION = "authentication"
KIND_TIMEOUT = "timeout"
KIND_RATE_LIMIT = "rate_limit"
KIND_API_ERROR = "api_error"
KIND_REFUSAL = "refusal"
KIND_INCOMPLETE = "incomplete"
KIND_INVALID_OUTPUT = "invalid_structured_output"
KIND_VALIDATION = "validation"

CATEGORY_PATTERN = re.compile(r"[a-z][a-z0-9]*(_[a-z0-9]+)*")

_TRANSIENT_STATUS_CODES = (408, 409, 429)
_TRANSIENT_RESPONSE_ERROR_CODES = ("server_error", "rate_limit_exceeded")
_NON_TRANSIENT_RATE_LIMIT_CODES = ("insufficient_quota",)


class TriageError(Exception):
    """Ошибка triage-вызова. kind — один из KIND_*; usage/response_id/attempts — если известны."""

    def __init__(self, kind: str, message: str, *, usage=None, response_id=None, attempts=None):
        super().__init__(message)
        self.kind = kind
        self.usage = usage
        self.response_id = response_id
        self.attempts = attempts


# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------

def load_settings(environ=None, model: str | None = None, reasoning_effort: str | None = None) -> dict:
    """
    {api_key, model, reasoning_effort}: явные аргументы > переменные окружения
    (OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TRIAGE_REASONING_EFFORT) > defaults. Пустая
    строка считается "не задано". api_key может быть None — это не ошибка здесь.
    TriageError(config) — недопустимый reasoning effort.
    """
    environ = os.environ if environ is None else environ

    def pick(explicit, name, default):
        for value in (explicit, environ.get(name)):
            if isinstance(value, str) and value.strip():
                return value.strip()
        return default

    effort = pick(reasoning_effort, "OPENAI_TRIAGE_REASONING_EFFORT", DEFAULT_REASONING_EFFORT)
    if effort not in REASONING_EFFORTS:
        raise TriageError(
            KIND_CONFIG, f"Недопустимый reasoning effort {effort!r} (ожидается одно из {REASONING_EFFORTS})",
        )
    return {
        "api_key": pick(None, "OPENAI_API_KEY", None),
        "model": pick(model, "OPENAI_MODEL", DEFAULT_MODEL),
        "reasoning_effort": effort,
    }


# --------------------------------------------------------------------------
# application validation (после validate_triage_result)
# --------------------------------------------------------------------------

def validate_mvp_triage_result(result: dict, triage_context: dict | None = None) -> dict:
    """
    validate_triage_result + правила procurement-MVP. ValueError при нарушении:
        - opportunity_type только из triage_prompt.MVP_OPPORTUNITY_TYPES;
        - допустимы ровно комбинации relevant+procurement, not_relevant+other_service/
          unrelated, maybe+unclear;
        - category: snake_case, не null для procurement/other_service; null для
          unrelated/unclear;
        - если передан triage_context: evidence source_type="document" должен ссылаться на
          реальный download_id/member_name из context.documents, остальные источники не
          могут иметь download_id/member_name (нельзя выдумывать источники).
    Возвращает нормализованный результат validate_triage_result.
    """
    validated = relevance_schema.validate_triage_result(result)
    status = validated["relevance_status"]
    opportunity_type = validated["opportunity_type"]
    category = validated["category"]

    if opportunity_type not in triage_prompt.MVP_OPPORTUNITY_TYPES:
        raise ValueError(f"opportunity_type={opportunity_type!r} не допускается в procurement-MVP")

    allowed = {
        "relevant": ("procurement",),
        "not_relevant": ("other_service", "unrelated"),
        "maybe": (relevance_schema.UNCLEAR_OPPORTUNITY_TYPE,),
    }
    if opportunity_type not in allowed[status]:
        raise ValueError(
            f"MVP: relevance_status={status!r} допускает opportunity_type {allowed[status]}, "
            f"получено {opportunity_type!r}"
        )

    if opportunity_type in ("procurement", "other_service"):
        if category is None or not CATEGORY_PATTERN.fullmatch(category):
            raise ValueError(f"MVP: category должна быть snake_case строкой, получено {category!r}")
    elif category is not None:
        raise ValueError(f"MVP: для opportunity_type={opportunity_type!r} category должна быть null: {category!r}")

    if triage_context is not None:
        _check_evidence_grounding(validated["evidence"], triage_context)
    return validated


def _check_evidence_grounding(evidence: list, triage_context: dict) -> None:
    members_by_download = {}
    for document in triage_context.get("documents") or []:
        members_by_download.setdefault(document["download_id"], set()).add(document["member_name"])

    for item in evidence:
        if item["source_type"] != "document":
            if item["download_id"] is not None or item["member_name"] is not None:
                raise ValueError(
                    f"evidence: source_type={item['source_type']!r} не может иметь download_id/member_name"
                )
            continue
        download_id = item["download_id"]
        if download_id not in members_by_download:
            raise ValueError(f"evidence: download_id={download_id!r} отсутствует в triage_context.documents")
        member_name = item["member_name"]
        if member_name is not None and member_name not in members_by_download[download_id]:
            raise ValueError(
                f"evidence: member_name={member_name!r} отсутствует у download_id={download_id}"
            )


# --------------------------------------------------------------------------
# response helpers
# --------------------------------------------------------------------------

def _usage_dict(response) -> dict | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "cached_tokens": getattr(input_details, "cached_tokens", None),
        "reasoning_tokens": getattr(output_details, "reasoning_tokens", None),
    }


def _output_parts(response) -> tuple[str, str | None]:
    """(склеенный output_text, текст refusal или None) по response.output."""
    texts = []
    refusal = None
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", None) or []:
            content_type = getattr(content, "type", None)
            if content_type == "output_text" and getattr(content, "text", None) is not None:
                texts.append(content.text)
            elif content_type == "refusal":
                refusal = getattr(content, "refusal", None) or "(refusal without text)"
    return "".join(texts), refusal


def classify_api_error(error: Exception) -> tuple[str, bool]:
    """(kind, retryable) для исключения SDK."""
    if isinstance(error, openai.APITimeoutError):
        return KIND_TIMEOUT, True
    if isinstance(error, openai.APIConnectionError):
        return KIND_API_ERROR, True
    if isinstance(error, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return KIND_AUTHENTICATION, False
    if isinstance(error, openai.RateLimitError):
        # Исчерпанная квота не пройдёт от повтора; обычный rate limit — transient.
        return KIND_RATE_LIMIT, getattr(error, "code", None) not in _NON_TRANSIENT_RATE_LIMIT_CODES
    if isinstance(error, openai.APIStatusError):
        return KIND_API_ERROR, error.status_code >= 500 or error.status_code in _TRANSIENT_STATUS_CODES
    return KIND_API_ERROR, False


# --------------------------------------------------------------------------
# analyzer
# --------------------------------------------------------------------------

class OpenAITriageAnalyzer:
    prompt_version = triage_prompt.TRIAGE_PROMPT_VERSION
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
        """Возвращает client; создаёт его при первом вызове. TriageError(config) — нет API key."""
        if self._client is None:
            if not self._api_key:
                raise TriageError(KIND_CONFIG, "OPENAI_API_KEY не задан: нельзя создать OpenAI client")
            # max_retries=0: повторы делает _create_with_retry (ограниченно и тестируемо).
            self._client = openai.OpenAI(api_key=self._api_key, timeout=self.timeout, max_retries=0)
        return self._client

    # -- request -----------------------------------------------------------

    def build_request(self, triage_context: dict) -> dict:
        """Kwargs для client.responses.create; без tools, web/file search и MCP."""
        request = {
            "model": self.model,
            "instructions": triage_prompt.SYSTEM_PROMPT,
            "input": triage_prompt.build_user_input(triage_context),
            "reasoning": {"effort": self.reasoning_effort},
            "text": triage_prompt.build_text_format(),
            "store": False,
        }
        if self.max_output_tokens is not None:
            request["max_output_tokens"] = self.max_output_tokens
        return request

    def _redact(self, text: str) -> str:
        return text.replace(self._api_key, "[redacted]") if self._api_key else text

    def _create_with_retry(self, client, request: dict):
        """(response, attempts). Повторяет только transient ошибки, не более max_attempts."""
        for attempt in range(1, self.max_attempts + 1):
            try:
                return client.responses.create(**request, timeout=self.timeout), attempt
            except openai.OpenAIError as error:
                kind, retryable = classify_api_error(error)
                if retryable and attempt < self.max_attempts:
                    delay = min(self.retry_base_delay * 2 ** (attempt - 1), MAX_RETRY_DELAY_SECONDS)
                    logger.warning(
                        "OpenAI transient ошибка (%s, %s), попытка %d/%d, повтор через %.1f с",
                        kind, type(error).__name__, attempt, self.max_attempts, delay,
                    )
                    self._sleep(delay)
                    continue
                if kind == KIND_AUTHENTICATION:
                    message = "OpenAI отклонил API key/доступ (authentication/permission)"
                else:
                    message = self._redact(f"{type(error).__name__}: {error}")
                logger.error("OpenAI вызов не удался (%s) после %d попыток: %s", kind, attempt, message)
                raise TriageError(kind, message, attempts=attempt) from error

    # -- public API --------------------------------------------------------

    def triage_with_metadata(self, triage_context: dict) -> dict:
        """
        {"result": validated triage dict, "usage": dict | None, "response_id": str | None,
        "attempts": int}. TriageError (см. KIND_*) — любая ошибка; usage/response_id/
        attempts прикладываются, если известны.
        """
        client = self.ensure_client()
        request = self.build_request(triage_context)
        logger.info("OpenAI triage: model=%s effort=%s prompt=%s", self.model, self.reasoning_effort, self.prompt_version)

        response, attempts = self._create_with_retry(client, request)
        usage = _usage_dict(response)
        response_id = getattr(response, "id", None)
        details = {"usage": usage, "response_id": response_id, "attempts": attempts}

        result = self._parse_response(response, triage_context, details)
        logger.info(
            "OpenAI triage результат: %s/%s category=%s confidence=%s usage=%s",
            result["relevance_status"], result["opportunity_type"], result["category"],
            result["confidence"], usage,
        )
        return {"result": result, **details}

    def triage(self, triage_context: dict) -> dict:
        """Analyzer interface relevance_pipeline: validated triage result."""
        return self.triage_with_metadata(triage_context)["result"]

    def deep_analyze(self, tender_context, deep_analysis_context, triage_result):
        raise NotImplementedError("Deep analysis не реализован на этом этапе (только triage)")

    # -- parsing -----------------------------------------------------------

    def _parse_response(self, response, triage_context: dict, details: dict) -> dict:
        status = getattr(response, "status", None)
        text, refusal = _output_parts(response)

        if refusal is not None:
            raise TriageError(KIND_REFUSAL, f"Модель отказалась отвечать: {refusal}", **details)
        if status == "incomplete":
            reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
            raise TriageError(KIND_INCOMPLETE, f"Ответ неполный (incomplete: {reason})", **details)
        if status == "failed":
            code = getattr(getattr(response, "error", None), "code", None)
            raise TriageError(KIND_API_ERROR, f"Response failed (code={code})", **details)
        if not text.strip():
            raise TriageError(KIND_INVALID_OUTPUT, "Пустой structured output", **details)

        try:
            raw = json.loads(text)
        except json.JSONDecodeError as error:
            raise TriageError(KIND_INVALID_OUTPUT, f"Ответ не является JSON: {error}", **details) from error
        if not isinstance(raw, dict):
            raise TriageError(KIND_INVALID_OUTPUT, f"Ответ должен быть JSON-объектом: {type(raw).__name__}", **details)

        try:
            return validate_mvp_triage_result(raw, triage_context)
        except ValueError as error:
            raise TriageError(KIND_VALIDATION, f"Application validation: {error}", **details) from error
