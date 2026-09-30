"""
Preflight-оценка размера и стоимости запроса ДО обращения к OpenAI: ни client, ни сеть здесь
не нужны — работает по dict из analyzer.build_request(...) (он client не создаёт).

Токены оцениваются ЭВРИСТИКОЙ, а не точным токенайзером: tiktoken в проекте нет и токенайзер
gpt-5.6-* локально недоступен. estimate_input_tokens = ceil(chars / CHARS_PER_TOKEN) — это
ОЦЕНКА, не точное число. CHARS_PER_TOKEN = 2.4 выбран консервативно (оценка токенов завышается)
по калибровке на реальных Deep-запусках (символов user input / input_tokens API):
    251 811 / 73 946 = 3.41;   139 559 / 39 572 = 3.53;   255 497 / 75 473 = 3.39;
    759 157 / 287 300 = 2.64 (худший наблюдённый, таблично-плотный, армянский текст).
Константа 2.4 ниже худшего наблюдения (~9% запаса). Для другого языка/шаблона наблюдаемое
значение может быть иным — поправьте константу и калибровку здесь.

Стоимость для Budget Guard тоже консервативная: cache hit НЕ предполагается (до запроса кэш
неизвестен), а весь оценённый input тарифицируется по большей из ставок input / cache_write
(у GPT-5.6 cache write дороже обычного input), output = max_output_tokens запроса (верхняя
граница, включая reasoning). См. pricing.calculate_conservative_cost. Реальную стоимость потом
записывает ledger по фактическому usage (ordinary / cached / cache_write раздельно).

Deep: если оценка input > MAX_DEEP_INPUT_TOKENS -> status deep_input_too_large; контекст НЕ
обрезается молча (позже — отдельный compact/selective context).
"""

import json
import math
from decimal import Decimal

from src.ai import pricing
from src.ai.budget_guard import ANALYSIS_DEEP, ANALYSIS_TRIAGE
from src.ai.budget_settings import BudgetSettings

CHARS_PER_TOKEN = Decimal("2.4")
# Запас на служебные токены запроса (схема Structured Outputs, обрамление), не учтённые в тексте.
REQUEST_OVERHEAD_TOKENS = 300
FALLBACK_MAX_OUTPUT_TOKENS = 16_000

STATUS_OK = "ok"
STATUS_DEEP_INPUT_TOO_LARGE = "deep_input_too_large"


def estimate_input_tokens(text: str) -> int:
    """Консервативная ОЦЕНКА числа токенов текста (не точный подсчёт), см. модульный docstring."""
    return math.ceil(Decimal(len(text)) / CHARS_PER_TOKEN)


def _request_text(request: dict) -> str:
    parts = [request.get("instructions") or "", request.get("input") or ""]
    text_format = request.get("text")
    if text_format is not None:
        parts.append(json.dumps(text_format, ensure_ascii=False, sort_keys=True))
    return "\n".join(str(part) for part in parts)


def estimate_request_tokens(request: dict) -> tuple[int, int]:
    """(оценка input tokens, оценка output tokens = max_output_tokens запроса или fallback)."""
    input_tokens = estimate_input_tokens(_request_text(request)) + REQUEST_OVERHEAD_TOKENS
    output_tokens = request.get("max_output_tokens") or FALLBACK_MAX_OUTPUT_TOKENS
    return input_tokens, output_tokens


def preflight_request(
    request: dict, model: str, analysis_type: str, settings: BudgetSettings,
    pricing_overrides: dict | None = None,
) -> dict:
    """
    {status, analysis_type, model, estimated_input_tokens, estimated_output_tokens,
    estimated_cost_usd (Decimal), estimated_input_cost_usd, estimated_output_cost_usd, rate_tier, max_input_tokens}. status: ok | deep_input_too_large.
    pricing.CostEstimationError для модели без цены пробрасывается (молчаливой цены нет).
    """
    if analysis_type not in (ANALYSIS_TRIAGE, ANALYSIS_DEEP):
        raise ValueError(f"Неизвестный analysis_type: {analysis_type!r}")
    input_tokens, output_tokens = estimate_request_tokens(request)
    breakdown = pricing.conservative_cost_breakdown(model, input_tokens, output_tokens, pricing_overrides)
    cost = breakdown["total_cost_usd"]
    max_input = settings.max_deep_input_tokens if analysis_type == ANALYSIS_DEEP else None
    status = STATUS_OK
    if max_input is not None and input_tokens > max_input:
        status = STATUS_DEEP_INPUT_TOO_LARGE
    return {
        "status": status,
        "analysis_type": analysis_type,
        "model": model,
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "estimated_cost_usd": cost,
        "estimated_input_cost_usd": breakdown["input_cost_usd"],
        "estimated_output_cost_usd": breakdown["output_cost_usd"],
        "rate_tier": breakdown["rate_tier"],
        "max_input_tokens": max_input,
    }
