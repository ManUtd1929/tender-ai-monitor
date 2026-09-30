"""
Versioned pricing config и детерминированный расчёт стоимости OpenAI-вызова по фактическому usage.

ЦЕНЫ — ОПЕРАЦИОННАЯ КОНФИГУРАЦИЯ ПРОЕКТА, а не факт о мире: провайдер может изменить тарифы.
Менять их нужно только здесь (MODEL_PRICING) и одновременно поднимать PRICING_VERSION —
версия пишется в usage ledger и в run artifact, поэтому старые записи остаются объяснимыми.
Цены НЕ запрашиваются из интернета в runtime.

Модель без цены -> CostEstimationError (тихой цены "по умолчанию" нет). Для экспериментов
можно передать overrides={"model-name": ModelPricing(...)}.

Разбивка input (все три части тарифицируются по РАЗНЫМ ставкам):

    ordinary_input = input_tokens - cached_tokens - cache_write_tokens
    cost = ordinary_input * input_rate + cached_tokens * cached_input_rate
           + cache_write_tokens * cache_write_rate + output_tokens * output_rate

Все ставки — USD за 1M токенов, деньги — Decimal, не float. Если cached + cache_write >
input_tokens, usage противоречив: CostEstimationError (ordinary_input никогда не бывает
отрицательным и не "подрезается" молча).

Семантика usage OpenAI Responses API: input_tokens УЖЕ включает cached_tokens и
cache_write_tokens (input_tokens_details), output_tokens УЖЕ включает reasoning_tokens.
Поэтому total_tokens нельзя умножать на одну цену, а reasoning_tokens отдельно не
тарифицируются (иначе двойной счёт) — они только сохраняются для наблюдаемости.
Отсутствующий cache_write_tokens (старые artifacts/ledger) трактуется как 0.

Long-context: если input_tokens > long_context_threshold_tokens (строго больше; ровно порог —
ещё short-context), ВЕСЬ запрос тарифицируется по long-ставкам модели (отдельная таблица
ставок, а не множитель).
"""

from dataclasses import dataclass
from decimal import Decimal

PRICING_VERSION = "2026-09-30"

TOKENS_PER_UNIT = Decimal(1_000_000)


class CostEstimationError(Exception):
    """Стоимость нельзя оценить (неизвестная модель или некорректный usage)."""


LONG_CONTEXT_THRESHOLD_TOKENS = 272_000


@dataclass(frozen=True)
class Rates:
    """Ставки USD за 1M токенов."""

    input_per_million: Decimal
    cached_input_per_million: Decimal
    cache_write_per_million: Decimal
    output_per_million: Decimal


@dataclass(frozen=True)
class ModelPricing:
    """short — обычные ставки; long — ставки для input_tokens > порога (None = отдельного тарифа нет)."""

    short: Rates
    long: Rates | None = None
    long_context_threshold_tokens: int | None = None

    def rates_for(self, input_tokens: int) -> Rates:
        threshold = self.long_context_threshold_tokens
        if self.long is not None and threshold is not None and input_tokens > threshold:
            return self.long
        return self.short


def _rates(input_, cached, cache_write, output) -> Rates:
    return Rates(Decimal(input_), Decimal(cached), Decimal(cache_write), Decimal(output))


MODEL_PRICING = {
    "gpt-5.6-luna": ModelPricing(
        short=_rates("0.20", "0.02", "0.25", "1.20"),
        long=_rates("0.40", "0.04", "0.50", "1.80"),
        long_context_threshold_tokens=LONG_CONTEXT_THRESHOLD_TOKENS,
    ),
    "gpt-5.6-terra": ModelPricing(
        short=_rates("2.00", "0.20", "2.50", "12.00"),
        long=_rates("4.00", "0.40", "5.00", "18.00"),
        long_context_threshold_tokens=LONG_CONTEXT_THRESHOLD_TOKENS,
    ),
}


def get_pricing(model: str, overrides: dict | None = None) -> ModelPricing:
    """CostEstimationError, если для модели нет цены ни в overrides, ни в MODEL_PRICING."""
    for table in (overrides or {}, MODEL_PRICING):
        if model in table:
            return table[model]
    raise CostEstimationError(
        f"Нет цены для модели {model!r}: добавьте её в src/ai/pricing.py::MODEL_PRICING "
        f"или передайте pricing override (известные модели: {sorted(MODEL_PRICING)})"
    )


def _token_count(name: str, value) -> int:
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CostEstimationError(f"{name} должен быть целым >= 0: {value!r}")
    return value


def split_input_tokens(input_tokens, cached_input_tokens=0, cache_write_tokens=0) -> tuple[int, int, int]:
    """
    (ordinary, cached, cache_write). CostEstimationError при отрицательных/нецелых значениях и
    при cached + cache_write > input_tokens (противоречивый usage не превращается в отрицательное).
    """
    input_tokens = _token_count("input_tokens", input_tokens)
    cached = _token_count("cached_input_tokens", cached_input_tokens)
    cache_write = _token_count("cache_write_tokens", cache_write_tokens)
    ordinary = input_tokens - cached - cache_write
    if ordinary < 0:
        raise CostEstimationError(
            f"Противоречивый usage: cached_tokens ({cached}) + cache_write_tokens ({cache_write}) "
            f"> input_tokens ({input_tokens})"
        )
    return ordinary, cached, cache_write


def calculate_cost(
    model: str,
    input_tokens,
    output_tokens,
    cached_input_tokens=0,
    overrides: dict | None = None,
    cache_write_tokens=0,
) -> Decimal:
    """Точная стоимость в USD (Decimal) по формуле модульного docstring."""
    pricing = get_pricing(model, overrides)
    ordinary, cached, cache_write = split_input_tokens(input_tokens, cached_input_tokens, cache_write_tokens)
    output_tokens = _token_count("output_tokens", output_tokens)

    rates = pricing.rates_for(ordinary + cached + cache_write)
    return (
        ordinary * rates.input_per_million
        + cached * rates.cached_input_per_million
        + cache_write * rates.cache_write_per_million
        + output_tokens * rates.output_per_million
    ) / TOKENS_PER_UNIT


def conservative_cost_breakdown(
    model: str, estimated_input_tokens, max_output_tokens, overrides: dict | None = None,
) -> dict:
    """
    Верхняя оценка стоимости ДО вызова (preflight), по компонентам. Кэш до запроса неизвестен, поэтому
    cache hit НЕ предполагается: весь оценённый input тарифицируется ОДИН раз по большей из ставок
    input / cache_write (для GPT-5.6 cache write дороже обычного input), output — по max_output_tokens
    (резерв на худший случай, включая reasoning; reasoning отдельно не тарифицируется). Long-context
    ставки выбираются по оценке input. Ставки — USD за 1M токенов.
    """
    pricing = get_pricing(model, overrides)
    input_tokens = _token_count("estimated_input_tokens", estimated_input_tokens)
    output_tokens = _token_count("max_output_tokens", max_output_tokens)
    rates = pricing.rates_for(input_tokens)
    input_rate = max(rates.input_per_million, rates.cache_write_per_million)
    input_cost = input_tokens * input_rate / TOKENS_PER_UNIT
    output_cost = output_tokens * rates.output_per_million / TOKENS_PER_UNIT
    return {
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "input_rate_per_million": input_rate, "output_rate_per_million": rates.output_per_million,
        "rate_tier": "long" if pricing.long is not None and rates is pricing.long else "short",
        "input_cost_usd": input_cost, "output_cost_usd": output_cost,
        "total_cost_usd": input_cost + output_cost,
    }


def calculate_conservative_cost(
    model: str, estimated_input_tokens, max_output_tokens, overrides: dict | None = None,
) -> Decimal:
    """Верхняя оценка стоимости ДО вызова (сумма conservative_cost_breakdown)."""
    return conservative_cost_breakdown(model, estimated_input_tokens, max_output_tokens, overrides)["total_cost_usd"]


def cost_from_usage(model: str, usage: dict, overrides: dict | None = None) -> Decimal:
    """
    Стоимость по dict из openai_triage.usage_dict (input_tokens, output_tokens, cached_tokens,
    cache_write_tokens, reasoning_tokens, total_tokens). Отсутствующий cache_write_tokens = 0.
    total_tokens и reasoning_tokens в расчёте не участвуют.
    """
    if usage is None:
        raise CostEstimationError("usage отсутствует: стоимость не оценивается без фактического usage")
    return calculate_cost(
        model,
        usage.get("input_tokens"),
        usage.get("output_tokens"),
        usage.get("cached_tokens"),
        overrides=overrides,
        cache_write_tokens=usage.get("cache_write_tokens"),
    )


USAGE_FIELDS = (
    "input_tokens", "cached_tokens", "cache_write_tokens", "output_tokens", "reasoning_tokens", "total_tokens",
)


def normalize_usage(usage: dict | None) -> dict:
    """Единый вид usage для artifacts: все USAGE_FIELDS, отсутствующие/None (в т.ч. legacy cache_write_tokens) -> 0."""
    usage = usage or {}
    return {name: usage.get(name) or 0 for name in USAGE_FIELDS}
