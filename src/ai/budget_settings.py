"""
Настройки AI-бюджета и production-моделей из переменных окружения.

    MONTHLY_AI_BUDGET_USD=15            HARD-лимит месяца (UTC)
    AI_SOFT_LIMIT_USD=12                выше него Deep откладывается (budget_deferred)
    TRIAGE_MODEL=gpt-5.6-luna                 TRIAGE_REASONING_EFFORT=medium
    DEEP_PRIMARY_MODEL=gpt-5.6-luna           DEEP_PRIMARY_REASONING_EFFORT=high
    DEEP_FALLBACK_MODEL=gpt-5.6-terra         DEEP_FALLBACK_REASONING_EFFORT=high
    MAX_DEEP_INPUT_TOKENS=140000        больше -> deferred_input_too_large (контекст не обрезается)
    MAX_SINGLE_DEEP_ESTIMATED_COST_USD=0.75   (опционально; пусто/0 не допускается, "none" — выключить)
    MIN_DEEP_VALUE_AMD                  порог value-gate; НЕТ значения по умолчанию: если не
                                        задан, value-based auto-skip выключен

Terra здесь — НАСТРОЕННАЯ fallback-модель: автоматического второго вызова пока нет
(см. src.ai.escalation — только причины будущей эскалации).

Precedence моделей/effort (явная миграция; CLI --model у evaluator'ов всегда выше всего):
    TRIAGE_MODEL         > OPENAI_MODEL (legacy) > gpt-5.6-luna
    DEEP_PRIMARY_MODEL   > DEEP_MODEL (legacy)   > OPENAI_MODEL (legacy) > gpt-5.6-luna
    DEEP_FALLBACK_MODEL  > gpt-5.6-terra
    TRIAGE_REASONING_EFFORT       > OPENAI_TRIAGE_REASONING_EFFORT > medium
    DEEP_PRIMARY_REASONING_EFFORT > DEEP_REASONING_EFFORT (legacy) > OPENAI_DEEP_REASONING_EFFORT > high
    DEEP_FALLBACK_REASONING_EFFORT > high
Legacy DEEP_MODEL / DEEP_REASONING_EFFORT читаются только пока не задан DEEP_PRIMARY_*; fallback
они НЕ задают. Evaluator'ы (openai_triage / openai_deep_analysis.load_settings) используют ту же
цепочку для своей роли.

Пустая строка = "не задано". Некорректное значение -> BudgetConfigError (без молчаливых defaults).
"""

import os
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from src.ai import openai_triage


class BudgetConfigError(ValueError):
    """Некорректная настройка бюджета/моделей."""


DEFAULT_MONTHLY_BUDGET_USD = Decimal("15")
DEFAULT_SOFT_LIMIT_USD = Decimal("12")
DEFAULT_TRIAGE_MODEL = "gpt-5.6-luna"
DEFAULT_TRIAGE_REASONING_EFFORT = "medium"
DEFAULT_DEEP_PRIMARY_MODEL = "gpt-5.6-luna"
DEFAULT_DEEP_PRIMARY_REASONING_EFFORT = "high"
DEFAULT_DEEP_FALLBACK_MODEL = "gpt-5.6-terra"
DEFAULT_DEEP_FALLBACK_REASONING_EFFORT = "high"
DEFAULT_MAX_DEEP_INPUT_TOKENS = 140_000
DEFAULT_MAX_SINGLE_DEEP_ESTIMATED_COST_USD = Decimal("0.75")


@dataclass(frozen=True)
class BudgetSettings:
    monthly_budget_usd: Decimal = DEFAULT_MONTHLY_BUDGET_USD
    soft_limit_usd: Decimal = DEFAULT_SOFT_LIMIT_USD
    triage_model: str = DEFAULT_TRIAGE_MODEL
    triage_reasoning_effort: str = DEFAULT_TRIAGE_REASONING_EFFORT
    deep_primary_model: str = DEFAULT_DEEP_PRIMARY_MODEL
    deep_primary_reasoning_effort: str = DEFAULT_DEEP_PRIMARY_REASONING_EFFORT
    deep_fallback_model: str = DEFAULT_DEEP_FALLBACK_MODEL
    deep_fallback_reasoning_effort: str = DEFAULT_DEEP_FALLBACK_REASONING_EFFORT
    max_deep_input_tokens: int = DEFAULT_MAX_DEEP_INPUT_TOKENS
    max_single_deep_estimated_cost_usd: Decimal | None = DEFAULT_MAX_SINGLE_DEEP_ESTIMATED_COST_USD
    min_deep_value_amd: int | None = None


def _get(environ, *names):
    for name in names:
        value = environ.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _decimal(name: str, raw: str) -> Decimal:
    try:
        value = Decimal(raw)
    except InvalidOperation:
        raise BudgetConfigError(f"{name} должен быть числом: {raw!r}") from None
    if not value.is_finite() or value <= 0:
        raise BudgetConfigError(f"{name} должен быть положительным числом: {raw!r}")
    return value


def _positive_int(name: str, raw: str) -> int:
    try:
        value = int(raw)
    except ValueError:
        raise BudgetConfigError(f"{name} должен быть целым числом: {raw!r}") from None
    if value <= 0:
        raise BudgetConfigError(f"{name} должен быть > 0: {raw!r}")
    return value


def _effort(name: str, value: str) -> str:
    if value not in openai_triage.REASONING_EFFORTS:
        raise BudgetConfigError(
            f"{name}: недопустимый reasoning effort {value!r} (ожидается одно из {openai_triage.REASONING_EFFORTS})"
        )
    return value


def load_budget_settings(environ=None) -> BudgetSettings:
    """BudgetSettings из environ (по умолчанию os.environ); .env загружает вызывающий код."""
    environ = os.environ if environ is None else environ

    budget = DEFAULT_MONTHLY_BUDGET_USD
    if (raw := _get(environ, "MONTHLY_AI_BUDGET_USD")) is not None:
        budget = _decimal("MONTHLY_AI_BUDGET_USD", raw)
    soft = DEFAULT_SOFT_LIMIT_USD
    if (raw := _get(environ, "AI_SOFT_LIMIT_USD")) is not None:
        soft = _decimal("AI_SOFT_LIMIT_USD", raw)
    if soft > budget:
        raise BudgetConfigError(f"AI_SOFT_LIMIT_USD ({soft}) не может превышать MONTHLY_AI_BUDGET_USD ({budget})")

    max_input = DEFAULT_MAX_DEEP_INPUT_TOKENS
    if (raw := _get(environ, "MAX_DEEP_INPUT_TOKENS")) is not None:
        max_input = _positive_int("MAX_DEEP_INPUT_TOKENS", raw)

    single_cap = DEFAULT_MAX_SINGLE_DEEP_ESTIMATED_COST_USD
    if (raw := _get(environ, "MAX_SINGLE_DEEP_ESTIMATED_COST_USD")) is not None:
        single_cap = None if raw.lower() == "none" else _decimal("MAX_SINGLE_DEEP_ESTIMATED_COST_USD", raw)

    min_value = None
    if (raw := _get(environ, "MIN_DEEP_VALUE_AMD")) is not None:
        min_value = _positive_int("MIN_DEEP_VALUE_AMD", raw)

    return BudgetSettings(
        monthly_budget_usd=budget,
        soft_limit_usd=soft,
        triage_model=_get(environ, "TRIAGE_MODEL", "OPENAI_MODEL") or DEFAULT_TRIAGE_MODEL,
        triage_reasoning_effort=_effort(
            "TRIAGE_REASONING_EFFORT",
            _get(environ, "TRIAGE_REASONING_EFFORT", "OPENAI_TRIAGE_REASONING_EFFORT")
            or DEFAULT_TRIAGE_REASONING_EFFORT,
        ),
        deep_primary_model=_get(environ, "DEEP_PRIMARY_MODEL", "DEEP_MODEL", "OPENAI_MODEL")
        or DEFAULT_DEEP_PRIMARY_MODEL,
        deep_primary_reasoning_effort=_effort(
            "DEEP_PRIMARY_REASONING_EFFORT",
            _get(
                environ, "DEEP_PRIMARY_REASONING_EFFORT", "DEEP_REASONING_EFFORT", "OPENAI_DEEP_REASONING_EFFORT",
            ) or DEFAULT_DEEP_PRIMARY_REASONING_EFFORT,
        ),
        deep_fallback_model=_get(environ, "DEEP_FALLBACK_MODEL") or DEFAULT_DEEP_FALLBACK_MODEL,
        deep_fallback_reasoning_effort=_effort(
            "DEEP_FALLBACK_REASONING_EFFORT",
            _get(environ, "DEEP_FALLBACK_REASONING_EFFORT") or DEFAULT_DEEP_FALLBACK_REASONING_EFFORT,
        ),
        max_deep_input_tokens=max_input,
        max_single_deep_estimated_cost_usd=single_cap,
        min_deep_value_amd=min_value,
    )
