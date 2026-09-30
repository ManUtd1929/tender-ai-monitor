"""
Детерминированный Budget Guard: чистая функция без IO, БД и сети.

Оценка следующего вызова приходит из preflight и уже консервативна: cache hit не предполагается,
input тарифицируется по большей из ставок input / cache_write (pricing.calculate_conservative_cost).
Фактический расход в ledger включает cache_write_tokens по своей ставке.

Получает фактический расход месяца из ledger (spend_usd — считает вызывающий код через
ai_usage_repository.current_month_cost), настройки и КОНСЕРВАТИВНУЮ оценку следующего вызова
(src.ai.preflight) и возвращает структурированное решение. Порядок проверок:

    1. remaining = hard_limit - spend; remaining <= 0          -> blocked_hard_limit
    2. deep и оценка > MAX_SINGLE_DEEP_ESTIMATED_COST_USD       -> blocked_single_call_limit
    3. оценка > remaining                                      -> insufficient_budget_remaining
    4. spend >= soft_limit ("выше soft limit"):
         triage -> soft_limit_mode, allowed=True (дешёвая модель, оценка укладывается в бюджет)
         deep   -> soft_limit_mode, allowed=False, deep_status=budget_deferred
                   (приоритета пока нет, поэтому по умолчанию Deep откладывается)
    5. иначе                                                   -> allowed

Отложенный из-за бюджета тендер — НЕ not_relevant: deep_status=budget_deferred, relevance не
затрагивается. Hard limit — жёсткое требование: система не начинает вызов, чей консервативный
прогноз не помещается в остаток бюджета.
"""

from dataclasses import dataclass
from decimal import Decimal

from src.ai.budget_settings import BudgetSettings

ALLOWED = "allowed"
SOFT_LIMIT_MODE = "soft_limit_mode"
BLOCKED_HARD_LIMIT = "blocked_hard_limit"
BLOCKED_SINGLE_CALL_LIMIT = "blocked_single_call_limit"
INSUFFICIENT_BUDGET_REMAINING = "insufficient_budget_remaining"

ANALYSIS_TRIAGE = "triage"
ANALYSIS_DEEP = "deep"

DEEP_STATUS_BUDGET_DEFERRED = "budget_deferred"


@dataclass(frozen=True)
class BudgetDecision:
    decision: str
    allowed: bool
    reason: str
    analysis_type: str
    model: str
    spend_usd: Decimal
    remaining_usd: Decimal
    estimated_cost_usd: Decimal
    deep_status: str | None = None  # budget_deferred, если Deep не разрешён из-за бюджета


class BudgetGuard:
    def __init__(self, settings: BudgetSettings):
        self.settings = settings

    def check(
        self, spend_usd: Decimal, estimated_cost_usd: Decimal, analysis_type: str, model: str,
    ) -> BudgetDecision:
        if analysis_type not in (ANALYSIS_TRIAGE, ANALYSIS_DEEP):
            raise ValueError(f"Неизвестный analysis_type: {analysis_type!r}")
        spend = Decimal(spend_usd)
        estimate = Decimal(estimated_cost_usd)
        if spend < 0 or estimate < 0:
            raise ValueError("spend_usd и estimated_cost_usd не могут быть отрицательными")

        settings = self.settings
        remaining = settings.monthly_budget_usd - spend
        is_deep = analysis_type == ANALYSIS_DEEP

        def decide(decision, allowed, reason, deep_status=None):
            return BudgetDecision(
                decision, allowed, reason, analysis_type, model, spend, remaining, estimate, deep_status,
            )

        deferred = DEEP_STATUS_BUDGET_DEFERRED if is_deep else None

        if remaining <= 0:
            return decide(
                BLOCKED_HARD_LIMIT, False,
                f"расход ${spend} достиг hard limit ${settings.monthly_budget_usd}", deferred,
            )
        cap = settings.max_single_deep_estimated_cost_usd
        if is_deep and cap is not None and estimate > cap:
            return decide(
                BLOCKED_SINGLE_CALL_LIMIT, False,
                f"оценка Deep ${estimate} > лимита одного Deep-вызова ${cap}", deferred,
            )
        if estimate > remaining:
            return decide(
                INSUFFICIENT_BUDGET_REMAINING, False,
                f"оценка ${estimate} > остатка бюджета ${remaining}", deferred,
            )
        if spend >= settings.soft_limit_usd:
            if is_deep:
                return decide(
                    SOFT_LIMIT_MODE, False,
                    f"расход ${spend} >= soft limit ${settings.soft_limit_usd}: Deep отложен "
                    "(приоритезации пока нет)", DEEP_STATUS_BUDGET_DEFERRED,
                )
            return decide(
                SOFT_LIMIT_MODE, True,
                f"расход ${spend} >= soft limit ${settings.soft_limit_usd}: дешёвый triage разрешён, "
                f"оценка ${estimate} укладывается в остаток ${remaining}",
            )
        return decide(ALLOWED, True, f"оценка ${estimate} укладывается в остаток ${remaining}")
