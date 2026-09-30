"""
Композиция «допускать ли Deep Analysis (Terra) для этого тендера»: единственная точка, где
сходятся commercial gate, preflight-оценка и Budget Guard. Модуль ничего не вызывает у
провайдера: analyzer нужен только ради build_request (он не создаёт client), а расход месяца
передаёт вызывающий код (ai_usage_repository.current_month_cost). Сам в monitor.py не подключён.

Будущий поток:  triage_result -> commercial_gate.evaluate -> (здесь) preflight -> budget_guard
-> deep analyzer. Зависимости однонаправленные: commercial_gate, preflight и budget_guard друг
о друге не знают.

deep_analysis_status:
    ready_for_deep            можно вызывать Deep (call_allowed=True)
    skipped_low_value         gate: skip_low_value        (relevance_status не меняется)
    skipped_too_small         gate: skip_too_small
    manual_review_required    gate: manual_review
    insufficient_information  gate: insufficient_information
    deferred_input_too_large  gate прошёл, но оценка input > MAX_DEEP_INPUT_TOKENS (контекст
                              не обрезается, модель не вызывается; reason_code
                              deep_input_limit_exceeded; позже — compact/selective context)
    budget_deferred           Budget Guard не разрешил Deep (soft/hard limit, остаток, cap
                              одного вызова) — тендер НЕ not_relevant, его можно вернуть позже
    pricing_unknown           для модели нет цены: консервативную стоимость не оценить, Deep не вызывается
Порядок: gate -> preflight (размер) -> Budget Guard.
"""

from src.ai import commercial_gate, escalation, pricing, preflight
from src.ai.budget_guard import ANALYSIS_DEEP, BudgetGuard
from src.ai.budget_settings import BudgetSettings

READY = "ready_for_deep"
DEFERRED_INPUT_TOO_LARGE = "deferred_input_too_large"
BUDGET_DEFERRED = "budget_deferred"
PRICING_UNKNOWN = "pricing_unknown"

_GATE_STATUS = {
    commercial_gate.SKIP_LOW_VALUE: "skipped_low_value",
    commercial_gate.SKIP_TOO_SMALL: "skipped_too_small",
    commercial_gate.MANUAL_REVIEW: "manual_review_required",
    commercial_gate.INSUFFICIENT_INFORMATION: "insufficient_information",
}


def evaluate_deep_admission(
    gate_result: dict,
    analyzer,
    deep_context: dict,
    spend_usd,
    settings: BudgetSettings,
    pricing_overrides: dict | None = None,
) -> dict:
    """
    {deep_analysis_status, call_allowed, reason, reason_code, gate, preflight, budget}. reason_code —
    значение src.ai.escalation (deep_input_limit_exceeded / budget_deferred) или None. preflight/budget —
    None, если до них дело не дошло. analyzer используется только через build_request и model.
    """
    def outcome(status, allowed, reason, pre=None, budget=None, reason_code=None):
        return {
            "deep_analysis_status": status, "call_allowed": allowed, "reason": reason,
            "reason_code": reason_code,
            "gate": gate_result, "preflight": pre, "budget": budget,
        }

    decision = gate_result["gate_decision"]
    if decision != commercial_gate.DEEP_CANDIDATE:
        return outcome(_GATE_STATUS[decision], False, gate_result["gate_reason"])

    model = analyzer.model
    request = analyzer.build_request(deep_context)
    try:
        pre = preflight.preflight_request(request, model, ANALYSIS_DEEP, settings, pricing_overrides)
    except pricing.CostEstimationError as error:
        return outcome(PRICING_UNKNOWN, False, str(error))

    if pre["status"] == preflight.STATUS_DEEP_INPUT_TOO_LARGE:
        return outcome(
            DEFERRED_INPUT_TOO_LARGE, False,
            f"оценка input {pre['estimated_input_tokens']} токенов > MAX_DEEP_INPUT_TOKENS={pre['max_input_tokens']}",
            pre, reason_code=escalation.DEEP_INPUT_LIMIT_EXCEEDED,
        )

    budget = BudgetGuard(settings).check(spend_usd, pre["estimated_cost_usd"], ANALYSIS_DEEP, model)
    if not budget.allowed:
        return outcome(
            budget.deep_status or BUDGET_DEFERRED, False, budget.reason, pre, budget,
            reason_code=escalation.BUDGET_DEFERRED,
        )
    return outcome(READY, True, budget.reason, pre, budget)
