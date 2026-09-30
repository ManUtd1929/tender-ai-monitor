"""
Детерминированное представление причин будущей эскалации Deep (primary -> fallback модель).

ВТОРОГО API-вызова (Terra) на этом этапе НЕТ: модуль только фиксирует словарь причин, чтобы
будущая логика и логи говорили на одном языке. Terra — настроенная fallback-модель
(budget_settings.deep_fallback_model), а не автоматически вызываемая.

ESCALATION_REASONS — причины, по которым позже МОЖНО будет повторить Deep на fallback-модели.
DEFERRAL_REASONS — Deep откладывается вообще (более дорогая модель проблему не решит).

Сознательно НЕ причины эскалации сами по себе:
    manual_review_required=true — сложный тендер (medical benchmark) может быть верно разобран
                                  Luna и всё равно требовать ручной проверки;
    confidence=medium           — то же самое.
Поэтому таких значений в словаре нет, а is_escalation_reason() для них возвращает False.
"""

VALIDATION_FAILURE = "validation_failure"
API_FAILURE = "api_failure"
DOCUMENT_COVERAGE_INCOMPLETE = "document_coverage_incomplete"
CRITICAL_SOURCE_CONFLICT = "critical_source_conflict"
HIGH_VALUE_MANUAL_REVIEW = "high_value_manual_review"
DEEP_INPUT_LIMIT_EXCEEDED = "deep_input_limit_exceeded"
BUDGET_DEFERRED = "budget_deferred"

ESCALATION_REASONS = frozenset({
    VALIDATION_FAILURE, API_FAILURE, DOCUMENT_COVERAGE_INCOMPLETE,
    CRITICAL_SOURCE_CONFLICT, HIGH_VALUE_MANUAL_REVIEW,
})
DEFERRAL_REASONS = frozenset({DEEP_INPUT_LIMIT_EXCEEDED, BUDGET_DEFERRED})
ALL_REASONS = ESCALATION_REASONS | DEFERRAL_REASONS


# Класс ошибки Deep-вызова (openai_triage.KIND_*) -> причина будущей эскалации. Строки совпадают
# со значениями KIND_*; config/authentication сюда не входят: другая модель с тем же ключом их не решит.
_ESCALATION_BY_ERROR_KIND = {
    "validation": VALIDATION_FAILURE,
    "invalid_structured_output": VALIDATION_FAILURE,
    "refusal": VALIDATION_FAILURE,
    "incomplete": VALIDATION_FAILURE,
    "timeout": API_FAILURE,
    "rate_limit": API_FAILURE,
    "api_error": API_FAILURE,
}


def reason_for_error_kind(error_kind) -> str | None:
    """Причина эскалации для ошибки Deep-вызова или None. Не зависит от содержимого результата."""
    return _ESCALATION_BY_ERROR_KIND.get(error_kind)


def is_escalation_reason(reason) -> bool:
    """True только для известных причин эскалации; произвольные строки (в т.ч. manual_review_required) — False."""
    return reason in ESCALATION_REASONS
