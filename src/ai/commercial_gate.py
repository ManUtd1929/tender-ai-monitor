"""
Дешёвый детерминированный commercial gate ПЕРЕД Deep Analysis. Это НЕ коммерческий анализ:
он отвечает только на вопрос «стоит ли тратить токены Terra на этот релевантный тендер
прямо сейчас». Без LLM, сети и БД; работает по triage_result и уже собранному tender_context
(факты — src.ai.value_facts).

gate_decision:
    deep_candidate           отправлять в Deep (дальше — preflight и Budget Guard)
    skip_low_value           надёжная официальная стоимость < MIN_DEEP_VALUE_AMD
    skip_too_small           ЗАРЕЗЕРВИРОВАНО, в этом этапе не выдаётся (см. ниже)
    manual_review            факты противоречат друг другу (например, стоимость != сумма лотов)
    insufficient_information нечего анализировать: нет ни одного извлечённого документа

Правила:
    - relevance_status НЕ меняется: gate работает поверх результата triage и не перезаписывает его.
    - Количество само по себе НИКОГДА не даёт skip (1 дорогой аппарат может быть интереснее
      500 канцтоваров): total_quantity/total_lots/category выводятся как факты, но в решение
      о skip не входят. Поэтому skip_too_small пока не производится — для него нужно
      value-обоснованное правило.
    - Value-based skip работает только если MIN_DEEP_VALUE_AMD задан. Значения по умолчанию нет:
      без порога денежных auto-skip нет (нет скрытого порога, теряющего реальные возможности).
    - Стоимость неизвестна -> НЕ skip: если есть документы, тендер остаётся deep_candidate
      (Deep как раз извлекает недостающее; расход всё равно ограничен preflight и Budget Guard).
    - Категория — только факт: никакой «медоборудование = дорого» здесь нет.

Применимо к relevance_status relevant/maybe (как и Deep в проде); для not_relevant — ValueError.
"""

from src.ai import value_facts

DEEP_CANDIDATE = "deep_candidate"
SKIP_LOW_VALUE = "skip_low_value"
SKIP_TOO_SMALL = "skip_too_small"
MANUAL_REVIEW = "manual_review"
INSUFFICIENT_INFORMATION = "insufficient_information"

GATE_DECISIONS = (DEEP_CANDIDATE, SKIP_LOW_VALUE, SKIP_TOO_SMALL, MANUAL_REVIEW, INSUFFICIENT_INFORMATION)

CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"

_ELIGIBLE_STATUSES = ("relevant", "maybe")


def _facts_summary(facts: dict) -> dict:
    return {
        "estimated_value_amd": facts["estimated_value_amd"],
        "value_source": facts["value_source"],
        "source_refs": facts["source_refs"],
        "total_lots": facts["total_lots"],
        "total_quantity": facts["total_quantity"],
        "notes": facts["notes"],
    }


def evaluate(triage_result: dict, tender_context: dict, min_deep_value_amd: int | None = None) -> dict:
    """
    {gate_decision, gate_reason, facts_used, estimated_value_amd, total_quantity, total_lots,
    category, confidence}. triage_result/tender_context не изменяются.
    """
    status = triage_result.get("relevance_status")
    if status not in _ELIGIBLE_STATUSES:
        raise ValueError(f"Commercial gate применим к relevant/maybe, получено relevance_status={status!r}")
    if min_deep_value_amd is not None and (
        isinstance(min_deep_value_amd, bool) or not isinstance(min_deep_value_amd, int) or min_deep_value_amd <= 0
    ):
        raise ValueError(f"min_deep_value_amd должен быть положительным целым или None: {min_deep_value_amd!r}")

    facts = value_facts.extract_value_facts(tender_context)
    value = facts["estimated_value_amd"]
    source = facts["value_source"]

    def result(decision: str, reason: str, confidence: str) -> dict:
        return {
            "gate_decision": decision,
            "gate_reason": reason,
            "facts_used": _facts_summary(facts),
            "estimated_value_amd": value,
            "total_quantity": facts["total_quantity"],
            "total_lots": facts["total_lots"],
            "category": triage_result.get("category"),
            "confidence": confidence,
        }

    if any(note.startswith("конфликт") for note in facts["notes"]):
        return result(MANUAL_REVIEW, "факты о стоимости противоречат друг другу: " + "; ".join(facts["notes"]), CONFIDENCE_LOW)

    if value is not None:
        confidence = CONFIDENCE_HIGH if source == value_facts.VALUE_SOURCE_OFFICIAL else CONFIDENCE_MEDIUM
        if min_deep_value_amd is None:
            return result(
                DEEP_CANDIDATE,
                f"стоимость {value} AMD ({source}) известна, но MIN_DEEP_VALUE_AMD не задан: value-based skip выключен",
                confidence,
            )
        if value < min_deep_value_amd:
            return result(
                SKIP_LOW_VALUE, f"стоимость {value} AMD ({source}) < MIN_DEEP_VALUE_AMD={min_deep_value_amd}", confidence,
            )
        return result(
            DEEP_CANDIDATE, f"стоимость {value} AMD ({source}) >= MIN_DEEP_VALUE_AMD={min_deep_value_amd}", confidence,
        )

    coverage = tender_context.get("document_coverage") or {}
    if not coverage.get("successful_extractions"):
        return result(
            INSUFFICIENT_INFORMATION, "стоимость неизвестна и нет ни одного извлечённого документа для анализа",
            CONFIDENCE_LOW,
        )
    return result(
        DEEP_CANDIDATE,
        "надёжная стоимость неизвестна: денежный порог применить нельзя, тендер не пропускается "
        "(количество/лоты/категория для skip не используются)",
        CONFIDENCE_LOW,
    )
