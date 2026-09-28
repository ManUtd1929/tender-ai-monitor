"""
Provider-independent схема и validation для двухэтапного AI relevance-анализа тендеров.

Никакого AI/HTTP здесь нет: модуль только описывает допустимую форму результата,
который вернёт injectable analyzer (см. src/ai/relevance_pipeline.py), и строго
проверяет её. Production/AI-интеграция подключается позже.

STAGE 1 = TRIAGE: relevant / maybe / not_relevant, быстрая классификация
(validate_triage_result). STAGE 2 = DEEP ANALYSIS: структурированный разбор,
отдельные поля для procurement и logistics (validate_deep_analysis_result).

Все validate_* функции принимают dict "как есть от AI" (или fake analyzer в тестах)
и возвращают новый нормализованный dict; входной dict не изменяется. Неизвестные enum
значения, неизвестные ключи верхнего уровня и нарушение бизнес-правил (например
not_relevant + requires_deep_analysis=True) -> ValueError. Это единственная линия
защиты от "свободного текста" в критичных полях (правило проекта №8).

Business context (BUSINESS GOAL из задачи): CIO интересуют LOGISTICS (международные/
внутренние перевозки, freight forwarding, доставка, customs-related logistics,
warehousing, multimodal transport) и PROCUREMENT физических товаров, которые
потенциально можно закупить у зарубежного поставщика и доставить заказчику
(компьютеры, мебель, оборудование, стройматериалы, медтехника, бытовая техника,
инструменты, автозапчасти, канцелярия, автомобили и спецтехника). Чистые услуги без
поставки товара или логистической составляющей (строительные работы, проектирование,
надзор, аудит, обучение, консалтинг, разработка ПО) сами по себе не интересны, но
финальное решение — за AI (STAGE 1/2), а не за hardcoded keyword-фильтром.
"""

RELEVANCE_STATUSES = ("relevant", "maybe", "not_relevant")

OPPORTUNITY_TYPES = (
    "logistics",
    "procurement",
    "logistics_and_procurement",
    "other_service",
    "unrelated",
    "unclear",
)

# "unclear" — доступного context недостаточно, чтобы определить тип возможности
# (procurement / service / works / ...). Допустим только вместе с relevance_status="maybe":
# relevant и not_relevant уже подразумевают, что тип определён. maybe при этом может
# использовать и любой другой opportunity_type, если общий тип понятен, а relevance — нет.
UNCLEAR_OPPORTUNITY_TYPE = "unclear"

CONFIDENCE_LEVELS = ("high", "medium", "low")

EVIDENCE_SOURCE_TYPES = ("announcement", "enrichment", "document")

PARTICIPATION_BARRIER_TYPES = (
    "manufacturer_authorization",
    "official_dealer_required",
    "origin_restriction",
    "certification",
    "license",
    "medical_registration",
    "local_service_required",
    "experience_requirement",
    "financial_requirement",
    "bid_security",
    "contract_security",
    "short_delivery_deadline",
    "other",
)

# Evidence text — короткая цитата/указатель на источник, не пересказ документа целиком.
EVIDENCE_TEXT_MAX_CHARS = 500

EVIDENCE_FIELDS = ("source_type", "field", "download_id", "member_name", "text")

TRIAGE_FIELDS = (
    "relevance_status",
    "opportunity_type",
    "category",
    "confidence",
    "reason",
    "requires_deep_analysis",
    "evidence",
)

DEEP_ANALYSIS_COMMON_FIELDS = (
    "summary",
    "opportunity_type",
    "category",
    "why_interesting",
    "participation_barriers",
    "missing_information",
    "manual_review_required",
    "evidence",
    "procurement",
    "logistics",
)

PROCUREMENT_SCALAR_FIELDS = (
    "subject",
    "quantity_summary",
    "brand_or_equivalent",
    "delivery_location",
    "delivery_deadline",
    "warranty",
    "estimated_value_amd",
)
PROCUREMENT_LIST_FIELDS = (
    "items",
    "lots",
    "technical_requirements",
    "country_of_origin_requirements",
    "certifications",
)
PROCUREMENT_FIELDS = PROCUREMENT_SCALAR_FIELDS + PROCUREMENT_LIST_FIELDS

LOGISTICS_SCALAR_FIELDS = (
    "service",
    "cargo",
    "origin",
    "destination",
    "transport_mode",
    "weight",
    "volume",
    "frequency",
    "customs_requirements",
    "insurance_requirements",
    "special_conditions",
)

# Основной принцип procurement-релевантности (пойдёт в будущий AI-prompt как есть).
PROCUREMENT_POLICY = """
Физический товар потенциально интересен, если CIO может теоретически построить
supply chain: найти зарубежного поставщика -> закупить товар -> импортировать ->
доставить заказчику. AI не должен утверждать, что CIO ТОЧНО может поставить товар —
только что закупка/импорт потенциально подходит для такой схемы. Наличие participation
barrier (см. PARTICIPATION_BARRIER_TYPES) само по себе не делает тендер not_relevant:
обычно "физический товар + barrier" -> maybe, до ручной проверки. Чистые услуги без
поставки товара и без логистической составляющей (строительные работы, проектирование,
технический надзор, аудит, обучение, консалтинг, разработка ПО) обычно не интересны,
но это не hardcoded keyword-отказ — решение принимает AI по контексту конкретного тендера.
"""

# Требования к AI-ответу, которые должны попасть в будущий prompt дословно (правило проекта №1, 8, 9).
NO_HALLUCINATION_RULES = """
- Использовать только предоставленный context (tender_context / triage_context / deep_analysis_context).
- Неизвестное значение -> null (для полей) или запись в missing_information, а не догадка.
- Не придумывать цену, количество, поставщика, сертификаты, возможность поставки CIO.
- evidence должен ссылаться на реальный context (источник, поле, download_id, member_name),
  а не на выдуманную цитату.
- coverage (document_coverage) вычисляется Python-кодом, а не AI: AI его только читает.
"""


def _is_blank(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _require_dict(value, what: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{what} должен быть dict: {value!r}")
    return value


def _require_exact_keys(value: dict, allowed_fields: tuple, what: str) -> None:
    extra = set(value.keys()) - set(allowed_fields)
    missing = set(allowed_fields) - set(value.keys())
    if extra:
        raise ValueError(f"{what}: неизвестные поля {sorted(extra)}")
    if missing:
        raise ValueError(f"{what}: отсутствуют обязательные поля {sorted(missing)}")


def _require_enum(value, allowed: tuple, field_name: str, what: str):
    if value not in allowed:
        raise ValueError(f"{what}.{field_name}: недопустимое значение {value!r} (ожидается одно из {allowed})")
    return value


def _require_bool(value, field_name: str, what: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{what}.{field_name} должен быть bool: {value!r}")
    return value


def _require_nonblank_str(value, field_name: str, what: str) -> str:
    if not isinstance(value, str) or _is_blank(value):
        raise ValueError(f"{what}.{field_name} должен быть непустой строкой: {value!r}")
    return value


def _optional_str(value, field_name: str, what: str):
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{what}.{field_name} должен быть str или None: {value!r}")
    return value


def _optional_list(value, field_name: str, what: str) -> list:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{what}.{field_name} должен быть list или None: {value!r}")
    return list(value)


def _str_list(value, field_name: str, what: str) -> list:
    items = _optional_list(value, field_name, what)
    for item in items:
        if not isinstance(item, str) or _is_blank(item):
            raise ValueError(f"{what}.{field_name}: элементы должны быть непустыми строками, получено {item!r}")
    return list(items)


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------

def validate_evidence_item(item: dict) -> dict:
    """Проверяет один evidence item; возвращает новый нормализованный dict."""
    item = _require_dict(item, "evidence item")
    _require_exact_keys(item, EVIDENCE_FIELDS, "evidence item")

    source_type = _require_enum(item["source_type"], EVIDENCE_SOURCE_TYPES, "source_type", "evidence item")
    text = _require_nonblank_str(item["text"], "text", "evidence item")
    if len(text) > EVIDENCE_TEXT_MAX_CHARS:
        raise ValueError(
            f"evidence item.text слишком длинный ({len(text)} > {EVIDENCE_TEXT_MAX_CHARS}); "
            "evidence должен быть короткой цитатой/указателем, а не пересказом документа"
        )

    field = _optional_str(item["field"], "field", "evidence item")
    member_name = _optional_str(item["member_name"], "member_name", "evidence item")

    download_id = item["download_id"]
    if download_id is not None and (isinstance(download_id, bool) or not isinstance(download_id, int)):
        raise ValueError(f"evidence item.download_id должен быть int или None: {download_id!r}")

    return {
        "source_type": source_type,
        "field": field,
        "download_id": download_id,
        "member_name": member_name,
        "text": text,
    }


def validate_evidence_list(evidence, what: str) -> list:
    if not isinstance(evidence, list):
        raise ValueError(f"{what}.evidence должен быть list: {evidence!r}")
    return [validate_evidence_item(item) for item in evidence]


def validate_participation_barriers(barriers) -> list:
    if not isinstance(barriers, list):
        raise ValueError(f"participation_barriers должен быть list: {barriers!r}")
    validated = []
    for barrier in barriers:
        _require_enum(barrier, PARTICIPATION_BARRIER_TYPES, "participation_barriers[]", "deep analysis")
        validated.append(barrier)
    return validated


# --------------------------------------------------------------------------
# STAGE 1: triage
# --------------------------------------------------------------------------

def validate_triage_result(result: dict) -> dict:
    """
    Проверяет и нормализует triage-результат (см. модульный docstring). ValueError —
    неизвестный enum, отсутствующее/лишнее поле, отсутствующий evidence item или
    нарушение правила relevance_status <-> requires_deep_analysis:
        not_relevant          -> requires_deep_analysis должен быть False;
        relevant / maybe      -> requires_deep_analysis должен быть True;
        opportunity_type="unclear" -> relevance_status должен быть "maybe".
    Входной dict не изменяется.
    """
    result = _require_dict(result, "triage result")
    _require_exact_keys(result, TRIAGE_FIELDS, "triage result")

    relevance_status = _require_enum(
        result["relevance_status"], RELEVANCE_STATUSES, "relevance_status", "triage result"
    )
    opportunity_type = _require_enum(
        result["opportunity_type"], OPPORTUNITY_TYPES, "opportunity_type", "triage result"
    )
    confidence = _require_enum(result["confidence"], CONFIDENCE_LEVELS, "confidence", "triage result")
    reason = _require_nonblank_str(result["reason"], "reason", "triage result")
    requires_deep_analysis = _require_bool(
        result["requires_deep_analysis"], "requires_deep_analysis", "triage result"
    )
    category = _optional_str(result["category"], "category", "triage result")
    evidence = validate_evidence_list(result["evidence"], "triage result")

    if opportunity_type == UNCLEAR_OPPORTUNITY_TYPE and relevance_status != "maybe":
        raise ValueError(
            f"triage result: opportunity_type='unclear' допустим только с relevance_status='maybe', "
            f"получено {relevance_status!r}"
        )
    if relevance_status == "not_relevant" and requires_deep_analysis is not False:
        raise ValueError("triage result: not_relevant требует requires_deep_analysis=False")
    if relevance_status in ("relevant", "maybe") and requires_deep_analysis is not True:
        raise ValueError(
            f"triage result: relevance_status={relevance_status!r} требует requires_deep_analysis=True"
        )

    return {
        "relevance_status": relevance_status,
        "opportunity_type": opportunity_type,
        "category": category,
        "confidence": confidence,
        "reason": reason,
        "requires_deep_analysis": requires_deep_analysis,
        "evidence": evidence,
    }


# --------------------------------------------------------------------------
# STAGE 2: deep analysis
# --------------------------------------------------------------------------

def validate_procurement_block(block: dict) -> dict:
    block = _require_dict(block, "procurement block")
    _require_exact_keys(block, PROCUREMENT_FIELDS, "procurement block")

    result = {
        name: _optional_str(block[name], name, "procurement block")
        for name in PROCUREMENT_SCALAR_FIELDS
    }
    for name in PROCUREMENT_LIST_FIELDS:
        result[name] = _optional_list(block[name], name, "procurement block")
    return result


def validate_logistics_block(block: dict) -> dict:
    block = _require_dict(block, "logistics block")
    _require_exact_keys(block, LOGISTICS_SCALAR_FIELDS, "logistics block")
    return {
        name: _optional_str(block[name], name, "logistics block")
        for name in LOGISTICS_SCALAR_FIELDS
    }


def _validate_optional_block(value, validator, field_name: str, what: str):
    if value is None:
        return None
    return validator(_require_dict(value, f"{what}.{field_name}"))


def validate_deep_analysis_result(result: dict) -> dict:
    """
    Проверяет и нормализует deep-analysis результат. ValueError — неизвестный enum,
    отсутствующее/лишнее поле, невалидный evidence/participation_barriers item, или
    несоответствие opportunity_type и procurement/logistics блоков:
        procurement                -> procurement обязателен, logistics должен быть None;
        logistics                  -> logistics обязателен, procurement должен быть None;
        logistics_and_procurement  -> оба блока обязательны;
        other_service / unrelated / unclear -> оба блока должны быть None
                                   (для unclear тип не определён, блоки заполнять нечем).
    Это и есть защита "procurement output не должен выдумывать logistics поля и наоборот".
    Входной dict не изменяется.
    """
    result = _require_dict(result, "deep analysis result")
    _require_exact_keys(result, DEEP_ANALYSIS_COMMON_FIELDS, "deep analysis result")

    summary = _require_nonblank_str(result["summary"], "summary", "deep analysis result")
    opportunity_type = _require_enum(
        result["opportunity_type"], OPPORTUNITY_TYPES, "opportunity_type", "deep analysis result"
    )
    category = _optional_str(result["category"], "category", "deep analysis result")
    why_interesting = _require_nonblank_str(
        result["why_interesting"], "why_interesting", "deep analysis result"
    )
    participation_barriers = validate_participation_barriers(result["participation_barriers"])
    missing_information = _str_list(
        result["missing_information"], "missing_information", "deep analysis result"
    )
    manual_review_required = _require_bool(
        result["manual_review_required"], "manual_review_required", "deep analysis result"
    )
    evidence = validate_evidence_list(result["evidence"], "deep analysis result")

    procurement = _validate_optional_block(
        result["procurement"], validate_procurement_block, "procurement", "deep analysis result"
    )
    logistics = _validate_optional_block(
        result["logistics"], validate_logistics_block, "logistics", "deep analysis result"
    )

    expects_procurement = opportunity_type in ("procurement", "logistics_and_procurement")
    expects_logistics = opportunity_type in ("logistics", "logistics_and_procurement")

    if expects_procurement and procurement is None:
        raise ValueError(f"deep analysis result: opportunity_type={opportunity_type!r} требует procurement")
    if not expects_procurement and procurement is not None:
        raise ValueError(f"deep analysis result: opportunity_type={opportunity_type!r} не должен иметь procurement")
    if expects_logistics and logistics is None:
        raise ValueError(f"deep analysis result: opportunity_type={opportunity_type!r} требует logistics")
    if not expects_logistics and logistics is not None:
        raise ValueError(f"deep analysis result: opportunity_type={opportunity_type!r} не должен иметь logistics")

    return {
        "summary": summary,
        "opportunity_type": opportunity_type,
        "category": category,
        "why_interesting": why_interesting,
        "participation_barriers": participation_barriers,
        "missing_information": missing_information,
        "manual_review_required": manual_review_required,
        "evidence": evidence,
        "procurement": procurement,
        "logistics": logistics,
    }
