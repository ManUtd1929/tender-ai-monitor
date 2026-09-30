"""
Детерминированная проверка grounding для evidence triage-результата.

Evidence должен быть дословным фрагментом того, что модель реально получила в
TENDER_CONTEXT (triage_context). Здесь нет семантического сопоставления, перевода или
fuzzy-поиска: для каждого evidence item проверяется, что
    - источник существует (announcement/enrichment: поле из белого списка и оно
      заполнено в context; document: download_id + member_name соответствуют реальному
      элементу document_previews);
    - evidence.text после ТОЛЬКО нормализации пробелов/переводов строк (любая серия
      whitespace -> один пробел, strip; регистр и юникод не трогаются) является точной
      подстрокой значения источника.

Структурированные поля (dict/list, например detail_titles, dates, cpv_codes) проверяются
по конкретному значению поля: text должен быть подстрокой либо одного строкового
значения внутри него, либо JSON-представления поля в том виде, в каком его видит модель
(triage_prompt.serialize_triage_context). Для строк дополнительно допускается их
JSON-экранированная форма (\\n, \\"), потому что именно так строка выглядит во входном JSON.

ValueError — evidence нельзя обосновать; вызывающий код превращает его в validation error.
"""

import json
import re

ANNOUNCEMENT_EVIDENCE_FIELDS = ("title", "section", "resource_type", "published_at", "deadline_at")
ENRICHMENT_EVIDENCE_FIELDS = (
    "detail_titles", "description", "procurement_type", "procedure_type",
    "contracting_authority", "estimated_value_amd", "dates", "cpv_codes",
)
DOCUMENT_EVIDENCE_FIELD = "preview_text"

_WHITESPACE = re.compile(r"\s+")


def normalize_evidence_text(text: str) -> str:
    """Единственная допустимая нормализация: серии whitespace (в т.ч. \\n) -> один пробел, strip."""
    return _WHITESPACE.sub(" ", text).strip()


def _leaf_strings(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _leaf_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _leaf_strings(item)
    elif value is not None and not isinstance(value, bool):
        yield str(value)


def _string_variants(value: str) -> list:
    """Сама строка и её JSON-экранированная форма (как строка выглядит во входном JSON)."""
    return [value, json.dumps(value, ensure_ascii=False)[1:-1]]


def _source_variants(value) -> list:
    """Все допустимые тексты, подстрокой одного из которых может быть evidence.text."""
    if isinstance(value, str):
        return _string_variants(value)
    variants = []
    if isinstance(value, (dict, list, tuple)):
        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        variants.append(serialized)
    for leaf in _leaf_strings(value):
        variants.extend(_string_variants(leaf))
    return variants


def _require_quoted(item: dict, source_value, what: str) -> None:
    text = normalize_evidence_text(item["text"])
    if not text:
        raise ValueError("evidence: text пустой после нормализации whitespace")
    for variant in _source_variants(source_value):
        if text in normalize_evidence_text(variant):
            return
    raise ValueError(
        f"evidence: text не является дословным фрагментом {what} "
        f"(перевод, пересказ и добавленные пояснения запрещены): {item['text']!r}"
    )


def _check_context_field(item: dict, triage_context: dict, allowed_fields: tuple) -> None:
    source_type = item["source_type"]
    if item["download_id"] is not None or item["member_name"] is not None:
        raise ValueError(f"evidence: source_type={source_type!r} не может иметь download_id/member_name")
    field = item["field"]
    if field not in allowed_fields:
        raise ValueError(
            f"evidence: field={field!r} не существует в источнике {source_type!r} (ожидается одно из {allowed_fields})"
        )
    value = triage_context.get(field)
    if value is None or value == "" or value == {} or value == []:
        raise ValueError(f"evidence: поле {source_type}.{field} пусто в triage_context, цитировать нечего")
    _require_quoted(item, value, f"{source_type}.{field}")


def _check_document(item: dict, triage_context: dict) -> None:
    download_id = item["download_id"]
    member_name = item["member_name"]
    documents = triage_context.get("documents") or []

    if download_id not in {document["download_id"] for document in documents}:
        raise ValueError(f"evidence: download_id={download_id!r} отсутствует в triage_context.documents")
    if member_name is None or member_name not in {
        document["member_name"] for document in documents if document["download_id"] == download_id
    }:
        raise ValueError(f"evidence: member_name={member_name!r} отсутствует у download_id={download_id}")
    if item["field"] != DOCUMENT_EVIDENCE_FIELD:
        raise ValueError(
            f"evidence: у document field должен быть {DOCUMENT_EVIDENCE_FIELD!r}, получено {item['field']!r}"
        )

    previews = [
        preview for preview in triage_context.get("document_previews") or []
        if preview["download_id"] == download_id and preview["member_name"] == member_name
    ]
    if not previews:
        raise ValueError(
            f"evidence: у документа download_id={download_id} member_name={member_name!r} нет preview в "
            "triage_context.document_previews, цитировать нечего"
        )
    for preview in previews:
        try:
            _require_quoted(item, preview.get("preview_text") or "", f"preview документа {member_name!r}")
            return
        except ValueError as error:
            last_error = error
    raise last_error


def validate_evidence_grounding(evidence: list, triage_context: dict) -> None:
    """ValueError на первом evidence item, который нельзя обосновать по triage_context."""
    for item in evidence:
        source_type = item["source_type"]
        if source_type == "announcement":
            _check_context_field(item, triage_context, ANNOUNCEMENT_EVIDENCE_FIELDS)
        elif source_type == "enrichment":
            _check_context_field(item, triage_context, ENRICHMENT_EVIDENCE_FIELDS)
        else:
            _check_document(item, triage_context)


# --------------------------------------------------------------------------
# deep analysis (chunk-aware) grounding — не меняет ничего выше (triage grounding).
#
# deep_context (src.ai.deep_prompt.build_deep_context) содержит announcement/enrichment
# те же имена полей, что и triage_context (ANNOUNCEMENT_EVIDENCE_FIELDS/
# ENRICHMENT_EVIDENCE_FIELDS переиспользуются как есть), но вместо document_previews —
# chunks (src.ai.tender_context.build_deep_analysis_context): документ мог быть разбит на
# несколько chunk'ов. Для document evidence мы восстанавливаем полный текст документа
# склейкой его chunk'ов по возрастанию index (chunk_id = "{download_id}:{member_name}:{index}",
# "".join(chunks) == исходный extracted text — см. tender_context._chunk_text) и проверяем
# quote тем же _require_quoted/normalize_evidence_text, что и triage: grounding не слабее.
# --------------------------------------------------------------------------

DEEP_DOCUMENT_EVIDENCE_FIELD = "text"


def _chunk_index(chunk: dict) -> int:
    return int(chunk["chunk_id"].rsplit(":", 1)[-1])


def _reconstruct_document_text(deep_context: dict, download_id, member_name) -> str | None:
    chunks = [
        chunk for chunk in deep_context.get("chunks") or []
        if chunk["download_id"] == download_id and chunk["member_name"] == member_name
    ]
    if not chunks:
        return None
    return "".join(chunk["text"] for chunk in sorted(chunks, key=_chunk_index))


def _check_deep_document(item: dict, deep_context: dict) -> None:
    download_id = item["download_id"]
    member_name = item["member_name"]
    documents = deep_context.get("documents") or []

    if download_id not in {document["download_id"] for document in documents}:
        raise ValueError(f"evidence: download_id={download_id!r} отсутствует в deep_context.documents")
    if member_name is None or member_name not in {
        document["member_name"] for document in documents if document["download_id"] == download_id
    }:
        raise ValueError(f"evidence: member_name={member_name!r} отсутствует у download_id={download_id}")
    if item["field"] != DEEP_DOCUMENT_EVIDENCE_FIELD:
        raise ValueError(
            f"evidence: у document field должен быть {DEEP_DOCUMENT_EVIDENCE_FIELD!r}, получено {item['field']!r}"
        )

    text = _reconstruct_document_text(deep_context, download_id, member_name)
    if not text:
        # Дедупликация: контент отправлен один раз под canonical_source. Цитата представленного
        # (represented) источника принимается, только если он реально числится в content_group
        # (identical extracted text) — проверяется по тому же canonical тексту.
        for group in deep_context.get("content_groups") or []:
            if any(
                source["download_id"] == download_id and source["member_name"] == member_name
                for source in group["represented_sources"]
            ):
                canonical = group["canonical_source"]
                text = _reconstruct_document_text(
                    deep_context, canonical["download_id"], canonical["member_name"],
                )
                break
    if not text:
        raise ValueError(
            f"evidence: у документа download_id={download_id} member_name={member_name!r} нет chunks в "
            "deep_context.chunks, цитировать нечего"
        )
    _require_quoted(item, text, f"документ {member_name!r}")


def validate_deep_evidence_grounding(evidence: list, deep_context: dict) -> None:
    """
    ValueError на первом evidence item, который нельзя обосновать по deep_context
    (src.ai.deep_prompt.build_deep_context). announcement/enrichment проверяются как в
    triage (_check_context_field); document — по восстановленному из chunks тексту.
    """
    for item in evidence:
        source_type = item["source_type"]
        if source_type == "announcement":
            _check_context_field(item, deep_context, ANNOUNCEMENT_EVIDENCE_FIELDS)
        elif source_type == "enrichment":
            _check_context_field(item, deep_context, ENRICHMENT_EVIDENCE_FIELDS)
        else:
            _check_deep_document(item, deep_context)
