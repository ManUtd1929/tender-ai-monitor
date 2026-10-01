"""
Детерминированный evidence catalog для DEEP analysis (procurement-deep-v4).

Раньше модель сама воспроизводила evidence.text (дословную цитату), и валидация падала на
любом неточном символе. Теперь текст источника принадлежит Python: из точных данных, которые
получает модель (announcement/enrichment поля + текст документов из chunks), строится каталог
"evidence units" с детерминированными ID.

procurement-deep-v7: модель НЕ воспроизводит строковые ID (live-сбои v5/v6: модель достраивала ID
16-символьным хешем). Модель-видимый протокол — request-local целочисленные ссылки: unit каталога
с позицией n (1-based, порядок каталога детерминирован) показывается как "[n] текст", а модель
возвращает evidence_refs: [n, ...], где 1 <= n <= len(catalog) (границы заданы в strict schema,
см. deep_prompt.build_deep_output_schema). Ссылка действует только внутри одного запроса и нигде не
сохраняется как постоянный ID. После строгой валидации (целое, 1..N, без повторов, без clamp/
fuzzy) Python материализует evidence с каноническим evidence_id и text ИСКЛЮЧИТЕЛЬНО из того же
объекта каталога, никогда из ответа модели. Канонические evidence_id (ниже) остаются внутренними.

Unit каталога:
    evidence_id, source_type, field, download_id, member_name, content_group_id, key,
    exact_text (+ start/end — смещения в тексте документа: exact_text == text[start:end]).

Формат ID (без random/UUID; зависит только от неизменного источника):
    announcement / enrichment: ev_ann_<field>_<n>, ev_enr_<field>_<n>
        n — порядковый номер unit'а внутри поля (строка/абзац, лист словаря, элемент списка);
    document: ev_doc_<download_id>_<sha256(text)[:6]>_<n>
        download_id и sha256 — canonical_source и точный текст content group (то же
        exact-content dedup, что и tender_context.build_deep_analysis_context),
        n — номер unit'а (строка / ограниченный кусок длинной строки) в тексте документа.

Размер unit'а: естественная единица — строка текста (абзац DOCX, строка таблицы XLSX,
значение поля). Строка длиннее MAX_UNIT_CHARS делится на смежные куски <= MAX_UNIT_CHARS
(предпочтительно по пробелу); куски вместе дают строку без потерь и изменений. Строки и
куски из одних whitespace пропускаются. Текст никогда не нормализуется, не переводится и не
пересказывается.

Дедупликация: chunks содержат текст только canonical_source каждой content group, поэтому
одинаковый контент 30 файлов даёт один набор units; кто ещё содержит этот контент — см.
deep_context["content_groups"][...]["represented_sources"] по content_group_id unit'а.
"""

import hashlib
import json

from src.ai import evidence_grounding, relevance_schema

# Материализованный evidence проходит relevance_schema.validate_evidence_item (<= 500 символов).
MAX_UNIT_CHARS = relevance_schema.EVIDENCE_TEXT_MAX_CHARS

EVIDENCE_REFS_KEY = "evidence_refs"
UNIT_FIELDS = (
    "evidence_id", "source_type", "field", "download_id", "member_name", "content_group_id", "key",
    "exact_text",
)
MATERIALIZED_FIELDS = ("evidence_id", "source_type", "field", "download_id", "member_name", "text")

_ID_PREFIX = {"announcement": "ev_ann", "enrichment": "ev_enr"}


# --------------------------------------------------------------------------
# splitting (текст не меняется: каждый кусок — точный срез источника)
# --------------------------------------------------------------------------

def _split_long_line(line: str, start: int, limit: int) -> list:
    """[(start, end)] смежных кусков line (line занимает text[start:start+len(line)])."""
    spans = []
    pos = 0
    while pos < len(line):
        end = min(pos + limit, len(line))
        if end < len(line):
            space = max(line.rfind(" ", pos, end), line.rfind("\t", pos, end))
            if space > pos:
                end = space + 1
        spans.append((start + pos, start + end))
        pos = end
    return spans


def _unit_spans(text: str, limit: int = MAX_UNIT_CHARS) -> list:
    """[(start, end)] units текста: по строкам ("\\n"), длинные строки — ограниченными кусками."""
    spans = []
    offset = 0
    for line in text.split("\n"):
        for start, end in _split_long_line(line, offset, limit):
            if text[start:end].strip():
                spans.append((start, end))
        offset += len(line) + 1
    return spans


# --------------------------------------------------------------------------
# catalog
# --------------------------------------------------------------------------

def _field_pieces(value) -> list:
    """[(key | None, text)] исходных строк поля announcement/enrichment (None/пусто -> [])."""
    if value is None or value == "" or value == {} or value == []:
        return []
    if isinstance(value, dict):
        pieces = []
        for key in sorted(value):
            pieces.extend((key, text) for _, text in _field_pieces(value[key]))
        return pieces
    if isinstance(value, (list, tuple)):
        pieces = []
        for element in value:
            if isinstance(element, str):
                pieces.extend(_field_pieces(element))
            elif element is not None:
                pieces.append((None, json.dumps(
                    element, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
                )))
        return pieces
    return [(None, str(value))]


def _context_units(source_type: str, fields: tuple, context: dict) -> list:
    units = []
    for field in fields:
        number = 0
        for key, text in _field_pieces(context.get(field)):
            for start, end in _unit_spans(text):
                units.append({
                    "evidence_id": f"{_ID_PREFIX[source_type]}_{field}_{number}",
                    "source_type": source_type, "field": field, "download_id": None,
                    "member_name": None, "content_group_id": None, "key": key,
                    "exact_text": text[start:end],
                })
                number += 1
    return units


def _chunk_index(chunk: dict) -> int:
    return int(chunk["chunk_id"].rsplit(":", 1)[-1])


def _document_units(chunks: list) -> list:
    """Units canonical-документов: текст восстанавливается склейкой chunks (порядок сохраняется)."""
    order = []
    by_document = {}
    for chunk in chunks:
        key = (chunk["download_id"], chunk["member_name"])
        if key not in by_document:
            by_document[key] = []
            order.append(key)
        by_document[key].append(chunk)

    units = []
    for download_id, member_name in order:
        document_chunks = sorted(by_document[(download_id, member_name)], key=_chunk_index)
        text = "".join(chunk["text"] for chunk in document_chunks)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        group_id = document_chunks[0].get("content_group_id") or f"content-{digest[:16]}"
        for number, (start, end) in enumerate(_unit_spans(text)):
            units.append({
                "evidence_id": f"ev_doc_{download_id}_{digest[:6]}_{number:04d}",
                "source_type": "document", "field": evidence_grounding.DEEP_DOCUMENT_EVIDENCE_FIELD,
                "download_id": download_id, "member_name": member_name,
                "content_group_id": group_id, "key": None,
                "exact_text": text[start:end], "start": start, "end": end,
            })
    return units


def build_evidence_catalog(context_fields: dict, chunks: list) -> list:
    """
    Список evidence units в детерминированном порядке: announcement поля, enrichment поля,
    документы (в порядке chunks). context_fields — плоские announcement/enrichment поля
    (deep_context: title, description, cpv_codes, ...). ValueError — повторяющийся evidence_id.
    """
    catalog = (
        _context_units("announcement", evidence_grounding.ANNOUNCEMENT_EVIDENCE_FIELDS, context_fields)
        + _context_units("enrichment", evidence_grounding.ENRICHMENT_EVIDENCE_FIELDS, context_fields)
        + _document_units(chunks)
    )
    index_catalog(catalog)
    return catalog


def index_catalog(catalog: list) -> dict:
    """{evidence_id: unit}. ValueError, если ID повторяется (каталог должен быть однозначным)."""
    index = {}
    for unit in catalog:
        if unit["evidence_id"] in index:
            raise ValueError(f"evidence catalog: повторяющийся evidence_id {unit['evidence_id']!r}")
        index[unit["evidence_id"]] = unit
    return index


def evidence_ref_map(catalog: list) -> dict:
    """{canonical evidence_id: request-local ref (1-based позиция в каталоге)}."""
    return {unit["evidence_id"]: number for number, unit in enumerate(catalog, start=1)}


def render_catalog(catalog: list) -> str:
    """
    Текст каталога для модели. Заголовок [SOURCE ...] повторяется только при смене источника;
    unit — одна строка "[n] exact_text", n — request-local ссылка 1..N (в exact_text нет перевода
    строки по построению). Канонические evidence_id, content_group_id и хеши модели не показываются.
    """
    if not catalog:
        return "(empty: no evidence units are available)"
    lines = []
    current = None
    for number, unit in enumerate(catalog, start=1):
        source = (unit["source_type"], unit["field"], unit["download_id"], unit["member_name"])
        if source != current:
            current = source
            header = f"[SOURCE type={unit['source_type']} field={unit['field']}"
            if unit["source_type"] == "document":
                header += (
                    f" download_id={unit['download_id']} member_name={json.dumps(unit['member_name'], ensure_ascii=False)}"
                )
            lines.append(header + "]")
        label = f"{number} key={unit['key']}" if unit["key"] is not None else str(number)
        lines.append(f"[{label}] {unit['exact_text']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# materialization (model output with evidence_refs -> validated-shape result)
# --------------------------------------------------------------------------

def materialize_evidence(evidence_refs, catalog: list, what: str) -> list:
    """
    evidence_refs модели (целые 1..N) -> [{evidence_id, source_type, field, download_id, member_name,
    text}], text и канонический evidence_id ТОЛЬКО из каталога. ValueError — не list, не целое
    (bool и строки отклоняются), ссылка < 1 или > N, повтор ссылки. Никакого clamp/wrap/fuzzy.
    """
    if not isinstance(evidence_refs, list):
        raise ValueError(f"{what}.{EVIDENCE_REFS_KEY} должен быть list: {evidence_refs!r}")
    size = len(catalog)
    seen = set()
    materialized = []
    for ref in evidence_refs:
        if isinstance(ref, bool) or not isinstance(ref, int):
            raise ValueError(f"{what}.{EVIDENCE_REFS_KEY}: ссылка должна быть целым числом: {ref!r}")
        if ref in seen:
            raise ValueError(f"{what}.{EVIDENCE_REFS_KEY}: ссылка повторяется: {ref!r}")
        seen.add(ref)
        if not 1 <= ref <= size:
            raise ValueError(f"{what}.{EVIDENCE_REFS_KEY}: неизвестная ссылка {ref!r} (допустимо 1..{size})")
        unit = catalog[ref - 1]
        materialized.append({
            "evidence_id": unit["evidence_id"], "source_type": unit["source_type"], "field": unit["field"],
            "download_id": unit["download_id"], "member_name": unit["member_name"], "text": unit["exact_text"],
        })
    return materialized


def _swap_evidence(container, catalog: list, what: str) -> dict:
    """Копия container, где evidence_refs заменён materialized evidence. Ключа evidence быть не должно."""
    if not isinstance(container, dict):
        raise ValueError(f"{what} должен быть dict: {container!r}")
    if "evidence" in container:
        raise ValueError(f"{what}: модель не должна возвращать evidence с текстом, только {EVIDENCE_REFS_KEY}")
    if EVIDENCE_REFS_KEY not in container:
        raise ValueError(f"{what}: отсутствует обязательное поле {EVIDENCE_REFS_KEY!r}")
    result = {key: value for key, value in container.items() if key != EVIDENCE_REFS_KEY}
    result["evidence"] = materialize_evidence(container[EVIDENCE_REFS_KEY], catalog, what)
    return result


def _swap_list(items, catalog: list, what: str) -> list:
    if not isinstance(items, list):
        raise ValueError(f"{what} должен быть list: {items!r}")
    return [_swap_evidence(item, catalog, f"{what}[{position}]") for position, item in enumerate(items)]


def materialize_deep_model_output(raw: dict, catalog: list) -> dict:
    """
    Raw deep-ответ v7 (evidence_refs везде) -> результат старой формы (evidence = materialized
    units), который затем проходит relevance_schema.validate_deep_analysis_result и
    business-правила. raw не изменяется. ValueError — невалидная/повторная ссылка, evidence с
    текстом от модели или отсутствующий evidence_refs.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"deep model output должен быть dict: {raw!r}")
    index_catalog(catalog)  # каталог должен быть однозначным (повторяющийся canonical ID -> ValueError)

    result = _swap_evidence(raw, catalog, "deep analysis result")
    result["participation_barriers"] = _swap_list(
        raw.get("participation_barriers"), catalog, "participation_barriers",
    )
    procurement = raw.get("procurement")
    if procurement is not None:
        if not isinstance(procurement, dict):
            raise ValueError(f"procurement должен быть dict или None: {procurement!r}")
        procurement = dict(procurement)
        procurement["items"] = _swap_list(procurement.get("items"), catalog, "procurement.items")
        procurement["lots"] = _swap_list(procurement.get("lots"), catalog, "procurement.lots")
    result["procurement"] = procurement
    return result
