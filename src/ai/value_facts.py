"""
Детерминированное извлечение денежных/количественных фактов тендера для commercial gate.
Без LLM, без сети, без БД: только tender_context (src.ai.tender_context.build_tender_context).

Источники стоимости (value_source) — строго три:
    official_tender_value      enrichment["estimated_value_amd"], если это чисто числовая строка;
    sum_of_official_lot_values сумма колонки "Цена закупки" таблицы лотов из текста объявления;
    unknown                    всё остальное (нет данных, неоднозначность, конфликт).
Произвольные числа из текста НЕ суммируются. Любая неоднозначность -> estimated_value_amd=None.

Таблица лотов — шаблон, реально наблюдаемый в документах ARMEPS/eauction (docx объявления),
две строки заголовка и строки данных, колонки разделены TAB:

    Лотов<TAB>Наименование лота            (или "Лот<TAB>Наименование лота")
    Номера<TAB>Цена закупки<TAB>           (или "Номер лота<TAB>Цена закупки")
    1<TAB>980 000<TAB>Шина для ...

Сумма принимается только если: таблица в документе одна; номера лотов 1..N подряд без
повторов; каждая цена — целое (пробелы/NBSP как разделители тысяч допустимы), пустая цена или
другой формат = неоднозначность; в документе нет маркера иной валюты и есть маркер драма
(таблица сама валюту не указывает — это проверка по тексту документа, консервативная);
все документы с такой таблицей дают ОДИНАКОВЫЙ набор лотов (RU/HY-версии одного объявления
совпадают, расхождение = unknown).

Duplicate-aware: документы группируются по точному sha256 текста (как в
tender_context.build_deep_analysis_context), факты считаются по одному представителю группы, поэтому
30 одинаковых XLSX не умножают ни суммы, ни количество.

total_quantity — сумма колонки "Քանակը" по XLSX-таблицам технической спецификации (заголовок
"Միավորի գինը"/"Ընդամենը Գումարը"/"Քանակը" наблюдался в реальных XLSX). Единицы измерения не
приводятся к общему знаменателю: это справочный факт, но НЕ основание для skip (см. commercial_gate).
"""

import hashlib
import re

VALUE_SOURCE_OFFICIAL = "official_tender_value"
VALUE_SOURCE_LOT_SUM = "sum_of_official_lot_values"
VALUE_SOURCE_UNKNOWN = "unknown"

_LOT_HEADER_1 = re.compile(r"^Лот(?:ов)?\tНаименование лота\s*$")
_LOT_HEADER_2 = re.compile(r"^Номер(?:а| лота)\tЦена закупки\s*$")
_LOT_ROW = re.compile(r"^(\d+)\t([^\t]*)\t(.*)$")
_INTEGER = re.compile(r"\d+")
_GROUPED_INTEGER = re.compile(r"\d{1,3}(?:[^\S\n]\d{3})+")

_AMD_MARKER = re.compile(r"\bдрам\w*|դրամ|\bAMD\b", re.IGNORECASE)
_OTHER_CURRENCY_MARKER = re.compile(r"\bдоллар|\bевро\b|\bрубл|\bруб\.|\b(?:USD|EUR|RUB)\b", re.IGNORECASE)

_QTY_HEADER_MARKERS = ("Միավորի գինը", "Ընդամենը Գումարը", "Քանակը")


class _Ambiguous(Exception):
    """Таблица найдена, но однозначно её прочитать нельзя."""


def parse_amount(raw) -> int | None:
    """Целое количество AMD из "980000" / "980 000" (NBSP тоже). Иное (десятичные, буквы, пусто) -> None."""
    if raw is None:
        return None
    text = str(raw).strip()
    if _INTEGER.fullmatch(text) or _GROUPED_INTEGER.fullmatch(text):
        return int(re.sub(r"[   ]", "", text))
    return None


def parse_lot_price_table(text: str) -> dict | None:
    """
    None — в тексте нет таблицы лотов. {lot_number: price_amd} — таблица прочитана однозначно.
    _Ambiguous — таблица есть, но неоднозначна (дубли/пропуски номеров, пустая или нечисловая
    цена, несколько таблиц, нет строк).
    """
    lines = text.split("\n")
    starts = [
        index for index in range(len(lines) - 1)
        if _LOT_HEADER_1.match(lines[index]) and _LOT_HEADER_2.match(lines[index + 1])
    ]
    if not starts:
        return None
    if len(starts) > 1:
        raise _Ambiguous("несколько таблиц лотов в одном документе")

    lots = {}
    for line in lines[starts[0] + 2:]:
        match = _LOT_ROW.match(line)
        if match is None:
            break
        number = int(match.group(1))
        price = parse_amount(match.group(2))
        if number in lots:
            raise _Ambiguous(f"повторяющийся номер лота {number}")
        if price is None:
            raise _Ambiguous(f"цена лота {number} пуста или не является целым числом: {match.group(2)!r}")
        lots[number] = price
    if not lots:
        raise _Ambiguous("таблица лотов без строк данных")
    if sorted(lots) != list(range(1, len(lots) + 1)):
        raise _Ambiguous(f"номера лотов не образуют 1..N: {sorted(lots)}")
    return lots


def _currency_is_amd(text: str) -> bool:
    return bool(_AMD_MARKER.search(text)) and not _OTHER_CURRENCY_MARKER.search(text)


def _unique_documents(tender_context: dict) -> list:
    """Успешно извлечённые документы с текстом; по одному (min по download_id, member_name) на sha256."""
    groups = {}
    for document in tender_context.get("documents") or []:
        text = document.get("text") or ""
        if document.get("extraction_status") == "success" and text:
            groups.setdefault(hashlib.sha256(text.encode("utf-8")).hexdigest(), []).append(document)
    return [
        min(group, key=lambda d: (str(d["download_id"]), str(d["member_name"])))
        for _, group in sorted(groups.items())
    ]


def _source_ref(document: dict) -> dict:
    return {
        "download_id": document["download_id"],
        "member_name": document["member_name"],
        "file_type": document["file_type"],
    }


def _lot_facts(documents: list) -> dict:
    """{lots: {n: price}|None, refs, ambiguity: str|None} по канонической уникальной выборке документов."""
    tables = []
    for document in documents:
        try:
            table = parse_lot_price_table(document["text"])
        except _Ambiguous as error:
            return {"lots": None, "refs": [_source_ref(document)], "ambiguity": str(error)}
        if table is None:
            continue
        if not _currency_is_amd(document["text"]):
            return {
                "lots": None, "refs": [_source_ref(document)],
                "ambiguity": "валюта таблицы лотов не подтверждена как AMD (нет маркера драма или есть иная валюта)",
            }
        tables.append((document, table))
    if not tables:
        return {"lots": None, "refs": [], "ambiguity": None}
    first = tables[0][1]
    refs = [_source_ref(document) for document, _ in tables]
    if any(table != first for _, table in tables):
        return {"lots": None, "refs": refs, "ambiguity": "документы содержат разные таблицы лотов"}
    return {"lots": first, "refs": refs, "ambiguity": None}


def extract_total_quantity(documents: list) -> dict:
    """{total_quantity: int|None, refs}: сумма колонки "Քանակը" по уникальным XLSX-таблицам спецификации."""
    total = 0
    refs = []
    for document in documents:
        if document.get("file_type") != "xlsx":
            continue
        column = None
        subtotal = 0
        found = False
        for line in document["text"].split("\n"):
            cells = line.split("\t")
            if all(marker in line for marker in _QTY_HEADER_MARKERS):
                column = next((i for i, cell in enumerate(cells) if cell.strip() == "Քանակը"), None)
                continue
            if column is None or column >= len(cells) or not cells[0].strip().isdigit():
                continue
            quantity = parse_amount(cells[column])
            if quantity is None:
                return {"total_quantity": None, "refs": []}  # нечисловое количество — не гадаем
            subtotal += quantity
            found = True
        if found:
            total += subtotal
            refs.append(_source_ref(document))
    return {"total_quantity": total if refs else None, "refs": refs}


def extract_value_facts(tender_context: dict) -> dict:
    """
    {estimated_value_amd: int|None, value_source, source_refs, total_lots: int|None,
    total_quantity: int|None, quantity_refs, notes: [str]}. Чистая функция; tender_context не меняется.
    """
    enrichment = tender_context.get("enrichment") or {}
    documents = _unique_documents(tender_context)
    notes = []

    official = parse_amount(enrichment.get("estimated_value_amd"))
    if enrichment.get("estimated_value_amd") not in (None, "") and official is None:
        notes.append(f"enrichment.estimated_value_amd не разобран как целое AMD: {enrichment['estimated_value_amd']!r}")

    lot_facts = _lot_facts(documents)
    lots = lot_facts["lots"]
    lot_sum = sum(lots.values()) if lots else None
    if lot_facts["ambiguity"]:
        notes.append(f"таблица лотов неоднозначна: {lot_facts['ambiguity']}")

    total_lots = enrichment.get("number_of_lots")
    if total_lots is None and lots:
        total_lots = len(lots)

    value, source, refs = None, VALUE_SOURCE_UNKNOWN, []
    if official is not None and lot_sum is not None and official != lot_sum:
        notes.append(f"конфликт: официальная стоимость {official} != сумма лотов {lot_sum}")
    elif official is not None:
        value, source = official, VALUE_SOURCE_OFFICIAL
        refs = [{"field": "enrichment.estimated_value_amd"}]
    elif lot_sum is not None:
        value, source, refs = lot_sum, VALUE_SOURCE_LOT_SUM, lot_facts["refs"]

    quantity = extract_total_quantity(documents)
    return {
        "estimated_value_amd": value,
        "value_source": source,
        "source_refs": refs,
        "total_lots": total_lots,
        "total_quantity": quantity["total_quantity"],
        "quantity_refs": quantity["refs"],
        "notes": notes,
    }
