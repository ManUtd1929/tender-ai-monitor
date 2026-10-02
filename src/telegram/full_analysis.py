"""
Детерминированный «Полный анализ» тендера (HTML parse mode) из СОХРАНЁННОГО deep result.

Без БД, сети и AI. В отличие от короткой карточки здесь НИЧЕГО не обрезается и нет «Показано X из Y»:
все позиции, требования, барьеры и missing_information попадают в текст. Evidence-цитаты не выводятся
(canonical evidence остаётся в БД). Результат — список сообщений, каждое <= лимита Telegram; деление идёт
по секциям и пунктам (пункт не режется; только пункт длиннее страницы делится по границам слов).
"""

import html
import re

from src.telegram import message
from src.telegram.message import CONFIDENCE_RU, SEVERITY_ORDER, TELEGRAM_MESSAGE_LIMIT, _clean, _esc

TITLE = "📋 Полный анализ тендера"
PAGE_TITLE = "📋 Полный анализ — {page}/{total}"
# Запас под самый длинный заголовок страницы + разделитель (до 99 страниц).
_HEADER_RESERVE = len(PAGE_TITLE.format(page=99, total=99)) + 2
CONTINUED = " (продолжение)"
NL = chr(10)

BARRIER_TYPE_RU = {
    "manufacturer_authorization": "Авторизация производителя",
    "official_dealer_required": "Требуется официальный дилер",
    "origin_restriction": "Ограничение по происхождению",
    "certification": "Сертификация",
    "license": "Лицензия",
    "medical_registration": "Медицинская регистрация",
    "local_service_required": "Локальный сервис",
    "experience_requirement": "Требование к опыту",
    "financial_requirement": "Финансовое требование",
    "bid_security": "Обеспечение заявки",
    "contract_security": "Обеспечение исполнения договора",
    "short_delivery_deadline": "Короткий срок поставки",
}
SOURCE_RU = {"announcement": "объявление", "enrichment": "страница тендера", "document": "документы"}
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _dicts(values) -> list[dict]:
    return [v for v in values or [] if isinstance(v, dict)]


def _item_lines(item: dict) -> str | None:
    name = _clean(item.get("item_name"))
    if not name or message._is_placeholder(name):
        return None
    quantity = " ".join(p for p in (_clean(item.get("quantity")), _clean(item.get("unit"))) if p)
    head = f"{name} — {quantity}" if quantity else name
    if lot := _clean(item.get("lot_number")):
        head += f" (лот {lot})"
    lines = [f"• {_esc(head)}"]
    specs = [s for s in (_clean(v) for v in item.get("key_specifications") or []) if s]
    if specs:
        lines.append("  Характеристики: " + _esc("; ".join(specs)))
    brand = item.get("brand_or_equivalent")
    if isinstance(brand, dict):
        parts = []
        if specified := _clean(brand.get("specified_brand")):
            parts.append(f"марка/модель: {specified}")
        if brand.get("equivalent_allowed") is True:
            parts.append("эквивалент допускается")
        elif brand.get("equivalent_allowed") is False:
            parts.append("эквивалент не допускается")
        if parts:
            lines.append("  " + _esc("; ".join(parts)))
    return "\n".join(lines)


def _lot_line(lot: dict) -> str | None:
    number = _clean(lot.get("lot_number"))
    description = _clean(lot.get("description"))
    count = lot.get("item_count")
    if not number and not description:
        return None
    text = f"Лот {number}" if number else "Лот"
    if description:
        text += f": {description}"
    if isinstance(count, int) and not isinstance(count, bool):
        text += f" (позиций: {count})"
    return f"• {_esc(text)}"


def _logistics_lines(logistics: dict) -> list[str]:
    route = " → ".join(p for p in (_clean(logistics.get("origin")), _clean(logistics.get("destination"))) if p)
    labeled = (
        ("Услуга", logistics.get("service")), ("Груз", logistics.get("cargo")), ("Маршрут", route),
        ("Вид транспорта", logistics.get("transport_mode")), ("Вес", logistics.get("weight")),
        ("Объём", logistics.get("volume")), ("Периодичность", logistics.get("frequency")),
        ("Таможенные требования", logistics.get("customs_requirements")),
        ("Страхование", logistics.get("insurance_requirements")),
        ("Особые условия", logistics.get("special_conditions")),
    )
    return [f"• {label}: {_esc(text)}" for label, value in labeled if (text := _clean(value))]


def _barrier_line(barrier: dict) -> str | None:
    description = _clean(barrier.get("description"))
    label = BARRIER_TYPE_RU.get(_clean(barrier.get("type")) or "")
    if description and label and not description.lower().startswith(label.lower()):
        text = f"{label} — {description}"
    else:
        text = description or label or message.category_label(barrier.get("type"))
    if not text:
        return None
    return f"• {_esc(text if text.endswith(('.', '!', '?')) else text + '.')}"


def _conflict_line(conflict: dict) -> str | None:
    description = _clean(conflict.get("conflict_description"))
    if not description:
        return None
    sources = ", ".join(SOURCE_RU.get(s, str(s)) for s in conflict.get("sources") or [])
    text = description + (f" (источники: {sources})" if sources else "")
    if impact := _clean(conflict.get("impact")):
        text += f" Влияние: {impact}"
    return f"• {_esc(text)}"


def _plain_bullets(values) -> list[str]:
    return [f"• {_esc(text)}" for text in (_clean(v) for v in values or []) if text]


def _why_bullets(value) -> list[str]:
    return [f"• {_esc(p)}" for p in _SENTENCE_END.split(_clean(value) or "") if p]


def _field(title: str, value) -> tuple[str, list[str]] | None:
    text = _clean(value)
    return (title, [_esc(text)]) if text else None


def _sections(result: dict, resource_url, deadline, value_amd) -> list[tuple[str, list[str]]]:
    procurement = result.get("procurement") if isinstance(result.get("procurement"), dict) else {}
    logistics = result.get("logistics") if isinstance(result.get("logistics"), dict) else {}
    subject = _clean(procurement.get("subject")) or _clean(logistics.get("service")) or _clean(result.get("summary"))
    items = [line for line in (_item_lines(i) for i in _dicts(procurement.get("items"))) if line]
    lots = [line for line in (_lot_line(l) for l in _dicts(procurement.get("lots"))) if line]
    requirements = [
        *(procurement.get("technical_requirements") or []),
        *(procurement.get("country_of_origin_requirements") or []),
        *(procurement.get("certifications") or []),
    ]
    barriers = sorted(_dicts(result.get("participation_barriers")), key=lambda b: SEVERITY_ORDER.get(b.get("severity"), 3))
    delivery_terms = [
        f"• {label}: {_esc(text)}" for label, value in (
            ("Количество", procurement.get("quantity_summary")), ("Лотов", procurement.get("total_lots")),
            ("Место поставки", procurement.get("delivery_location")),
            ("Срок поставки", procurement.get("delivery_deadline")), ("Гарантия", procurement.get("warranty")),
        ) if (text := _clean(value))
    ]
    url = _clean(resource_url)
    link = None
    if url and url.lower().startswith(("http://", "https://")):
        link = ("Официальная ссылка:", [f'<a href="{html.escape(url, quote=True)}">{message.LINK_TEXT}</a>'])

    candidates = [
        _field("Предмет:", subject),
        _field("Заказчик:", result.get("contracting_authority") or message.NOT_SPECIFIED),
        _field("Процедура:", result.get("procedure_code")),
        _field("Категория:", message.category_label(result.get("category"))),
        _field("Дедлайн:", deadline),
        _field("Оценочная стоимость:", message.format_value_amd(value_amd)),
        ("Краткое резюме:", [_esc(_clean(result.get("summary")) or "")]),
        ("Почему может быть интересно:", _why_bullets(result.get("why_interesting"))),
        ("Что закупают:", items + (_logistics_lines(logistics) if logistics else [])),
        ("Лоты:", lots),
        ("Условия поставки:", delivery_terms),
        ("Ключевые требования:", _plain_bullets(requirements)),
        ("⚠️ Барьеры участия:", [line for line in (_barrier_line(b) for b in barriers) if line]),
        ("❓ Нужно уточнить:", _plain_bullets(result.get("missing_information"))),
        ("Конфликты источников:", [l for l in (_conflict_line(c) for c in _dicts(result.get("source_conflicts"))) if l]),
        _field("Уверенность анализа:", CONFIDENCE_RU.get(result.get("confidence"), result.get("confidence"))),
        link,
    ]
    return [(title, lines) for title, lines in (c for c in candidates if c) if any(lines)]


def _wrap_words(line: str, limit: int) -> list[str]:
    parts, current = [], ""
    for word in line.split(" "):
        while len(word) > limit:  # слово длиннее страницы (практически не встречается)
            if current:
                parts.append(current)
                current = ""
            parts.append(word[:limit])
            word = word[limit:]
        candidate = f"{current} {word}" if current else word
        if len(candidate) > limit:
            parts.append(current)
            current = word
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def _split_long(unit: str, limit: int) -> list[str]:
    """Единственный пункт длиннее страницы: деление по границам строк, затем слов (слово не режется)."""
    pieces = []
    for line in unit.splitlines():
        pieces += [line] if len(line) <= limit else _wrap_words(line, limit)
    parts, current = [], ""
    for piece in pieces:
        candidate = NL.join((current, piece)) if current else piece
        if len(candidate) > limit:
            parts.append(current)
            current = piece
        else:
            current = candidate
    parts.append(current)
    return parts


def _pack(sections: list[tuple[str, list[str]]], capacity: int) -> list[str]:
    """
    Раскладывает секции по страницам. Заголовок секции всегда вместе хотя бы с одним пунктом; секция,
    продолжающаяся на следующей странице, получает заголовок «(продолжение)».
    """
    pages: list[str] = []
    current = ""
    for title, lines in sections:
        heading = f"<b>{title}</b>"
        heading_cont = f"<b>{title.rstrip(':')}{CONTINUED}:</b>"
        limit = capacity - len(heading_cont) - 1
        units = [piece for line in lines for piece in ([line] if len(line) <= limit else _split_long(line, limit))]
        heading_open, started = False, False  # заголовок уже на текущей странице / секция уже начата
        for unit in units:
            head = heading_cont if started else heading
            addition = NL + unit if heading_open else (NL * 2 if current else "") + head + NL + unit
            if current and len(current) + len(addition) > capacity:
                pages.append(current)
                current = ""
                addition = (heading_cont if started else heading) + NL + unit
            current += addition
            heading_open = started = True
    if current:
        pages.append(current)
    return pages


def build_full_analysis_messages(result: dict, resource_url: str | None, deadline: str | None = None,
                                 value_amd: int | None = None) -> list[str]:
    """
    result — сохранённый deep result (dict), deadline — готовая строка срока или None, value_amd — ТОЛЬКО
    официально подтверждённая стоимость (value_facts), иначе None. Возвращает >= 1 сообщений <= 4096 символов.
    """
    sections = _sections(result, resource_url, deadline, value_amd)
    pages = _pack(sections, TELEGRAM_MESSAGE_LIMIT - _HEADER_RESERVE)
    total = len(pages)
    if total <= 1:
        return [f"{TITLE}\n\n{pages[0]}" if pages else TITLE]
    return [f"{PAGE_TITLE.format(page=n, total=total)}\n\n{page}" for n, page in enumerate(pages, 1)]
