"""
Детерминированная Telegram-карточка тендера (HTML parse mode) из СОХРАНЁННОГО deep result.

Без БД, сети и AI: функция получает уже готовые данные. Пустые секции и значения None не выводятся,
evidence в карточку не попадает, длинные списки обрезаются ("... и ещё N"), весь текст экранируется
(& < >), чтобы название тендера/заказчика не ломало разметку.
"""

import html
import re

TELEGRAM_MESSAGE_LIMIT = 4096
LINK_TEXT = "Открыть тендер"
HEADER = "🟢 Новый подходящий тендер"
NOT_SPECIFIED = "Не указано"

# (max_items, max_chars_per_line): первый профиль, который укладывается в лимит Telegram, побеждает.
SIZE_PROFILES = ((5, 160), (3, 100), (1, 60))
# Лимиты списков компактной карточки (профиль размера может только уменьшить их).
MAX_ITEMS, MAX_REQUIREMENTS, MAX_BARRIERS, MAX_MISSING, MAX_WHY = 5, 3, 3, 4, 2

# Отображаемые названия категорий; сохранённый category в БД не меняется.
CATEGORY_RU = {
    "aviation_fuel": "Авиационное топливо",
    "blinds": "Жалюзи",
    "computer_equipment": "Компьютерное оборудование",
    "drinking_water": "Питьевая вода",
    "household_goods": "Хозяйственные товары",
    "infrastructure_goods": "Инфраструктурные товары",
    "laboratory_supplies": "Лабораторные материалы",
    "medical_equipment": "Медицинское оборудование",
    "modular_buildings": "Модульные здания",
    "pharmaceuticals_and_lab_supplies": "Фармацевтика и лабораторные материалы",
    "plants": "Растения / озеленение",
    "tires": "Шины",
    "archiving_services": "Услуги архивирования",
    "design_and_cost_estimation": "Проектирование и сметы",
    "expertise_services": "Экспертные услуги",
    "software_license": "Лицензии на ПО",
    "sports_event_services": "Организация спортивных мероприятий",
    "technical_supervision": "Технический надзор",
}
_LOT_PLACEHOLDER = re.compile(r"^(лот|lot|լոտ)\s*(№|#|n)?\s*\d+$", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

CONFIDENCE_RU = {"high": "высокая", "medium": "средняя", "low": "низкая"}
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}
_EMPTY_WORDS = {"", "none", "null", "n/a"}


def _clean(value) -> str | None:
    """Строка без лишних пробелов; None для пустого значения и текстовых "None"/"null"."""
    if value is None or isinstance(value, bool):
        return None
    text = " ".join(str(value).split())
    return None if text.lower() in _EMPTY_WORDS else text


def _clip(text: str, max_chars: int) -> str:
    """Сокращает по границе предложения, иначе слова; слово посередине не режется."""
    if len(text) <= max_chars:
        return text
    head = text[: max_chars - 3]
    if text[len(head)] != " ":  # граница обрезки внутри слова — откатываемся к пробелу
        head = head.rsplit(" ", 1)[0] if " " in head else head
    ends = [m.start() for m in re.finditer(r"[.!?](?=\s|$)", head)]
    if ends and ends[-1] + 1 >= max_chars * 0.4:
        return head[: ends[-1] + 1]
    return head.rstrip(" ,;:-–—") + "..."


def category_label(category) -> str | None:
    code = _clean(category)
    if not code:
        return None
    if code in CATEGORY_RU:
        return CATEGORY_RU[code]
    text = code.replace("_", " ")
    return text[0].upper() + text[1:]


def _is_placeholder(name: str) -> bool:
    return bool(_LOT_PLACEHOLDER.match(name.strip()))


def _esc(text: str) -> str:
    return html.escape(text, quote=False)


def _bullets(values, max_items: int, max_chars: int) -> list[str]:
    cleaned = [text for text in (_clean(v) for v in values) if text]
    lines = [f"• {_esc(_clip(text, max_chars))}" for text in cleaned[:max_items]]
    if len(cleaned) > max_items:
        lines.append(f"... и ещё {len(cleaned) - max_items}")
    return lines


def _section(title: str, lines: list[str]) -> str | None:
    return f"<b>{title}</b>\n" + "\n".join(lines) if lines else None


def _item_line(item: dict) -> str | None:
    name = _clean(item.get("item_name"))
    if not name or _is_placeholder(name):
        return None
    quantity = " ".join(p for p in (_clean(item.get("quantity")), _clean(item.get("unit"))) if p)
    return f"{name} — {quantity}" if quantity else name


def _lot_line(lot: dict) -> str | None:
    number, description = _clean(lot.get("lot_number")), _clean(lot.get("description"))
    if not description:  # «Лот N» без описания — не информация о предмете (число лотов — в «Количество / лоты»)
        return None
    return f"Лот {number}: {description}" if number else description


def _what_is_procured(procurement: dict, logistics: dict) -> list[str]:
    lines = [_item_line(i) for i in procurement.get("items") or [] if isinstance(i, dict)]
    if not any(lines):  # позиций нет — хотя бы лоты
        lines = [_lot_line(l) for l in procurement.get("lots") or [] if isinstance(l, dict)]
    if logistics:
        route = " → ".join(p for p in (_clean(logistics.get("origin")), _clean(logistics.get("destination"))) if p)
        lines += [
            f"Услуга: {s}" if (s := _clean(logistics.get("service"))) else None,
            f"Груз: {s}" if (s := _clean(logistics.get("cargo"))) else None,
            f"Маршрут: {route}" if route else None,
            f"Вид транспорта: {s}" if (s := _clean(logistics.get("transport_mode"))) else None,
        ]
    return [line for line in lines if line]


def _quantity_lines(procurement: dict) -> list[str]:
    lines = []
    if summary := _clean(procurement.get("quantity_summary")):
        lines.append(summary)
    total_lots = procurement.get("total_lots")
    mentions_lots = "лот" in " ".join(lines).lower()  # summary уже говорит про лоты — не дублируем
    if isinstance(total_lots, int) and not isinstance(total_lots, bool) and not mentions_lots:
        lines.append(f"Лотов: {total_lots}")
    return lines


def _why_lines(value) -> list[str]:
    text = _clean(value)
    return [p for p in _SENTENCE_END.split(text) if p][:MAX_WHY] if text else []


def _barrier_texts(barriers) -> list[str]:
    valid = [b for b in barriers or [] if isinstance(b, dict)]
    valid.sort(key=lambda b: SEVERITY_ORDER.get(b.get("severity"), 3))  # стабильно: сначала серьёзные
    return [_clean(b.get("description")) or _clean(b.get("type")) for b in valid]


def format_value_amd(value) -> str | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return f"{value:,}".replace(",", " ") + " AMD"


def _build(result: dict, resource_url, deadline, value_amd, max_items: int, max_chars: int) -> str:
    procurement = result.get("procurement") if isinstance(result.get("procurement"), dict) else {}
    logistics = result.get("logistics") if isinstance(result.get("logistics"), dict) else {}
    subject = _clean(procurement.get("subject")) or _clean(logistics.get("service")) or _clean(result.get("summary"))
    requirements = [
        *(procurement.get("technical_requirements") or []),
        *(procurement.get("country_of_origin_requirements") or []),
        *(procurement.get("certifications") or []),
    ]

    def one(title, value):
        value = _clean(value)
        return f"<b>{title}</b>\n{_esc(_clip(value, max_chars * 2))}" if value else None

    sections = [
        HEADER,
        one("Предмет:", subject),
        one("Заказчик:", result.get("contracting_authority") or NOT_SPECIFIED),
        one("Процедура:", result.get("procedure_code")),
        one("Категория:", category_label(result.get("category"))),
        one("Дедлайн:", deadline),
        _section("Что закупают:", _bullets(_what_is_procured(procurement, logistics), min(MAX_ITEMS, max_items), max_chars)),
        _section("Количество / лоты:", [_esc(_clip(t, max_chars)) for t in _quantity_lines(procurement)]),
        one("Оценочная стоимость:", format_value_amd(value_amd)),
        _section("Почему может быть интересно:", _bullets(_why_lines(result.get("why_interesting")), MAX_WHY, max_chars)),
        _section("Ключевые требования:", _bullets(requirements, min(MAX_REQUIREMENTS, max_items), max_chars)),
        _section("⚠️ Барьеры участия:", _bullets(_barrier_texts(result.get("participation_barriers")),
                                                  min(MAX_BARRIERS, max_items), max_chars)),
        _section("❓ Нужно уточнить:", _bullets(result.get("missing_information") or [],
                                                min(MAX_MISSING, max_items + 1), max_chars)),
        one("Уверенность анализа:", CONFIDENCE_RU.get(result.get("confidence"), result.get("confidence"))),
    ]
    url = _clean(resource_url)
    if url and url.lower().startswith(("http://", "https://")):
        sections.append(f'<b>Ссылка:</b>\n<a href="{html.escape(url, quote=True)}">{LINK_TEXT}</a>')
    return "\n\n".join(s for s in sections if s)


def build_card(result: dict, resource_url: str | None, deadline: str | None = None,
               value_amd: int | None = None) -> str:
    """
    result — сохранённый deep result (dict), deadline — уже готовая строка срока (или None),
    value_amd — ТОЛЬКО официально подтверждённая стоимость (value_facts), иначе None.
    """
    text = ""
    for max_items, max_chars in SIZE_PROFILES:
        text = _build(result, resource_url, deadline, value_amd, max_items, max_chars)
        if len(text) <= TELEGRAM_MESSAGE_LIMIT:
            break
    return text
