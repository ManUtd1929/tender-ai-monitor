"""
Диагностический скрипт: исследование отдельных разделов закупочных процедур
сайта gnumner.minfin.am.

Ничего не сохраняет (ни в базу, ни в файлы), не скачивает вложения и не
загружает страницы пагинации — только по одному GET на каждый раздел.

Эталон структуры div.tender берётся из уже сохранённого снимка общей страницы
data/gnumner_main.html (тот же файл, на котором построен tender_list_parser).
Для разбора объявлений используется существующий parse_tenders() — без
изменений; скрипт лишь проверяет, подходит ли он для каждого раздела.
"""

import re
import sys
from collections import Counter
from datetime import datetime

import requests
from bs4 import BeautifulSoup, Tag

from src import http_tls
from src.scraper.gnumner import HEADERS, TIMEOUT, configure_tls
from src.scraper.tender_list_parser import (
    PUBLISHED_AT_PATTERN,
    parse_tenders,
    read_html_file,
)

SECTIONS = [
    (
        "Электронный аукцион",
        "https://gnumner.minfin.am/ru/page/obyavlenie_i_priglashenie_na_elektronnyi_auktcion/",
    ),
    (
        "Открытый конкурс",
        "https://gnumner.minfin.am/ru/page/obyavlenie_i_priglashenie_na_otkrytyi_konkurs/",
    ),
    (
        "Запрос котировок",
        "https://gnumner.minfin.am/ru/page/obyavlenie_i_priglashenie_po_zaprosu_kotirovok/",
    ),
    (
        "Запрос ценовых предложений",
        "https://gnumner.minfin.am/ru/page/zapros_tcenovykh_predlozhenii/",
    ),
]

SAMPLE_SIZE = 5
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# --------------------------------------------------------------------------
# Структура div.tender
# --------------------------------------------------------------------------

def _element_key(tag: Tag) -> tuple:
    """Тег без текста: имя, классы и имена атрибутов (значения не важны)."""
    return (tag.name, tuple(sorted(tag.get("class", []))), tuple(sorted(tag.attrs)))


def signature(tag: Tag) -> tuple:
    """Вложенная «форма» тега: сравнивает структуру, а не содержимое."""
    return _element_key(tag) + (
        tuple(signature(child) for child in tag.children if isinstance(child, Tag)),
    )


def element_keys(tag: Tag) -> set:
    return {_element_key(tag)} | {_element_key(t) for t in tag.find_all(True)}


def render_signature(sig: tuple, indent: int = 0) -> list[str]:
    name, classes, attrs, children = sig
    label = name + "".join(f".{c}" for c in classes)
    other_attrs = [a for a in attrs if a != "class"]
    if other_attrs:
        label += " [" + ", ".join(other_attrs) + "]"

    lines = ["  " * indent + label]
    for child in children:
        lines.extend(render_signature(child, indent + 1))
    return lines


def load_reference() -> dict:
    """Эталон: структура и адреса объявлений из сохранённой общей страницы."""
    html = read_html_file()
    soup = BeautifulSoup(html, "html.parser")
    divs = soup.find_all("div", class_="tender")
    if not divs:
        raise RuntimeError("В сохранённой общей странице нет div.tender")

    parsed = parse_tenders(html)
    dates = sorted(t["published_at"] for t in parsed if t["published_at"])

    return {
        "signatures": {signature(d) for d in divs},
        "known_keys": set().union(*(element_keys(d) for d in divs)),
        "urls": {t["attachment_url"] for t in parsed},
        "date_min": dates[0] if dates else None,
        "date_max": dates[-1] if dates else None,
        "section_links": {a["href"] for a in soup.select("#breadcrumb ul a[href]")},
    }


# --------------------------------------------------------------------------
# Анализ одного раздела
# --------------------------------------------------------------------------

def analyze_pagination(soup: BeautifulSoup, url: str) -> dict:
    node = soup.find(class_=re.compile("pagin", re.I))
    if node is None:
        return {"present": False}

    scheme = re.compile(re.escape(url) + r"(\d+)")
    numbers = []          # (номер, href или None) в порядке следования в HTML
    foreign_links = []    # ссылки с номером, но не по схеме <URL раздела>N
    nav_links = []        # стрелки и прочие нечисловые ссылки

    for el in node.find_all(["a", "span"]):
        text = el.get_text(strip=True)
        if el.name == "a" and el.get("href"):
            if not text.isdigit():
                nav_links.append(el["href"])
                continue
            match = scheme.fullmatch(el["href"])
            if match:
                numbers.append((int(match.group(1)), el["href"]))
            else:
                foreign_links.append(el["href"])
        elif el.name == "span" and text.isdigit():
            numbers.append((int(text), None))  # текущая страница — не ссылка

    max_page = None
    if numbers:
        highest = max(n for n, _ in numbers)
        # Максимум достоверен, только если это последний номер в списке страниц.
        if numbers[-1][0] == highest:
            max_page = highest

    second_url = next((href for n, href in numbers if n == 2 and href), None)

    return {
        "present": True,
        "text": node.get_text(" ", strip=True),
        "max_page": max_page,
        "second_url": second_url,
        "scheme_ok": bool(numbers) and not foreign_links,
        "foreign_links": foreign_links,
        "nav_links": nav_links,
    }


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text)


def find_extra_data(tender_divs: list[Tag], ref: dict) -> dict:
    """Ищет в div.tender всё, что выходит за title / вложение / tender_time."""
    extra_elements = Counter()
    multi_link_count = 0
    extra_text_examples = []
    extra_text_count = 0

    for div in tender_divs:
        for key in element_keys(div) - ref["known_keys"]:
            extra_elements[key] += 1

        if len(div.find_all("a")) > 1:
            multi_link_count += 1

        link = div.select_one("div.tender_title a")
        time_tag = div.find("p", class_="tender_time")
        leftover = _normalize(div.get_text())
        for part in (link, time_tag):
            if part is not None:
                leftover = leftover.replace(_normalize(part.get_text()), "", 1)
        if leftover:
            extra_text_count += 1
            if len(extra_text_examples) < 2:
                extra_text_examples.append(div.get_text(" ", strip=True)[:200])

    return {
        "extra_elements": extra_elements,
        "multi_link_count": multi_link_count,
        "extra_text_count": extra_text_count,
        "extra_text_examples": extra_text_examples,
        "has_extra": bool(extra_elements or multi_link_count or extra_text_count),
    }


def analyze_section(name: str, url: str, ref: dict) -> dict:
    response = http_tls.get(url, headers=HEADERS, timeout=TIMEOUT, verify=True)
    response.raise_for_status()
    html = response.text

    soup = BeautifulSoup(html, "html.parser")
    tender_divs = soup.find_all("div", class_="tender")
    parsed = parse_tenders(html)

    signatures = Counter(signature(d) for d in tender_divs)
    different = [s for s in signatures if s not in ref["signatures"]]
    same_structure = None if not tender_divs else not different

    reuse_problems = []
    if tender_divs:
        if same_structure is False:
            reuse_problems.append("структура div.tender отличается от общей страницы")
        if len(parsed) != len(tender_divs):
            reuse_problems.append(
                f"parse_tenders распознал {len(parsed)} из {len(tender_divs)} div.tender"
            )
        no_date = sum(1 for t in parsed if t["published_at"] is None)
        if no_date:
            reuse_problems.append(f"объявлений без published_at: {no_date}")
        no_type = sum(1 for t in parsed if not t["file_type"])
        if no_type:
            reuse_problems.append(f"объявлений без file_type: {no_type}")

    dates = sorted(t["published_at"] for t in parsed if t["published_at"])
    in_ref_range = [
        t for t in parsed
        if t["published_at"] and ref["date_min"] and ref["date_min"] <= t["published_at"] <= ref["date_max"]
    ]

    return {
        "name": name,
        "url": url,
        "final_url": response.url,
        "status": response.status_code,
        "html_size": len(html),
        "page_title": soup.title.get_text(strip=True) if soup.title else None,
        "tender_count": len(tender_divs),
        "links_total": sum(len(d.find_all("a", href=True)) for d in tender_divs),
        "links_in_title": sum(len(d.select("div.tender_title a[href]")) for d in tender_divs),
        "parsed": parsed,
        "file_types": Counter(t["file_type"] or "(пусто)" for t in parsed),
        "time_templates": Counter(
            PUBLISHED_AT_PATTERN.sub("<дата>", t["tender_time_raw"].strip()) for t in parsed
        ),
        "date_newest": dates[-1] if dates else None,
        "date_oldest": dates[0] if dates else None,
        "pagination": analyze_pagination(soup, url),
        "extra": find_extra_data(tender_divs, ref),
        "same_structure": same_structure,
        "different_signatures": different,
        "reusable": None if not tender_divs else not reuse_problems,
        "reuse_problems": reuse_problems,
        "in_ref_range_total": len(in_ref_range),
        "in_ref_range_present": sum(1 for t in in_ref_range if t["attachment_url"] in ref["urls"]),
    }


# --------------------------------------------------------------------------
# Вывод
# --------------------------------------------------------------------------

def types_text(counter: Counter) -> str:
    return " ".join(f"{k}:{v}" for k, v in counter.most_common()) or "—"


def pagination_text(p: dict) -> str:
    if not p["present"]:
        return "нет"
    max_page = p["max_page"] if p["max_page"] is not None else "не определён"
    return f"да, макс. страница: {max_page}"


def structure_text(same_structure) -> str:
    if same_structure is None:
        return "нет данных"
    return "такая же" if same_structure else "отличается"


def reuse_text(reusable) -> str:
    if reusable is None:
        return "нет данных"
    return "да" if reusable else "нет"


def print_section(r: dict):
    print("=" * 78)
    print(f"Раздел: {r['name']}")
    print(f"URL: {r['url']}")
    if r["final_url"] != r["url"]:
        print(f"Конечный URL (после редиректов): {r['final_url']}")
    print(f"<title>: {r['page_title']}")
    print(f"HTTP status code: {r['status']}")
    print(f"Размер HTML: {r['html_size']} символов")
    print(f"div.tender на странице: {r['tender_count']}")
    print(
        f"Ссылок <a href> внутри div.tender: {r['links_total']} "
        f"(из них в div.tender_title: {r['links_in_title']})"
    )

    p = r["pagination"]
    print(f"Pagination: {pagination_text(p)}")
    if p["present"]:
        print(f"  Текст блока pagination: {p['text']}")
        print(f"  URL второй страницы (из ссылки в HTML): {p['second_url'] or 'не найден'}")
        if p["scheme_ok"]:
            print("  Схема <URL раздела>N: да, все числовые ссылки ей соответствуют (по HTML, запросом не проверялось)")
        else:
            print("  Схема <URL раздела>N: НЕ подтверждена")
        if p["foreign_links"]:
            print(f"  Числовые ссылки не по схеме: {p['foreign_links']}")

    print()
    print(f"Первые {min(SAMPLE_SIZE, len(r['parsed']))} объявлений (по parse_tenders):")
    for i, t in enumerate(r["parsed"][:SAMPLE_SIZE], start=1):
        print(f"  [{i}] title:          {t['title']}")
        print(f"      attachment_url: {t['attachment_url']}")
        print(f"      filename:       {t['filename']}")
        print(f"      file_type:      {t['file_type']}")
        print(f"      tender_time_raw: {t['tender_time_raw'].strip()}")
        print(f"      published_at:   {t['published_at'] or 'даты нет'}")

    print()
    print(f"Типы файлов в разделе: {types_text(r['file_types'])}")
    for template, count in r["time_templates"].most_common():
        print(f"Шаблон tender_time_raw: {template!r} — {count}")

    print()
    extra = r["extra"]
    if not extra["has_extra"]:
        print(
            "Дополнительные данные в div.tender: НЕТ. Есть только заголовок-ссылка "
            "на вложение, иконка типа файла и tender_time (заказчик, deadline, CPV, "
            "стоимость, отдельная ссылка на страницу тендера — отсутствуют)."
        )
    else:
        print("Дополнительные данные в div.tender: ЕСТЬ")
        for (tag, classes, attrs), count in extra["extra_elements"].items():
            print(f"  элемент вне эталона: {tag}.{'.'.join(classes)} атрибуты={attrs} — в {count} div.tender")
        if extra["multi_link_count"]:
            print(f"  div.tender с более чем одной ссылкой: {extra['multi_link_count']}")
        if extra["extra_text_count"]:
            print(f"  div.tender с текстом кроме title и tender_time: {extra['extra_text_count']}")
            for example in extra["extra_text_examples"]:
                print(f"    пример: {example}")

    print()
    print(f"Структура div.tender по сравнению с общей страницей: {structure_text(r['same_structure'])}")
    for sig in r["different_signatures"]:
        print("  Отличающаяся структура:")
        for line in render_signature(sig, 2):
            print(line)
    print(f"Можно переиспользовать parse_tenders(): {reuse_text(r['reusable'])}")
    for problem in r["reuse_problems"]:
        print(f"  - {problem}")
    print()


def print_reference(ref: dict):
    print("=" * 78)
    print("Эталон: структура div.tender общей страницы (data/gnumner_main.html)")
    for sig in ref["signatures"]:
        for line in render_signature(sig, 1):
            print(line)
    print(f"Диапазон published_at в снимке: {ref['date_min']} .. {ref['date_max']}")
    print()


def print_summary(reports: list[dict], errors: list[tuple]):
    header = (
        "Раздел", "Тендеров на странице", "Пагинация", "Типы файлов",
        "HTML структура", "Можно переиспользовать parse_tenders",
    )
    rows = [
        (
            r["name"], str(r["tender_count"]), pagination_text(r["pagination"]),
            types_text(r["file_types"]), structure_text(r["same_structure"]),
            reuse_text(r["reusable"]),
        )
        for r in reports
    ]
    rows += [(name, "ОШИБКА", "—", "—", "—", "—") for name, _, _ in errors]

    widths = [max(len(row[i]) for row in [header] + rows) for i in range(len(header))]

    print("=" * 78)
    print("СРАВНИТЕЛЬНАЯ СВОДКА")
    print(" | ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(c.ljust(w) for c, w in zip(row, widths)))
    for name, url, error in errors:
        print(f"Ошибка раздела «{name}» ({url}): {type(error).__name__}: {error}")
    print()


def print_conclusion(reports: list[dict], errors: list[tuple], ref: dict):
    print("=" * 78)
    print("ТЕХНИЧЕСКИЙ ВЫВОД")

    with_tenders = [r["name"] for r in reports if r["tender_count"] > 0]
    empty = [r["name"] for r in reports if r["tender_count"] == 0]
    print()
    print("1. Страницы с объявлениями (div.tender > 0):")
    print(f"   содержат: {', '.join(with_tenders) or 'ни одна'}")
    if empty:
        print(f"   не содержат div.tender: {', '.join(empty)}")
    if errors:
        print(f"   не удалось проверить (ошибка): {', '.join(name for name, _, _ in errors)}")

    print()
    print("2. Одинаковость структуры:")
    differing = [r["name"] for r in reports if r["same_structure"] is False]
    if not with_tenders:
        print("   сравнивать нечего")
    elif not differing:
        print("   у всех проверенных разделов структура div.tender совпадает с общей страницей")
    else:
        print(f"   отличаются от общей страницы: {', '.join(differing)}")
    not_reusable = [r["name"] for r in reports if r["reusable"] is False]
    if with_tenders:
        if not_reusable:
            print(f"   parse_tenders() без изменений НЕ подходит для: {', '.join(not_reusable)}")
        else:
            print("   parse_tenders() без изменений подходит для всех разделов с объявлениями")

    print()
    print("3. Признаки актуальности (только то, что видно в HTML):")
    now = datetime.now()
    for r in reports:
        if not r["date_newest"]:
            print(f"   {r['name']}: дат публикации нет")
            continue
        age_days = (now - datetime.strptime(r["date_newest"], DATE_FORMAT)).days
        max_page = r["pagination"].get("max_page") if r["pagination"]["present"] else 1
        print(
            f"   {r['name']}: новейшее {r['date_newest']} (~{age_days} дн. назад), "
            f"самое старое на 1-й странице {r['date_oldest']}, "
            f"страниц в разделе: {max_page if max_page is not None else 'не определено'}"
        )
    if any(r["extra"]["has_extra"] for r in reports):
        print("   В части разделов найдены дополнительные поля (см. подробный вывод выше).")
    else:
        print(
            "   Полей статуса, срока приёма заявок или заказчика в div.tender нет ни в одном "
            "разделе: по HTML списка нельзя отличить текущую процедуру от завершённой."
        )
    templates = Counter()
    for r in reports:
        templates.update(r["time_templates"])
    for template, count in templates.most_common():
        print(f"   tender_time_raw имеет вид {template!r} ({count}); смысл текста после даты по HTML не установлен")

    print()
    print("4. Общая страница и отдельные разделы:")
    found_links = [url for _, url in SECTIONS if url in ref["section_links"]]
    print(
        f"   В снимке общей страницы меню разделов содержит {len(ref['section_links'])} ссылок; "
        f"из 4 исследуемых разделов в нём присутствуют: {len(found_links)}"
    )
    print(f"   Диапазон published_at в снимке общей страницы: {ref['date_min']} .. {ref['date_max']}")
    missing_somewhere = False
    for r in reports:
        total, present = r["in_ref_range_total"], r["in_ref_range_present"]
        print(
            f"   {r['name']}: объявлений раздела в этом диапазоне дат: {total}, "
            f"из них есть на общей странице (по attachment_url): {present}"
        )
        if present < total:
            missing_somewhere = True
    if not any(r["in_ref_range_total"] for r in reports):
        print("   Проверка неинформативна: объявления разделов не попадают в диапазон дат снимка.")
    elif missing_somewhere:
        print(
            "   Часть объявлений разделов в пределах диапазона дат отсутствует на общей "
            "странице (1-я страница) — общая страница не содержит всё, что есть в разделах."
        )
    else:
        print(
            "   Все объявления разделов в пределах диапазона дат присутствуют на общей "
            "странице — она включает их."
        )
    print()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    configure_tls()

    try:
        ref = load_reference()
    except (OSError, RuntimeError) as e:
        print(f"Не удалось загрузить эталон общей страницы: {type(e).__name__}: {e}")
        print("Сначала сохраните её через src/scraper/site_probe.py.")
        return

    print_reference(ref)

    reports, errors = [], []
    for name, url in SECTIONS:
        try:
            report = analyze_section(name, url, ref)
        except Exception as e:  # ошибка одного раздела не должна останавливать остальные
            errors.append((name, url, e))
            print("=" * 78)
            print(f"Раздел: {name}\nURL: {url}")
            print(f"ОШИБКА: {type(e).__name__}: {e}\n")
            continue
        reports.append(report)
        print_section(report)

    print_summary(reports, errors)
    print_conclusion(reports, errors, ref)


if __name__ == "__main__":
    main()
