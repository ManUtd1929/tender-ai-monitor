"""
Production-модуль получения объявлений из отдельных разделов закупочных
процедур сайта gnumner.minfin.am.

Отвечает только за сеть (HTML страницы раздела) и приведение объявлений к
нормализованной структуре. В базу ничего не пишет, документы не скачивает,
на страницы eAuction/ARMEPS не заходит — тип ресурса определяется только по
самому URL.

Разбор div.tender переиспользуется из tender_list_parser.parse_tenders():
по результатам диагностики (section_probe.py) структура объявлений во всех
разделах такая же, как на общей странице.
"""

import logging
import re
import sys
from collections import Counter
from urllib.parse import urlparse

import requests

from src.scraper.gnumner import HEADERS, TIMEOUT, configure_tls
from src.scraper.tender_list_parser import parse_tenders

logger = logging.getLogger(__name__)

SECTIONS = {
    "electronic_auction": {
        "section_key": "electronic_auction",
        "section_name": "Электронный аукцион",
        "section_url": "https://gnumner.minfin.am/ru/page/obyavlenie_i_priglashenie_na_elektronnyi_auktcion/",
    },
    "open_competition": {
        "section_key": "open_competition",
        "section_name": "Открытый конкурс",
        "section_url": "https://gnumner.minfin.am/ru/page/obyavlenie_i_priglashenie_na_otkrytyi_konkurs/",
    },
    "request_for_quotations": {
        "section_key": "request_for_quotations",
        "section_name": "Запрос котировок",
        "section_url": "https://gnumner.minfin.am/ru/page/obyavlenie_i_priglashenie_po_zaprosu_kotirovok/",
    },
    "price_proposals": {
        "section_key": "price_proposals",
        "section_name": "Запрос ценовых предложений",
        "section_url": "https://gnumner.minfin.am/ru/page/zapros_tcenovykh_predlozhenii/",
    },
}

RESOURCE_EAUCTION_TENDER_PAGE = "eauction_tender_page"
RESOURCE_ARMEPS_DOCUMENTS_PAGE = "armeps_documents_page"
RESOURCE_DIRECT_FILE = "direct_file"
RESOURCE_UNKNOWN = "unknown"

EAUCTION_HOST = "eauction.armeps.am"
EAUCTION_TENDER_PATH = "/public/tender_details/"

ARMEPS_HOST = "armeps.am"
ARMEPS_DOCUMENTS_PATH = "/epps/cft/listContractDocuments.do"

GNUMNER_HOST = "gnumner.minfin.am"
GNUMNER_FILES_PATH = "/website/images/original/"

DIRECT_FILE_EXTENSIONS = (".pdf", ".doc", ".docx", ".xls", ".xlsx", ".zip", ".rar", ".7z", ".xml")

DATETIME_REGEX = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
DEADLINE_PATTERN = re.compile(rf"\bДо\s+({DATETIME_REGEX})", re.IGNORECASE)
NO_DEADLINE_PATTERN = re.compile(r"Бессрочн", re.IGNORECASE)


def get_section(section_key: str) -> dict:
    try:
        return SECTIONS[section_key]
    except KeyError:
        raise ValueError(
            f"Неизвестный раздел: {section_key!r}. Доступны: {', '.join(SECTIONS)}"
        ) from None


# --------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------

def build_section_page_url(section_key: str, page: int = 1) -> str:
    if page < 1:
        raise ValueError(f"page должен быть >= 1, получено: {page}")

    base_url = get_section(section_key)["section_url"]
    if page == 1:
        return base_url
    return f"{base_url.rstrip('/')}/{page}"


def fetch_section_page(section_key: str, page: int = 1) -> str:
    url = build_section_page_url(section_key, page)

    response = requests.get(url, headers=HEADERS, timeout=TIMEOUT, verify=True)
    response.raise_for_status()

    logger.info(
        "Страница раздела получена: %s, страница %d, %s (%d символов)",
        section_key, page, url, len(response.text),
    )
    return response.text


# --------------------------------------------------------------------------
# Разбор
# --------------------------------------------------------------------------

def classify_resource_url(url: str) -> str:
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower().removeprefix("www.")
    path = parsed.path  # без query string и fragment

    if host == EAUCTION_HOST and EAUCTION_TENDER_PATH in path:
        return RESOURCE_EAUCTION_TENDER_PAGE

    if host == ARMEPS_HOST and ARMEPS_DOCUMENTS_PATH in path:
        return RESOURCE_ARMEPS_DOCUMENTS_PAGE

    if path.lower().endswith(DIRECT_FILE_EXTENSIONS):
        return RESOURCE_DIRECT_FILE

    if host == GNUMNER_HOST and path.startswith(GNUMNER_FILES_PATH):
        return RESOURCE_DIRECT_FILE

    return RESOURCE_UNKNOWN


def extract_deadline(tender_time_raw: str) -> str | None:
    """
    Срок подачи из текста вида
    "(опубликован 2026-09-25 18:59:34-от До 2026-10-02 11:10:00 час включен)".

    "Бессрочный" и любой нераспознанный шаблон дают None — срок не
    придумывается. Нераспознанный шаблон дополнительно попадает в лог.
    """
    match = DEADLINE_PATTERN.search(tender_time_raw)
    if match:
        return match.group(1)

    if not NO_DEADLINE_PATTERN.search(tender_time_raw):
        logger.warning("Шаблон срока не распознан, deadline_at=None: %r", tender_time_raw.strip())
    return None


def parse_section_announcements(
    html: str,
    section_key: str,
    page_url: str | None = None,
) -> list[dict]:
    """
    Нормализованные объявления из HTML страницы раздела.

    page_url — адрес реально загруженной страницы (для source_page_url);
    если не задан, берётся основной URL раздела.
    """
    section = get_section(section_key)
    source_page_url = page_url or section["section_url"]

    announcements = []
    for tender in parse_tenders(html):
        resource_url = tender["attachment_url"]
        tender_time_raw = tender["tender_time_raw"]

        announcements.append({
            "title": tender["title"],
            "source_section": section["section_key"],
            "source_section_name": section["section_name"],
            "source_page_url": source_page_url,
            "resource_url": resource_url,
            "resource_type": classify_resource_url(resource_url),
            "published_at": tender["published_at"],
            "deadline_at": extract_deadline(tender_time_raw),
            "tender_time_raw": tender_time_raw,
        })

    logger.info("Раздел %s: разобрано объявлений: %d", section_key, len(announcements))
    return announcements


def fetch_section_announcements(section_key: str, page: int = 1) -> list[dict]:
    html = fetch_section_page(section_key, page)
    page_url = build_section_page_url(section_key, page)
    return parse_section_announcements(html, section_key, page_url)


def _collect_sections(page: int) -> tuple[list[dict], list[str]]:
    """Все разделы по очереди; возвращает (объявления, ключи упавших разделов)."""
    announcements = []
    failed_sections = []

    for section_key in SECTIONS:
        try:
            announcements.extend(fetch_section_announcements(section_key, page))
        except Exception:  # ошибка одного раздела не должна останавливать остальные
            logger.exception("Не удалось обработать раздел %s, страница %d", section_key, page)
            failed_sections.append(section_key)

    return announcements, failed_sections


def fetch_all_sections(page: int = 1) -> list[dict]:
    announcements, failed_sections = _collect_sections(page)

    logger.info(
        "Все разделы, страница %d: объявлений %d, упавших разделов %d",
        page, len(announcements), len(failed_sections),
    )
    return announcements


# --------------------------------------------------------------------------
# Диагностический вывод
# --------------------------------------------------------------------------

def find_duplicate_urls(announcements: list[dict]) -> dict[str, int]:
    """resource_url, встречающиеся больше одного раза, с числом повторов."""
    counts = Counter(a["resource_url"] for a in announcements)
    return {url: n for url, n in counts.items() if n > 1}


def print_section_report(section_key: str, announcements: list[dict], failed: bool):
    section = SECTIONS[section_key]
    print("=" * 78)
    print(f"Раздел: {section['section_name']} ({section_key})")

    if failed:
        print("ОШИБКА: раздел не удалось получить (подробности в логе)")
        print()
        return

    with_deadline = sum(1 for a in announcements if a["deadline_at"])
    types = Counter(a["resource_type"] for a in announcements)

    print(f"Объявлений: {len(announcements)}")
    print("Типы ресурсов: " + (" ".join(f"{k}:{v}" for k, v in types.most_common()) or "—"))
    print(f"С deadline_at: {with_deadline}")
    print()

    for i, a in enumerate(announcements[:3], start=1):
        print(f"  [{i}] source_section: {a['source_section']}")
        print(f"      title:          {a['title']}")
        print(f"      published_at:   {a['published_at']}")
        print(f"      deadline_at:    {a['deadline_at']}")
        print(f"      resource_type:  {a['resource_type']}")
        print(f"      resource_url:   {a['resource_url']}")
    print()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    configure_tls()

    announcements, failed_sections = _collect_sections(page=1)

    for section_key in SECTIONS:
        section_items = [a for a in announcements if a["source_section"] == section_key]
        print_section_report(section_key, section_items, section_key in failed_sections)

    duplicates = find_duplicate_urls(announcements)
    unique_count = len({a["resource_url"] for a in announcements})
    duplicate_count = len(announcements) - unique_count

    print("=" * 78)
    print(f"Всего объявлений: {len(announcements)}")
    print(f"Уникальных resource_url: {unique_count}")
    print(f"Дубликатов resource_url: {duplicate_count}")
    if failed_sections:
        print(f"Не удалось получить разделы: {', '.join(failed_sections)}")

    if duplicates:
        print()
        print("Повторяющиеся resource_url:")
        for url, count in duplicates.items():
            sections = sorted({a["source_section"] for a in announcements if a["resource_url"] == url})
            print(f"  x{count} [{', '.join(sections)}] {url}")


if __name__ == "__main__":
    main()
