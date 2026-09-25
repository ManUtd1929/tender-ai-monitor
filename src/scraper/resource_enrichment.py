"""
Production-модуль обогащения объявления данными со страницы ресурса
(resource_url), в зависимости от resource_type.

Отвечает только за сеть (одна HTML-страница ресурса) и разбор этой страницы.
В базу ничего не пишет, документы не скачивает и не разбирает: для ARMEPS
только собирается СПИСОК документов (documentId, имя файла, язык) из HTML.

Структура страниц проверена по реальному HTML (диагностика, 2026-09-25):

  eAuction (eauction.armeps.am/.../tender_details/...):
    пары <div class="de_t">метка</div><div class="de_v">значение</div>;
    даты публикации и срока — такие же пары, но с классами fe_t / fe_v.
    Ссылка на приглашение содержит application/documents/public_invitation/.

  ARMEPS (armeps.am/epps/cft/listContractDocuments.do):
    метки и значения в <dt>метка:</dt><dd>значение</dd>;
    документы — строки таблицы с <a onclick="downloadDocForAnonymous(<id>)">.
    Даты здесь в формате DD/MM/YYYY HH:MM.

Значения возвращаются как в HTML (только с нормализацией пробелов), пустое
или отсутствующее поле даёт None. Даты detail-страницы не приводятся к
формату list-страницы — это отдельный шаг сравнения.

Ошибки сети (requests.RequestException, HTTP-статусы) наружу не скрываются:
их обрабатывает вызывающий код.
"""

import json
import logging
import re
import sys
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup

from src.scraper.gnumner import HEADERS, TIMEOUT
from src.scraper.sections import (
    RESOURCE_ARMEPS_DOCUMENTS_PAGE,
    RESOURCE_DIRECT_FILE,
    RESOURCE_EAUCTION_TENDER_PAGE,
    classify_resource_url,
)

logger = logging.getLogger(__name__)

STATUS_SUCCESS = "success"
STATUS_PARTIAL = "partial"
STATUS_NOT_REQUIRED = "not_required"
STATUS_UNSUPPORTED = "unsupported"

# Метки eAuction (армянский) -> поле результата
EAUCTION_LABELS = {
    "Ծածկագիր": "procedure_code",
    "Վերնագիր": "detail_title",
    "Կարգավիճակ": "procedure_status",
    "Հրապարակման ժամանակ": "published_at_detail",
    "Հայտերի ընդունման վերջնաժամկետ": "deadline_at_detail",
}
EAUCTION_DOCUMENT_URL_MARKER = "application/documents/public_invitation/"

# Метки ARMEPS (без двоеточия, в нижнем регистре) -> поле результата
ARMEPS_LABELS = {
    "cft id": "procedure_code",
    "name of contracting authority": "contracting_authority",
    "title": "detail_title",
    "title (ru)": "detail_title_ru",
    "title (en)": "detail_title_en",
    "procurement type": "procurement_type",
    "procedure": "procedure_type",
    "cpv codes": "cpv_codes",
    "description": "description",
    "estimated value (amd)": "estimated_value_amd",
    "time-limit for receipt of tenders or requests to participate": "deadline_at_detail",
    "date of publication/invitation": "published_at_detail",
    "number of lots": "number_of_lots",
}

# Поля, без которых статус результата — partial (а не success)
EAUCTION_REQUIRED = (
    "procedure_code", "detail_title", "published_at_detail",
    "deadline_at_detail", "procedure_status", "document_url",
)
ARMEPS_REQUIRED = (
    "procedure_code", "contracting_authority", "detail_title",
    "published_at_detail", "deadline_at_detail",
)

DOWNLOAD_ONCLICK = re.compile(r"downloadDocForAnonymous\(\s*(\d+)\s*\)")
CPV_LINE = re.compile(r"^(\d{8})\s*-\s*(.*)$")

# Диагностические примеры для main (реальные страницы, проверенные ранее)
DIAGNOSTIC_URLS = (
    "https://eauction.armeps.am/hy/public/tender_details/tmid/d903e27c-3159-45ae-be43-6f33a82068ba",
    "https://armeps.am/epps/cft/listContractDocuments.do?resourceId=12486627",
)


# --------------------------------------------------------------------------
# Вспомогательные функции разбора
# --------------------------------------------------------------------------

def _clean(text: str) -> str | None:
    """Нормализует пробелы; пустая строка и заглушка N/A дают None."""
    text = " ".join(text.split())
    if not text or text.upper() == "N/A":
        return None
    return text


def _element_text(element) -> str | None:
    return _clean(element.get_text(" ", strip=True))


def _normalize_label(text: str) -> str:
    """'Date of Publication/Invitation :' -> 'date of publication/invitation'."""
    text = " ".join(text.split()).rstrip(":").rstrip()
    return text.casefold()


def _parse_cpv_codes(dd) -> list[dict] | None:
    """
    Строки dd (разделены <br/>) вида '44131300-название' -> {"code", "name"}.
    Строка, не подошедшая под шаблон, сохраняется как есть: code=None.
    """
    cpv_codes = []
    for line in dd.stripped_strings:
        line = " ".join(line.split())
        match = CPV_LINE.match(line)
        if match:
            cpv_codes.append({"code": match.group(1), "name": match.group(2) or None})
        else:
            cpv_codes.append({"code": None, "name": line})
    return cpv_codes or None


def _parse_int(text: str | None) -> int | None:
    return int(text) if text and text.isdigit() else None


# --------------------------------------------------------------------------
# eAuction
# --------------------------------------------------------------------------

def parse_eauction_detail(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    result = {field: None for field in EAUCTION_LABELS.values()}
    result["document_url"] = None

    for label_div in soup.find_all("div", class_=["de_t", "fe_t"]):
        field = EAUCTION_LABELS.get(" ".join(label_div.get_text().split()))
        if field is None or result[field] is not None:
            continue
        value_div = label_div.find_next_sibling("div", class_=["de_v", "fe_v"])
        if value_div is not None:
            result[field] = _element_text(value_div)

    for link in soup.find_all("a", href=True):
        if EAUCTION_DOCUMENT_URL_MARKER in link["href"]:
            result["document_url"] = link["href"].strip()
            break

    return result


def fetch_eauction_page(resource_url: str) -> str:
    response = requests.get(resource_url, headers=HEADERS, timeout=TIMEOUT, verify=True)
    response.raise_for_status()

    logger.info("Страница eAuction получена: %s (%d символов)", resource_url, len(response.text))
    return response.text


# --------------------------------------------------------------------------
# ARMEPS
# --------------------------------------------------------------------------

def parse_armeps_detail(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    result = {field: None for field in ARMEPS_LABELS.values()}

    for dt in soup.find_all("dt"):
        field = ARMEPS_LABELS.get(_normalize_label(dt.get_text()))
        if field is None or result[field] is not None:
            continue
        dd = dt.find_next_sibling("dd")
        if dd is None:
            continue

        if field == "cpv_codes":
            result[field] = _parse_cpv_codes(dd)
        elif field == "number_of_lots":
            result[field] = _parse_int(_element_text(dd))
        else:
            result[field] = _element_text(dd)

    return result


def _header_columns(table) -> dict[str, int]:
    """Индексы колонок таблицы по тексту <th> ('title', 'file', 'description', 'lang.')."""
    return {
        _normalize_label(th.get_text()): index
        for index, th in enumerate(table.find_all("th"))
    }


def _cell_text(cells: list, columns: dict[str, int], header: str) -> str | None:
    index = columns.get(header)
    if index is None or index >= len(cells):
        return None
    return _element_text(cells[index])


def parse_armeps_documents(html: str) -> list[dict]:
    """
    Документы из строк таблицы, где есть onclick="downloadDocForAnonymous(<id>)".
    Колонки определяются по заголовкам таблицы, а не по номерам.
    """
    soup = BeautifulSoup(html, "html.parser")
    documents = []

    for link in soup.find_all("a", onclick=DOWNLOAD_ONCLICK):
        row = link.find_parent("tr")
        table = link.find_parent("table")
        if row is None or table is None:
            continue

        columns = _header_columns(table)
        cells = row.find_all("td", recursive=False)

        documents.append({
            "document_id": DOWNLOAD_ONCLICK.search(link["onclick"]).group(1),
            "filename": _element_text(link),
            "language": _cell_text(cells, columns, "lang."),
            "title": _cell_text(cells, columns, "title"),
            "description": _cell_text(cells, columns, "description"),
        })

    return documents


def extract_resource_id(resource_url: str) -> str | None:
    values = parse_qs(urlparse(resource_url).query).get("resourceId")
    return values[0] if values else None


def fetch_armeps_page(resource_url: str) -> str:
    session = requests.Session()
    try:
        response = session.get(resource_url, headers=HEADERS, timeout=TIMEOUT, verify=True)
        response.raise_for_status()
    finally:
        session.close()

    logger.info("Страница ARMEPS получена: %s (%d символов)", resource_url, len(response.text))
    return response.text


# --------------------------------------------------------------------------
# Общая функция
# --------------------------------------------------------------------------

def _status_for(fields: dict, required: tuple[str, ...]) -> str:
    missing = [name for name in required if not fields.get(name)]
    if missing:
        logger.warning("Обогащение частичное, не найдены поля: %s", ", ".join(missing))
        return STATUS_PARTIAL
    return STATUS_SUCCESS


def enrich_announcement(announcement: dict) -> dict:
    """
    Новый dict: копия announcement + enrichment_status + данные detail-страницы.
    Исходный announcement не изменяется. Сетевые ошибки не перехватываются.
    """
    resource_type = announcement.get("resource_type")
    resource_url = announcement.get("resource_url")
    enriched = dict(announcement)

    if resource_type == RESOURCE_EAUCTION_TENDER_PAGE:
        fields = parse_eauction_detail(fetch_eauction_page(resource_url))
        enriched.update(fields)
        enriched["enrichment_status"] = _status_for(fields, EAUCTION_REQUIRED)

    elif resource_type == RESOURCE_ARMEPS_DOCUMENTS_PAGE:
        html = fetch_armeps_page(resource_url)
        fields = parse_armeps_detail(html)
        documents = parse_armeps_documents(html)
        enriched.update(fields)
        enriched["resource_id"] = extract_resource_id(resource_url)
        enriched["documents"] = documents
        enriched["enrichment_status"] = _status_for(
            {**fields, "documents": documents}, ARMEPS_REQUIRED + ("documents",)
        )

    elif resource_type == RESOURCE_DIRECT_FILE:
        enriched["direct_file_url"] = resource_url
        enriched["enrichment_status"] = STATUS_NOT_REQUIRED

    else:
        logger.warning("Тип ресурса не поддерживается: %r (%s)", resource_type, resource_url)
        enriched["enrichment_status"] = STATUS_UNSUPPORTED

    logger.info(
        "Обогащение: %s, %s -> %s", resource_type, resource_url, enriched["enrichment_status"],
    )
    return enriched


# --------------------------------------------------------------------------
# Диагностический запуск
# --------------------------------------------------------------------------

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    for url in DIAGNOSTIC_URLS:
        announcement = {"resource_url": url, "resource_type": classify_resource_url(url)}
        print("=" * 78)
        print(f"{announcement['resource_type']}: {url}")

        try:
            result = enrich_announcement(announcement)
        except requests.exceptions.RequestException as e:
            logger.error("Не удалось обогатить %s: %s", url, e)
            print(f"ОШИБКА: {type(e).__name__}: {e}")
            continue

        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
