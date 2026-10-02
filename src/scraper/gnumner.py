"""
Первый рабочий модуль получения тендеров с сайта gnumner.minfin.am.

Отвечает только за сетевой запрос страницы списка объявлений и передачу
полученного HTML в уже существующий parser (tender_list_parser.parse_tenders).
Сам парсинг здесь не дублируется.
"""

import sys

import requests
import truststore

from src import http_tls
from src.scraper.tender_list_parser import parse_tenders

BASE_URL = "https://gnumner.minfin.am/ru/page/obyavleniya_o_zakupkakh_/"
TIMEOUT = 15

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}


def configure_tls():
    """
    Подключает системное хранилище сертификатов ОС (Windows Schannel/CryptoAPI)
    вместо certifi. Проверка сертификата (verify=True) остаётся включённой —
    меняется только источник доверенных сертификатов.

    Вызывается один раз за весь процесс: в отличие от диагностического
    site_probe.py, здесь truststore.extract_from_ssl() намеренно не
    вызывается — подмена должна действовать на протяжении всей работы
    приложения.
    """
    truststore.inject_into_ssl()


def fetch_page(page: int = 1) -> str:
    if page < 1:
        raise ValueError(f"page должен быть >= 1, получено: {page}")

    url = BASE_URL if page == 1 else f"{BASE_URL}{page}"

    response = http_tls.get(url, headers=HEADERS, timeout=TIMEOUT, verify=True)
    response.raise_for_status()
    return response.text


def fetch_tenders(page: int = 1) -> list[dict]:
    html = fetch_page(page)
    return parse_tenders(html)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    configure_tls()

    try:
        tenders = fetch_tenders(page=1)
    except requests.exceptions.RequestException as e:
        print(f"Ошибка запроса к сайту: {e}")
        return

    print(f"Найдено тендеров: {len(tenders)}")
    print()

    for i, tender in enumerate(tenders, start=1):
        print(f"[{i}] {tender['published_at']} | {tender['file_type']} | {tender['title']}")
        print(f"    {tender['attachment_url']}")


if __name__ == "__main__":
    main()
