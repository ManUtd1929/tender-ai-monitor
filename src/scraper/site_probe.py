"""
Диагностический скрипт для изучения структуры сайта gnumner.minfin.am.

Этот скрипт НЕ парсит тендеры. Он только проверяет, как сайт отвечает
на обычный HTTP-запрос: код ответа, редиректы, кодировку и сырой HTML.
Результат используется для планирования следующего этапа (написание парсера).
"""

import os
import sys
import warnings

import requests
from urllib3.exceptions import InsecureRequestWarning

URL = "https://gnumner.minfin.am/ru/page/obyavleniya_o_zakupkakh_/"
TIMEOUT = 15

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "gnumner_main.html")


def print_diagnostics(response):
    html = response.text
    print(f"HTTP status code: {response.status_code}")
    print(f"Конечный URL: {response.url}")
    print(f"Content-Type: {response.headers.get('Content-Type')}")
    print(f"Encoding: {response.encoding}")
    print(f"Размер HTML: {len(html)} символов")
    print("Первые 500 символов HTML:")
    print(html[:500])


def save_if_ok(response):
    if response.ok:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"HTML сохранён в: {os.path.abspath(OUTPUT_FILE)}")


def probe_site():
    # На Windows-консоли вывод может быть в cp1252, а не UTF-8 —
    # без этого print() с кириллицей падает с UnicodeEncodeError.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    # Основной, "боевой" вариант запроса — проверка сертификата включена.
    try:
        response = requests.get(URL, headers=HEADERS, timeout=TIMEOUT, verify=True)
    except requests.exceptions.SSLError as e:
        print(f"SSL/TLS ошибка при verify=True: {e}")
        print(
            "Запускаю диагностический запрос с verify=False "
            "только для проверки доступности сайта."
        )
        _probe_with_verify_disabled()
        return
    except requests.exceptions.RequestException as e:
        print(f"Ошибка запроса: {e}")
        return

    print_diagnostics(response)
    save_if_ok(response)


def _probe_with_verify_disabled():
    """
    ВРЕМЕННЫЙ диагностический обход.

    verify=False используется здесь ТОЛЬКО чтобы проверить, доступен ли сайт
    вообще, если основной запрос с verify=True упал из-за SSL-ошибки.
    Это НЕ безопасное production-решение: проверка сертификата сервера
    полностью отключена, что открывает риск MITM. Перед реализацией
    настоящего scraper'а нужно заменить этот обход нормальным решением
    (например, пакетом truststore, использующим системное хранилище
    сертификатов Windows, или дополненным CA bundle с недостающим
    промежуточным сертификатом).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InsecureRequestWarning)
        try:
            response = requests.get(URL, headers=HEADERS, timeout=TIMEOUT, verify=False)
        except requests.exceptions.RequestException as e:
            print(f"Диагностический запрос (verify=False) тоже завершился ошибкой: {e}")
            return

    print_diagnostics(response)
    save_if_ok(response)


if __name__ == "__main__":
    probe_site()
