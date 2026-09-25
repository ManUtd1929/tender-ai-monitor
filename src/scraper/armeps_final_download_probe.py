"""
Диагностический скрипт: HTTP-поведение конечных download-endpoints ARMEPS.

Для каждого из трёх тестов, в одной общей requests.Session:
  1. открывается страница listContractDocuments.do (cookies, JSESSIONID);
  2. открывается prepareAnonymousDownload.do (Referer = страница из п.1);
  3. выполняется GET конечного endpoint (Referer = страница из п.2) сначала с
     allow_redirects=False, чтобы увидеть реальное поведение; если это редирект —
     второй GET на Location с allow_redirects=True.

Везде verify=True, User-Agent, timeout, stream=True. Тело ответов конечных
endpoints НЕ читается (ни content, ни text, ни iter_content), не сохраняется и
не разбирается — печатаются только HTTP-метаданные; ответ закрывается сразу
после чтения заголовков. Единственное исключение: при allow_redirects=True
библиотека requests сама вычитывает тела промежуточных 3xx-ответов (обычно
крошечные); тело итогового ответа не читается. Страница prepareAnonymousDownload
тоже закрывается без чтения тела. HTML страницы listContractDocuments.do
читается — только чтобы найти имя файла документа для сравнения.

Ничего не пишет в файлы и БД, POST не отправляется. Проверка TLS не отключается.
"""

import re
import sys
import unicodedata
from urllib.parse import unquote, urljoin

import requests
from bs4 import BeautifulSoup

from src.scraper.armeps_download_probe import DOWNLOAD_URL, choose_list_page, cookie_names_sent
from src.scraper.armeps_probe import DOC_EXTENSIONS, cookie_line, section, short, text_of
from src.scraper.gnumner import HEADERS, TIMEOUT

DOWNLOAD_DOC_URL = "https://armeps.am/epps/cft/downloadContractDocument.do?documentId={}&resourceId=null"
DOWNLOAD_ZIP_URL = "https://armeps.am/epps/cft/downloadCftResourceItems.do?resourceId={}&resourceType=ContractDocument"

# (название, параметр prepareAnonymousDownload, значение, конечный URL,
#  имя файла из текста задачи — только для справки; авторитетно имя из HTML страницы)
TESTS = [
    ("1. Отдельный документ documentId=12486728", "documentId", "12486728",
     DOWNLOAD_DOC_URL.format("12486728"), "hraver Eng (1).docx"),
    ("2. Отдельный документ documentId=12486849", "documentId", "12486849",
     DOWNLOAD_DOC_URL.format("12486849"), "АՄԱՀ-ԱՊ-ГՀԱՊՁԲ-26-117 rus.docx"),
    ("3. ZIP-комплект resourceId=12486673", "resourceId", "12486673",
     DOWNLOAD_ZIP_URL.format("12486673"), None),
]

FILE_NAME_END = re.compile(rf"\.(?:{DOC_EXTENSIONS})\s*$", re.I)
LOGIN_HINT = re.compile(r"log[\s_-]?in|logon|sign[\s_-]?in|authenticat", re.I)
NON_FILE_TEXT_TYPES = ("text/", "application/json", "application/javascript")

DIRECT_FILE, REDIRECT, HTML_PAGE, ERROR, UNKNOWN = "direct_file", "redirect", "html_page", "error", "unknown"


# --------------------------------------------------------------------------
# Запросы (тело не читается)
# --------------------------------------------------------------------------

def get_headers_only(session: requests.Session, url: str, referer: str, allow_redirects: bool) -> dict:
    """GET с stream=True; читает только заголовки и сразу закрывает ответ."""
    response = session.get(url, headers={"Referer": referer}, timeout=TIMEOUT, verify=True,
                           stream=True, allow_redirects=allow_redirects)
    try:
        headers = response.headers
        return {
            "url": url,
            "allow_redirects": allow_redirects,
            "status": response.status_code,
            "final_url": response.url,
            "redirects": [f"{h.status_code} {h.url} -> Location: {h.headers.get('Location')}" for h in response.history],
            "location": headers.get("Location"),
            "content_type": headers.get("Content-Type", ""),
            "content_disposition": headers.get("Content-Disposition", ""),
            "content_length": headers.get("Content-Length"),
            "content_encoding": headers.get("Content-Encoding"),
            "referer_sent": response.request.headers.get("Referer"),
            "cookie_names_sent": cookie_names_sent(response),
            "set_cookies": [cookie_line(c) for c in response.cookies],
            "session_cookies": [cookie_line(c) for c in session.cookies],
        }
    finally:
        response.close()


def print_meta(meta: dict):
    print(f"GET {meta['url']} (allow_redirects={meta['allow_redirects']})")
    print(f"HTTP status:          {meta['status']}")
    print(f"Final URL:            {meta['final_url']}")
    print(f"Redirects:            {len(meta['redirects'])}")
    for step in meta["redirects"]:
        print(f"  {step}")
    print(f"Location:             {meta['location'] or 'нет'}")
    print(f"Content-Type:         {meta['content_type'] or 'не указан'}")
    print(f"Content-Disposition:  {meta['content_disposition'] or 'не указан'}")
    print(f"Content-Length:       {meta['content_length'] or 'не указан'}")
    print(f"Content-Encoding:     {meta['content_encoding'] or 'не указан'}")
    print(f"Отправлен Referer:    {meta['referer_sent']}")
    print(f"Имена cookies в запросе: {meta['cookie_names_sent'] or 'нет'}")
    print(f"Cookies, установленные этим ответом: {len(meta['set_cookies'])}")
    for line in meta["set_cookies"]:
        print(f"  {line}")
    print(f"Cookies сессии (всего): {len(meta['session_cookies'])}")
    for line in meta["session_cookies"]:
        print(f"  {line}")


# --------------------------------------------------------------------------
# Классификация и имя файла
# --------------------------------------------------------------------------

def classify(meta: dict) -> str:
    status = meta["status"]
    if 300 <= status < 400:
        return REDIRECT
    if status >= 400 or status < 200:
        return ERROR
    disposition = meta["content_disposition"].lower()
    content_type = meta["content_type"].split(";")[0].strip().lower()
    if "attachment" in disposition:
        return DIRECT_FILE
    if content_type == "text/html" or content_type == "application/xhtml+xml":
        return HTML_PAGE
    if not content_type or content_type.startswith(NON_FILE_TEXT_TYPES):
        return UNKNOWN
    return DIRECT_FILE


def fix_mojibake(name: str) -> str:
    """requests декодирует заголовки как latin-1; если это были UTF-8 байты, восстанавливаем."""
    try:
        return name.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


def filename_from_disposition(disposition: str) -> tuple[str | None, str | None]:
    """(имя, как получено). filename*= приоритетнее filename=."""
    star = re.search(r"filename\*\s*=\s*([^';]*)'[^']*'([^;]+)", disposition, re.I)
    if star:
        return unquote(star.group(2).strip().strip('"'), encoding=star.group(1) or "utf-8"), "filename*="
    plain = re.search(r'filename\s*=\s*(?:"([^"]*)"|([^;]+))', disposition, re.I)
    if plain:
        raw = (plain.group(1) if plain.group(1) is not None else plain.group(2)).strip()
        fixed = fix_mojibake(raw)
        return fixed, "filename=" + (" (перекодировано latin-1 -> utf-8)" if fixed != raw else "")
    return None, None


def normalize(name: str) -> str:
    return " ".join(unicodedata.normalize("NFC", name).split()).casefold()


def find_names_in_list_page(html: str, document_id: str) -> list[str]:
    """Тексты ячеек строки таблицы, где встречается documentId и которые заканчиваются расширением файла."""
    soup = BeautifulSoup(html, "html.parser")
    id_pattern = re.compile(rf"(?<!\d){re.escape(document_id)}(?!\d)")
    names = []
    for tr in soup.find_all("tr"):
        if tr.find("tr") or not id_pattern.search(str(tr)):
            continue
        for cell in tr.find_all(["td", "th"]):
            text = text_of(cell)
            if text and len(text) <= 250 and FILE_NAME_END.search(text) and text not in names:
                names.append(text)
    return names


def compare_filename(header_name: str | None, html_names: list[str], task_name: str | None):
    print(f"Имя файла из Content-Disposition: {header_name!r}")
    print(f"Имя(имена) на listContractDocuments (из HTML): {html_names or 'не найдены в строке таблицы'}")
    if not header_name:
        print("Сравнение невозможно: filename в Content-Disposition нет.")
        return
    if html_names:
        same = any(normalize(header_name) == normalize(n) for n in html_names)
        print(f"Совпадает с именем из HTML (без учёта регистра/пробелов, NFC): {'да' if same else 'НЕТ'}")
    if task_name:
        same = normalize(header_name) == normalize(task_name)
        print(f"Совпадает с именем из текста задачи {task_name!r}: {'да' if same else 'НЕТ'}")


# --------------------------------------------------------------------------
# Один тест
# --------------------------------------------------------------------------

def run_test(session: requests.Session, name: str, param: str, value: str, final_url: str,
             task_name: str | None) -> dict:
    prepare_url = DOWNLOAD_URL.format(param, value)
    print("=" * 78)
    print(name)
    result = {"name": name, "error": None, "result": ERROR, "final_url": final_url,
              "filename": None, "cookie_names": [], "referer": prepare_url, "login_hint": False}

    section("Шаг 1. listContractDocuments.do")
    page = choose_list_page(session, param, value)
    print(f"URL: {page['url']}")
    print(f"Выбор страницы: {page['reason']}")
    if page["error"]:
        result["error"] = f"исходная страница не открылась: {page['error']}"
        print(f"ОШИБКА: {result['error']}")
        return result
    print(f"HTTP status: {page['status']}")
    print(f"Cookies сессии: {[c.name for c in session.cookies] or 'нет'}")
    html_names = find_names_in_list_page(page["html"], value) if param == "documentId" else []

    section("Шаг 2. prepareAnonymousDownload.do (тело не читается)")
    print(f"URL: {prepare_url}")
    prepare = get_headers_only(session, prepare_url, page["url"], allow_redirects=True)
    print(f"HTTP status: {prepare['status']}; Content-Type: {prepare['content_type'] or 'не указан'}; "
          f"Referer: {prepare['referer_sent']}")
    print(f"Cookies сессии: {[c.name for c in session.cookies] or 'нет'}")

    section("Шаг 3. Конечный endpoint, allow_redirects=False")
    meta = get_headers_only(session, final_url, prepare_url, allow_redirects=False)
    print_meta(meta)
    kind = classify(meta)
    print(f"Результат: {kind}")

    if kind == REDIRECT:
        location = urljoin(final_url, meta["location"]) if meta["location"] else None
        result["login_hint"] = bool(location and LOGIN_HINT.search(location))
        section("Шаг 4. GET на Location, allow_redirects=True")
        if not location:
            print("В 3xx-ответе нет Location — переход невозможен.")
        else:
            print(f"Location (абсолютный): {location}")
            meta = get_headers_only(session, location, prepare_url, allow_redirects=True)
            print_meta(meta)
            kind = classify(meta)
            print(f"Результат после редиректа: {kind}")
    result["login_hint"] = result["login_hint"] or bool(LOGIN_HINT.search(meta["final_url"]))

    result.update(result=kind, status=meta["status"], content_type=meta["content_type"],
                  content_disposition=meta["content_disposition"], content_length=meta["content_length"],
                  content_encoding=meta["content_encoding"], cookie_names=meta["cookie_names_sent"],
                  referer=meta["referer_sent"], reached_url=meta["final_url"])

    section("Итог теста")
    if kind == DIRECT_FILE:
        print("Подтверждено: endpoint отдаёт файл напрямую (тело не читалось и не сохранялось).")
        filename, how = filename_from_disposition(meta["content_disposition"])
        result["filename"] = filename
        if filename:
            print(f"filename получен через {how}")
            compare_filename(filename, html_names, task_name)
        else:
            print("Content-Disposition не содержит filename — имя файла сервер не сообщает.")
            if html_names:
                print(f"Имя на listContractDocuments (из HTML): {html_names}")
    else:
        print(f"Endpoint НЕ подтверждён как прямая отдача файла: результат {kind}.")
        if kind == HTML_PAGE:
            print("Вернулась HTML-страница; тело не читалось, поэтому причина (login/captcha/ошибка/промежуточная страница) неизвестна.")
    return result


# --------------------------------------------------------------------------
# Вывод
# --------------------------------------------------------------------------

def print_conclusion(results: list[dict]):
    print("=" * 78)
    print("ТЕХНИЧЕСКИЙ ВЫВОД")
    print("Основан только на HTTP-метаданных выполненных GET; тела ответов не читались.")
    print()
    for r in results:
        line = f"{r['name']}: {r['result']}"
        if r["error"]:
            line += f" ({r['error']})"
        elif r["result"] == DIRECT_FILE:
            line += f" | {r['content_type']} | filename={r['filename']!r} | Content-Length={r['content_length'] or 'не указан'}"
        else:
            line += f" | status={r.get('status')} | {r.get('reached_url')}"
        print(line)
    print()

    docs = [r for r in results if "documentId" in r["name"]]
    zips = [r for r in results if "ZIP" in r["name"]]

    def verdict(group: list[dict]) -> str:
        if not group:
            return "не проверялось"
        good = [r for r in group if r["result"] == DIRECT_FILE]
        if len(good) == len(group):
            return f"да, подтверждено ({len(good)} из {len(group)}: direct_file)"
        return f"НЕ подтверждено: direct_file {len(good)} из {len(group)} (остальное: {[r['result'] for r in group if r not in good]})"

    print(f"1. Отдельный документ обычным requests.Session: {verdict(docs)}")
    print(f"2. ZIP-комплект обычным requests.Session: {verdict(zips)}")

    logins = [r["name"] for r in results if r["login_hint"]]
    if logins:
        print(f"3. Авторизация: в URL редиректа/итоговом URL есть признаки login для: {logins}")
    else:
        print("3. Авторизация: логин/пароль не передавались; признаков перехода на страницу login в URL нет. "
              "Для тестов с direct_file авторизация не потребовалась.")

    html_tests = [r["name"] for r in results if r["result"] == HTML_PAGE]
    if html_tests:
        print(f"4. CAPTCHA: для {html_tests} вернулся HTML, тело не читалось — причину определить нельзя.")
    else:
        print("4. CAPTCHA: при direct_file-ответах не понадобилась (файл получен без неё); "
              "для остальных тестов HTML-страниц с неизвестной причиной не было.")

    all_direct = all(r["result"] == DIRECT_FILE for r in results)
    print(f"5. Браузер: {'не нужен для проверенных запросов — только GET из requests.Session' if all_direct else 'по результатам нельзя утверждать, что не нужен: не все endpoint-ы вернули файл'}.")

    cookies = sorted({n for r in results for n in r["cookie_names"]})
    print("6. Данные, с которыми запросы выполнялись:")
    print("   documentId — для отдельного документа; resourceId — для ZIP-комплекта;")
    print(f"   cookies сессии {cookies or 'нет'}; Referer = страница prepareAnonymousDownload.do.")
    print("   Обязательность cookies и Referer скриптом НЕ проверялась (контрольных запросов без них нет) — "
          "минимально необходимое подтверждено только для набора выше.")

    ready = all_direct and not logins
    print("7. Переход к production ARMEPS downloader: "
          + ("можно — все три endpoint-а отдали файл напрямую при цепочке list -> prepare -> download."
             if ready else "рано — не все endpoint-ы подтверждены как direct_file."))
    if ready:
        print("   Учесть: не проверено поведение без cookies/Referer, повторные запросы, лимиты и размер файлов; "
              "содержимое файлов и целостность не проверялись.")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    session = requests.Session()
    session.headers.update(HEADERS)

    results = []
    for name, param, value, final_url, task_name in TESTS:
        try:
            results.append(run_test(session, name, param, value, final_url, task_name))
        except requests.exceptions.SSLError as e:
            print(f"ОШИБКА TLS (проверка сертификата НЕ отключалась): {e}")
            results.append({"name": name, "error": f"TLS: {e}", "result": ERROR, "cookie_names": [],
                            "login_hint": False, "filename": None})
        except Exception as e:  # ошибка одного теста не останавливает остальные
            print(f"ОШИБКА: {type(e).__name__}: {e}")
            results.append({"name": name, "error": f"{type(e).__name__}: {e}", "result": ERROR,
                            "cookie_names": [], "login_hint": False, "filename": None})

    print_conclusion(results)


if __name__ == "__main__":
    main()
