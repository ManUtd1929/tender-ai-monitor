"""
Диагностический скрипт: промежуточная страница ARMEPS prepareAnonymousDownload.do.

Для каждого из трёх примеров:
  1. одной общей requests.Session открывается исходная страница
     listContractDocuments.do (получаем JSESSIONID и прочие cookies);
  2. выполняется ровно один GET на prepareAnonymousDownload.do с Referer этой
     страницы, verify=True, User-Agent и timeout.

Скрипт НЕ отправляет POST, НЕ переходит по ссылкам и формам дальше и НЕ
скачивает документы: если ответ — не HTML (или Content-Disposition: attachment),
тело не читается и на диск не сохраняется, печатаются только HTTP-метаданные.
Ничего не пишет в файлы и БД. Внешние JS-файлы не загружаются.

Для documentId-примеров неизвестно, с какой именно страницы listContractDocuments.do
идёт ссылка. Поэтому среди известных страниц (CANDIDATE_RESOURCE_IDS) выбирается
та, в чьём HTML действительно встречается этот documentId; если не найден нигде —
используется первая страница, и об этом печатается предупреждение.
"""

import re
import sys
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from src.scraper.armeps_probe import (
    analyze_forms,
    cookie_line,
    event_attrs,
    inline_scripts,
    section,
    short,
    text_of,
)
from src.scraper.gnumner import HEADERS, TIMEOUT

LIST_URL = "https://armeps.am/epps/cft/listContractDocuments.do?resourceId={}"
DOWNLOAD_URL = "https://armeps.am/epps/cft/prepareAnonymousDownload.do?{}={}"

# (название, параметр, значение)
EXAMPLES = [
    ("1. documentId=12486728", "documentId", "12486728"),
    ("2. documentId=12486849", "documentId", "12486849"),
    ("3. resourceId=12486673", "resourceId", "12486673"),
]
# Страницы listContractDocuments.do, известные из предыдущих исследований.
CANDIDATE_RESOURCE_IDS = ["12486673", "12486627"]

MAX_LINKS = 60
MAX_MENTIONS = 5
MAX_SCRIPTS = 6
SCRIPT_LIMIT = 3000
SNIPPET_CONTEXT = 80

# Слова, упоминания которых нужно найти (по условию задачи) + login для вывода.
KEYWORDS = [
    ("download", re.compile("download", re.I)),
    ("documentId", re.compile("documentId", re.I)),
    ("resourceId", re.compile("resourceId", re.I)),
    ("token", re.compile("token", re.I)),
    ("captcha", re.compile("captcha", re.I)),
    ("agree", re.compile("agree", re.I)),
    ("anonymous", re.compile("anonymous", re.I)),
    ("downloadContractDocument", re.compile("downloadContractDocument", re.I)),
    ("downloadCftResourceItems", re.compile("downloadCftResourceItems", re.I)),
    ("login (доп.)", re.compile(r"log[\s_-]?in|logon", re.I)),
]
DOWNLOAD_JS = re.compile(
    r"download|documentId|resourceId|token|captcha|agree|anonymous|\.submit\s*\(|location", re.I
)
ENDPOINT_PATTERN = re.compile(r"[\w/.\-]*download\w*\.do[^\s\"'<>)]*", re.I)
JS_NAVIGATION = re.compile(r"(?:window\.|document\.)?location(?:\.href)?\s*=|location\.(?:replace|assign)\s*\(|window\.open\s*\(", re.I)
JS_SUBMIT = re.compile(r"\.submit\s*\(", re.I)
META_REFRESH = re.compile(r"^\s*(\d+)\s*;\s*url\s*=\s*(.+)$", re.I)


# --------------------------------------------------------------------------
# Запросы
# --------------------------------------------------------------------------

def cookie_names_sent(response) -> list[str]:
    header = response.request.headers.get("Cookie", "")
    return [part.split("=")[0].strip() for part in header.split(";") if part.strip()]


def open_list_page(session: requests.Session, resource_id: str) -> dict:
    """GET исходной страницы: нужен для cookies и (для documentId) проверки, что id на ней есть."""
    url = LIST_URL.format(resource_id)
    result = {"url": url, "html": None, "error": None}
    try:
        response = session.get(url, timeout=TIMEOUT, verify=True)
        result["status"] = response.status_code
        result["final_url"] = response.url
        result["html"] = response.text
    except requests.exceptions.SSLError as e:
        result["error"] = f"TLS-ошибка (проверка сертификата НЕ отключалась): {e}"
    except requests.exceptions.RequestException as e:
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def choose_list_page(session: requests.Session, param: str, value: str) -> dict:
    """Открывает исходную страницу для примера и возвращает информацию о ней."""
    if param == "resourceId":
        page = open_list_page(session, value)
        page["reason"] = "страница самого resourceId"
        return page

    id_pattern = re.compile(rf"(?<!\d){re.escape(value)}(?!\d)")
    first = None
    for resource_id in CANDIDATE_RESOURCE_IDS:
        page = open_list_page(session, resource_id)
        first = first or page
        if page["html"] and id_pattern.search(page["html"]):
            page["reason"] = f"documentId={value} найден в HTML этой страницы"
            return page
    first["reason"] = (f"ПРЕДУПРЕЖДЕНИЕ: documentId={value} не найден ни на одной из страниц "
                       f"{CANDIDATE_RESOURCE_IDS}; Referer взят с первой — это предположение")
    return first


def probe(session: requests.Session, url: str, referer: str) -> dict:
    """Один GET. Тело читается только для HTML без Content-Disposition: attachment."""
    response = session.get(url, headers={"Referer": referer}, timeout=TIMEOUT, verify=True, stream=True)
    try:
        headers = response.headers
        content_type = headers.get("Content-Type", "")
        disposition = headers.get("Content-Disposition", "")
        info = {
            "status": response.status_code,
            "final_url": response.url,
            "redirects": [f"{h.status_code} {h.url} -> Location: {h.headers.get('Location')}" for h in response.history],
            "content_type": content_type,
            "content_disposition": disposition,
            "content_length": headers.get("Content-Length"),
            "referer_sent": response.request.headers.get("Referer"),
            "user_agent_sent": response.request.headers.get("User-Agent"),
            "cookie_names_sent": cookie_names_sent(response),
            "set_cookies": [cookie_line(c) for c in response.cookies],
            "session_cookies": [cookie_line(c) for c in session.cookies],
            "is_html": "html" in content_type.lower() and "attachment" not in disposition.lower(),
            "soup": None,
            "html": None,
        }
        if info["is_html"]:
            content = response.content
            info["soup"] = BeautifulSoup(content, "html.parser")
            info["html"] = content.decode(info["soup"].original_encoding or "utf-8", errors="replace")
        return info
    finally:
        response.close()


def print_response(info: dict):
    print(f"HTTP status:         {info['status']}")
    print(f"Final URL:           {info['final_url']}")
    print(f"Redirects:           {len(info['redirects'])}")
    for step in info["redirects"]:
        print(f"  {step}")
    print(f"Content-Type:        {info['content_type'] or 'не указан'}")
    print(f"Content-Disposition: {info['content_disposition'] or 'не указан'}")
    print(f"Content-Length:      {info['content_length'] or 'не указан'}")
    print(f"Отправлен Referer:   {info['referer_sent']}")
    print(f"Отправлен User-Agent: {short(info['user_agent_sent'] or '', 60)}")
    print(f"Имена cookies в запросе: {info['cookie_names_sent'] or 'нет'}")
    print(f"Cookies, установленные этим ответом: {len(info['set_cookies'])}")
    for line in info["set_cookies"]:
        print(f"  {line}")
    print(f"Cookies сессии (всего): {len(info['session_cookies'])}")
    for line in info["session_cookies"]:
        print(f"  {line}")


# --------------------------------------------------------------------------
# Анализ HTML
# --------------------------------------------------------------------------

def print_links(soup: BeautifulSoup, base_url: str):
    anchors = soup.find_all("a")
    print(f"Ссылок <a> на странице: {len(anchors)}")
    for a in anchors[:MAX_LINKS]:
        href = a.get("href")
        absolute = urljoin(base_url, href) if href else None
        events = event_attrs(a)
        print(f"  текст={short(text_of(a), 60)!r} href={href!r}"
              + (f" -> {absolute}" if absolute and absolute != href else "")
              + (f" события={events}" if events else ""))
    if len(anchors) > MAX_LINKS:
        print(f"  … ссылок не показано: {len(anchors) - MAX_LINKS}")


def script_spans(html: str) -> list[tuple]:
    return [m.span() for m in re.finditer(r"<script\b.*?</script>", html, re.I | re.S)]


def print_keyword_mentions(html: str) -> dict:
    spans = script_spans(html)
    counts = {}
    for label, pattern in KEYWORDS:
        matches = list(pattern.finditer(html))
        counts[label] = len(matches)
        in_script = sum(1 for m in matches if any(a <= m.start() < b for a, b in spans))
        print(f"  «{label}»: всего {len(matches)} (в <script>: {in_script}, вне: {len(matches) - in_script})")
        for m in matches[:MAX_MENTIONS]:
            fragment = html[max(0, m.start() - SNIPPET_CONTEXT): m.end() + SNIPPET_CONTEXT]
            print(f"      …{short(fragment, 2 * SNIPPET_CONTEXT + 30)}…")
    return counts


def print_download_scripts(soup: BeautifulSoup):
    scripts = inline_scripts(soup)
    related = [s for s in scripts if DOWNLOAD_JS.search(s)]
    print(f"Inline-скриптов: {len(scripts)}; связанных со скачиванием/навигацией: {len(related)}")
    for n, code in enumerate(related[:MAX_SCRIPTS], start=1):
        code = code.strip()
        print(f"--- inline script #{n} (длина {len(code)}) ---")
        print(code[:SCRIPT_LIMIT] + ("\n… (обрезано)" if len(code) > SCRIPT_LIMIT else ""))
    if len(related) > MAX_SCRIPTS:
        print(f"… скриптов не показано: {len(related) - MAX_SCRIPTS}")
    external = [s["src"] for s in soup.find_all("script", src=True)]
    print(f"Внешних JS (не загружались): {len(external)}")
    for src in external:
        print(f"    {src}")


def find_next_step(soup: BeautifulSoup, html: str, forms: list[dict], base_url: str) -> list[str]:
    """Признаки дополнительного шага в статическом HTML (ничего не выполняется)."""
    steps = []

    for form in forms:
        if form["method"] == "post":
            steps.append(f"форма method=POST action={form['action']} (hidden: {form['hidden_names'] or 'нет'}) — нужен POST")
        else:
            steps.append(f"форма method=GET action={form['action']} (hidden: {form['hidden_names'] or 'нет'}) — нужен GET")

    for meta in soup.find_all("meta", attrs={"http-equiv": re.compile("refresh", re.I)}):
        match = META_REFRESH.match(meta.get("content", ""))
        if match:
            steps.append(f"meta refresh через {match.group(1)} с на {urljoin(base_url, match.group(2).strip())} — GET")
        else:
            steps.append(f"meta refresh: {meta.get('content')!r}")

    body_onload = soup.body.get("onload") if soup.body else None
    if body_onload:
        steps.append(f"<body onload={short(body_onload, 200)!r}>")

    scripts_text = "\n".join(inline_scripts(soup))
    for label, pattern in (("JS-навигация (location/window.open)", JS_NAVIGATION), ("JS submit()", JS_SUBMIT)):
        for m in list(pattern.finditer(scripts_text))[:3]:
            snippet = scripts_text[max(0, m.start() - 60): m.end() + 120]
            steps.append(f"{label}: …{short(snippet, 220)}…")
    return steps


def analyze_html(info: dict) -> dict:
    soup, html, final_url = info["soup"], info["html"], info["final_url"]
    facts = {"title": text_of(soup.title) if soup.title else None}

    print(f"<title>: {facts['title'] or 'нет'}")
    for h in soup.find_all(["h1", "h2", "h3"]):
        print(f"  <{h.name}>: {short(text_of(h))}")

    section("Формы (method, action, hidden inputs, submit/button, onclick)")
    facts["forms"] = analyze_forms(soup, final_url)
    passwords = soup.find_all("input", attrs={"type": re.compile("^password$", re.I)})
    print(f"input type=password на странице: {len(passwords)}")
    facts["password_inputs"] = len(passwords)

    section("Все ссылки")
    print_links(soup, final_url)

    section("Упоминания ключевых слов (весь HTML, включая inline JS)")
    facts["keywords"] = print_keyword_mentions(html)

    section("Endpoint-ы вида *download*.do, встречающиеся в HTML")
    endpoints = sorted({m.group(0) for m in ENDPOINT_PATTERN.finditer(html)})
    facts["endpoints"] = endpoints
    print(f"Найдено: {len(endpoints)}")
    for endpoint in endpoints:
        print(f"  {endpoint}")

    section("Inline JavaScript, связанный со скачиванием")
    print_download_scripts(soup)

    section("Признаки дополнительного шага (GET/POST) в статическом HTML")
    facts["next_steps"] = find_next_step(soup, html, facts["forms"], final_url)
    print(f"Найдено: {len(facts['next_steps'])}")
    for step in facts["next_steps"]:
        print(f"  {step}")
    if not facts["next_steps"]:
        print("  Форм, meta refresh, onload и JS-навигации в статическом HTML не найдено.")
    print("  (Ничего из перечисленного не выполнялось.)")
    return facts


# --------------------------------------------------------------------------
# Один пример
# --------------------------------------------------------------------------

def run_example(session: requests.Session, name: str, param: str, value: str) -> dict:
    url = DOWNLOAD_URL.format(param, value)
    print("=" * 78)
    print(name)
    print(f"URL: {url}")
    result = {"name": name, "url": url, "error": None, "kind": None}

    section("Шаг 1. Исходная страница listContractDocuments.do")
    page = choose_list_page(session, param, value)
    print(f"URL: {page['url']}")
    print(f"Выбор Referer: {page['reason']}")
    if page["error"]:
        result["error"] = f"исходная страница не открылась: {page['error']}"
        print(f"ОШИБКА: {result['error']}")
        return result
    print(f"HTTP status: {page['status']}; final URL: {page['final_url']}")
    print(f"Cookies сессии после открытия: {[c.name for c in session.cookies] or 'нет'}")
    result["referer"] = page["url"]

    section("Шаг 2. GET prepareAnonymousDownload.do")
    try:
        info = probe(session, url, page["url"])
    except requests.exceptions.SSLError as e:
        result["error"] = f"TLS-ошибка (проверка сертификата НЕ отключалась): {e}"
    except requests.exceptions.RequestException as e:
        result["error"] = f"{type(e).__name__}: {e}"
    if result["error"]:
        print(f"ОШИБКА: {result['error']}")
        return result

    print_response(info)
    result.update(
        status=info["status"], final_url=info["final_url"], content_type=info["content_type"],
        content_disposition=info["content_disposition"], content_length=info["content_length"],
        cookie_names_sent=info["cookie_names_sent"], redirects=info["redirects"],
    )

    if not info["is_html"]:
        result["kind"] = "file"
        print()
        print("Endpoint сразу возвращает файл (тело не читалось и на диск не сохранялось).")
        return result

    result["kind"] = "html"
    print()
    result.update(analyze_html(info))
    return result


# --------------------------------------------------------------------------
# Вывод
# --------------------------------------------------------------------------

def print_conclusion(results: list[dict]):
    print("=" * 78)
    print("ТЕХНИЧЕСКИЙ ВЫВОД")
    print("Основан только на выполненных GET и статическом HTML; POST и дальнейшие шаги не выполнялись.")
    print()

    ok = [r for r in results if not r["error"]]
    for r in results:
        print(f"{r['name']}:")
        if r["error"]:
            print(f"  не проверен: {r['error']}")
        elif r["kind"] == "file":
            print(f"  сразу файл: {r['content_type']} | {r['content_disposition'] or 'без Content-Disposition'} "
                  f"| Content-Length={r['content_length'] or 'не указан'}")
        else:
            print(f"  HTML-страница (status {r['status']}, title={r['title']!r}), final URL: {r['final_url']}")
    print()

    html_results = [r for r in ok if r["kind"] == "html"]
    file_results = [r for r in ok if r["kind"] == "file"]

    print("1. Что возвращает prepareAnonymousDownload:")
    if not ok:
        print("   ни один запрос не выполнен — вывода нет.")
        return
    print(f"   файл сразу: {len(file_results)} из {len(ok)}; HTML-страница: {len(html_results)} из {len(ok)}.")

    print("2. Нужен ли дополнительный шаг:")
    for r in html_results:
        steps = r["next_steps"]
        print(f"   {r['name']}: " + (f"признаков в HTML: {len(steps)}" if steps else "признаков в статическом HTML нет"))
        for step in steps[:5]:
            print(f"     {short(step, 250)}")
    if file_results and not html_results:
        print("   нет — файл отдаётся сразу.")

    print("3. Какой endpoint реально отдаёт файл:")
    for r in file_results:
        print(f"   {r['name']}: {r['final_url']} (после редиректов: {len(r['redirects'])})")
    for r in html_results:
        found = r["endpoints"]
        print(f"   {r['name']}: prepareAnonymousDownload файл НЕ отдаёт; "
              f"endpoint-ы в HTML (кандидаты, не проверялись): {found or 'не найдены'}")

    print("4. Нужны ли cookies/JSESSIONID и Referer:")
    print(f"   запросы отправлялись с cookies {sorted({n for r in ok for n in r['cookie_names_sent']}) or 'нет'} "
          f"и с Referer=listContractDocuments.do; контрольных запросов без них по условию не делалось,")
    print("   поэтому обязательность cookies/Referer этим скриптом НЕ доказана (только то, что с ними запрос проходит).")

    print("5. Captcha / согласие / login (число упоминаний слова во всём HTML, включая скрипты):")
    for r in html_results:
        kw = r["keywords"]
        print(f"   {r['name']}: captcha={kw['captcha']}, agree={kw['agree']}, "
              f"login={kw['login (доп.)']}, input password={r['password_inputs']}")
    if file_results:
        print("   для ответов-файлов страница не получена — captcha/согласие/login на этом шаге не встретились.")
    print("   Упоминание слова не доказывает наличие требования: смысл проверять по фрагментам выше.")

    print("6. Можно ли воспроизвести обычным requests.Session без браузера:")
    if html_results and any(r["forms"] and any(f["method"] == "post" for f in r["forms"]) for r in html_results):
        print("   есть POST-форма: скорее всего да (сначала GET страницы с cookies, затем POST с hidden-полями),")
    elif html_results and any(r["next_steps"] for r in html_results):
        print("   есть GET-переход: скорее всего да (GET по найденному URL с теми же cookies),")
    elif html_results:
        print("   в статическом HTML нет механизма следующего шага — нужен анализ browser network request,")
    if file_results and not html_results:
        print("   да — достаточно GET с Session и Referer,")
    print("   но это подтверждается только реальным выполнением следующего шага, которое здесь намеренно не делалось.")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    session = requests.Session()
    session.headers.update(HEADERS)

    results = []
    for name, param, value in EXAMPLES:
        try:
            results.append(run_example(session, name, param, value))
        except Exception as e:  # ошибка разбора одного примера не останавливает остальные
            print(f"ОШИБКА разбора: {type(e).__name__}: {e}")
            results.append({"name": name, "url": DOWNLOAD_URL.format(param, value),
                            "error": f"{type(e).__name__}: {e}", "kind": None})

    print_conclusion(results)


if __name__ == "__main__":
    main()
