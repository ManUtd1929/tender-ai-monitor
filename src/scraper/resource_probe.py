"""
Диагностический скрипт: исследование целевых ресурсов объявлений (resource_url).

По одному реальному примеру каждого типа ссылки:
  A. электронный аукцион  -> страница тендера на eauction.armeps.am
  B. открытый конкурс     -> страница документов на armeps.am
  C. прямой DOCX          -> файл на gnumner.minfin.am

Ничего не сохраняет (ни в базу, ни в файлы) и не скачивает документы.
Каждый URL запрашивается одним GET с stream=True: заголовки читаются всегда,
тело — только если Content-Type говорит, что это HTML. Файл C после получения
заголовков закрывается без чтения тела.

Проверка TLS-сертификата (verify=True) нигде не отключается. Для внешних
доменов используется обычное хранилище certifi; системное хранилище ОС
(truststore, configure_tls) подключается только перед запросом к gnumner.
При TLS-ошибке скрипт только сообщает о ней.

Поиск полей на странице тендера — по подсказкам-ключевым словам в тексте
меток, а не по CSS-селекторам. Список слов (в том числе армянских основ)
не исчерпывающий: для каждого найденного поля печатается реальный HTML-фрагмент,
чтобы вывод можно было проверить глазами.
"""

import re
import sys
from urllib.parse import unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from src.scraper.gnumner import HEADERS, TIMEOUT, configure_tls

TARGETS = [
    (
        "auction",
        "A. Электронный аукцион",
        "https://eauction.armeps.am/hy/public/tender_details/tmid/d903e27c-3159-45ae-be43-6f33a82068ba",
    ),
    (
        "tender_docs",
        "B. Открытый конкурс",
        "https://armeps.am/epps/cft/listContractDocuments.do?resourceId=12486673",
    ),
    (
        "direct",
        "C. Прямой DOCX с Gnumner",
        "https://gnumner.minfin.am/website/images/original/36f98164.docx",
    ),
]

GNUMNER_HOST = "gnumner.minfin.am"
MAX_HTML_BYTES = 5 * 1024 * 1024
FRAGMENT_LIMIT = 300
VALUE_LIMIT = 200
MAX_MATCHES = 3
MAX_OTHER_LINKS = 30
CORE_FIELDS_FOR_TENDER_PAGE = 3  # столько «доменных» полей → html_tender_page

# key, название, подсказки (нижний регистр)
FIELDS = [
    ("title", "Название тендера", ["անվանում", "наименование", "название", "title", "name"]),
    ("procedure_code", "Номер/код процедуры", ["ծածկագիր", "համար", "код", "номер", "code", "number"]),
    ("customer", "Заказчик", ["պատվիրատու", "заказчик", "customer", "buyer", "procuring", "contracting"]),
    ("published", "Дата публикации", ["հրապարակ", "опубликов", "дата публикации", "publish"]),
    ("deadline", "Срок подачи", ["վերջնաժամկետ", "ժամկետ", "срок", "deadline", "closing", "submission"]),
    ("cpv", "CPV / код предмета закупки", ["cpv", "код предмета"]),
    ("description", "Описание / предмет закупки", ["նկարագր", "առարկա", "описание", "предмет", "description", "subject"]),
    ("budget", "Стоимость / бюджет", ["արժեք", "գին", "բյուջե", "стоимость", "бюджет", "цена", "сумма", "budget", "price", "amount", "cost"]),
    ("status", "Статус процедуры", ["կարգավիճակ", "վիճակ", "статус", "status"]),
]
CORE_FIELD_KEYS = {"procedure_code", "customer", "published", "deadline", "cpv", "budget"}

DOC_EXTENSIONS = {"pdf", "doc", "docx", "xls", "xlsx", "zip", "rar", "7z", "odt", "ods", "rtf", "txt", "csv"}
DOWNLOAD_HINTS = ("download", "attachment", "getfile", "fileid", "documentid")
TEXTUAL_TYPES = {"application/json", "application/xml", "application/javascript"}


# --------------------------------------------------------------------------
# Вспомогательные функции
# --------------------------------------------------------------------------

def clean_text(element) -> str:
    return " ".join(element.get_text(" ", strip=True).split())


def shorten(text: str, limit: int = FRAGMENT_LIMIT) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


def content_type_main(content_type: str) -> str:
    return content_type.split(";")[0].strip().lower()


def is_html_type(content_type: str) -> bool:
    return "html" in content_type_main(content_type)


def is_textual_type(content_type: str) -> bool:
    main = content_type_main(content_type)
    return main.startswith("text/") or main in TEXTUAL_TYPES or "html" in main


# --------------------------------------------------------------------------
# Сетевой запрос
# --------------------------------------------------------------------------

def read_capped(response) -> tuple[bytes, bool]:
    chunks, size = [], 0
    for chunk in response.iter_content(65536):
        chunks.append(chunk)
        size += len(chunk)
        if size >= MAX_HTML_BYTES:
            return b"".join(chunks), True
    return b"".join(chunks), False


def fetch(url: str) -> tuple[dict, bytes | None]:
    """Один GET. Тело читается только для HTML, иначе соединение просто закрывается."""
    response = requests.get(url, headers=HEADERS, timeout=TIMEOUT, verify=True, stream=True)
    try:
        meta = {
            "final_url": response.url,
            "redirects": [f"{h.status_code} {h.url}" for h in response.history],
            "status": response.status_code,
            "content_type": response.headers.get("Content-Type", ""),
            "content_disposition": response.headers.get("Content-Disposition", ""),
            "content_length": response.headers.get("Content-Length"),
            "body_truncated": False,
        }
        body = None
        if is_html_type(meta["content_type"]):
            body, meta["body_truncated"] = read_capped(response)
        return meta, body
    finally:
        response.close()


# --------------------------------------------------------------------------
# Анализ HTML
# --------------------------------------------------------------------------

def context_fragment(node) -> str:
    """HTML-фрагмент вокруг текстового узла: родитель, при малом размере — на 1–2 уровня выше."""
    element, hops = node.parent, 0
    while (
        element.parent is not None
        and element.parent.name not in ("body", "html", "[document]")
        and len(str(element)) < 120
        and hops < 2
    ):
        element, hops = element.parent, hops + 1
    return shorten(str(element))


def make_pair(via: str, label_el, value_el, container) -> dict:
    return {
        "via": via,
        "label": clean_text(label_el),
        "value": clean_text(value_el),
        "fragment": shorten(str(container)),
    }


def extract_pairs(soup: BeautifulSoup) -> list[dict]:
    """Пары «метка → значение» из типовых конструкций: tr из двух ячеек, dt/dd, label + следующий элемент."""
    pairs = []
    for tr in soup.find_all("tr"):
        cells = tr.find_all(["th", "td"], recursive=False)
        if len(cells) == 2:
            pairs.append(make_pair("tr", cells[0], cells[1], tr))
    for dt in soup.find_all("dt"):
        dd = dt.find_next_sibling("dd")
        if dd is not None:
            pairs.append(make_pair("dt/dd", dt, dd, f"{dt}{dd}"))
    for label in soup.find_all("label"):
        value = label.find_next_sibling()
        if value is not None:
            pairs.append(make_pair("label", label, value, f"{label}{value}"))
    return pairs


def structural_titles(soup: BeautifulSoup) -> list[dict]:
    found = []
    if soup.title and soup.title.get_text(strip=True):
        found.append({"via": "<title>", "label": None, "value": clean_text(soup.title), "fragment": shorten(str(soup.title))})
    h1 = soup.find("h1")
    if h1 is not None:
        found.append({"via": "<h1>", "label": None, "value": clean_text(h1), "fragment": shorten(str(h1))})
    og = soup.find("meta", attrs={"property": "og:title"})
    if og is not None and og.get("content"):
        found.append({"via": "og:title", "label": None, "value": og["content"], "fragment": shorten(str(og))})
    return found


def find_field(key: str, keywords: list[str], soup, pairs, text_nodes) -> list[dict]:
    matches = structural_titles(soup) if key == "title" else []

    for pair in pairs:
        if any(k in pair["label"].lower() for k in keywords):
            matches.append(pair)

    # Резерв: любой короткий текстовый узел с ключевым словом (менее надёжно)
    if not any(m["via"] in ("tr", "dt/dd", "label") for m in matches):
        for text, node in text_nodes:
            if any(k in text.lower() for k in keywords):
                matches.append({"via": "text", "label": text, "value": None, "fragment": context_fragment(node)})

    unique, seen = [], set()
    for m in matches:
        if m["fragment"] not in seen:
            seen.add(m["fragment"])
            unique.append(m)
    return unique[:MAX_MATCHES]


def find_fields(soup: BeautifulSoup) -> dict:
    pairs = extract_pairs(soup)
    text_nodes = [
        (text, node)
        for node in soup.find_all(string=True)
        if node.parent.name not in ("script", "style", "noscript", "title")
        and 0 < len(text := " ".join(node.split())) <= 150
    ]
    return {key: find_field(key, kw, soup, pairs, text_nodes) for key, _, kw in FIELDS}


def guess_extension(href: str, text: str) -> str | None:
    parsed = urlparse(href)
    for candidate in (unquote(parsed.path), unquote(parsed.query), text):
        match = re.search(r"\.([A-Za-z0-9]{2,5})(?=$|[&\s\"'])", candidate)
        if match and match.group(1).lower() in DOC_EXTENSIONS:
            return match.group(1).lower()
    return None


def find_links(soup: BeautifulSoup, base_url: str) -> dict:
    documents, others, script_links = [], [], 0
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#") or href.lower().startswith(("javascript:", "mailto:")):
            script_links += 1
            continue
        absolute = urljoin(base_url, href)
        text = clean_text(a)
        ext = guess_extension(absolute, text)
        if ext:
            reason = "расширение файла в URL/тексте"
        elif any(h in absolute.lower() for h in DOWNLOAD_HINTS):
            reason = "слово-подсказка в URL"
        else:
            others.append((text, absolute))
            continue
        documents.append({"text": text, "href": absolute, "ext": ext, "reason": reason})
    return {"documents": documents, "others": others, "non_navigating": script_links}


def dynamic_hints(soup: BeautifulSoup) -> dict:
    scripts = soup.find_all("script")
    inline = [s.get_text() for s in scripts if not s.get("src")]
    body = soup.body or soup
    return {
        "script_total": len(scripts),
        "external_scripts": [s["src"] for s in scripts if s.get("src")],
        "ajax_calls": any(
            re.search(r"fetch\(|XMLHttpRequest|\$\.ajax|\$\.get\(|\$\.post\(|axios", code) for code in inline
        ),
        "visible_text_len": sum(
            len(t.strip()) for t in body.find_all(string=True) if t.parent.name not in ("script", "style")
        ),
        "forms": len(soup.find_all("form")),
        "iframes": len(soup.find_all("iframe")),
        "html_lang": soup.html.get("lang") if soup.html else None,
    }


# --------------------------------------------------------------------------
# Классификация
# --------------------------------------------------------------------------

def classify(meta: dict, url: str, fields: dict | None, links: dict | None) -> tuple[str, list[str]]:
    """direct_file / html_tender_page / html_documents_page / unknown — только по HTTP и HTML."""
    reasons = []
    disposition = meta["content_disposition"]
    content_type = meta["content_type"]

    if not 200 <= meta["status"] < 300:
        return "unknown", [f"HTTP status {meta['status']}"]

    if "attachment" in disposition.lower() or "filename" in disposition.lower():
        reasons.append(f"Content-Disposition: {disposition}")
    if content_type and not is_textual_type(content_type):
        reasons.append(f"Content-Type {content_type_main(content_type)} — не текстовый/не HTML")
    if reasons:
        ext = guess_extension(meta["final_url"], "")
        if ext:
            reasons.append(f"расширение .{ext} в URL (подтверждающий признак)")
        return "direct_file", reasons

    if is_html_type(content_type) and fields is not None and links is not None:
        core_found = sorted(k for k in CORE_FIELD_KEYS if fields.get(k))
        doc_count = len(links["documents"])
        reasons.append(f"Content-Type {content_type_main(content_type)}")
        reasons.append(f"доменных полей найдено: {len(core_found)} из {len(CORE_FIELD_KEYS)} ({', '.join(core_found) or '—'})")
        reasons.append(f"ссылок на документы в HTML: {doc_count}")
        if len(core_found) >= CORE_FIELDS_FOR_TENDER_PAGE:
            return "html_tender_page", reasons
        if doc_count:
            return "html_documents_page", reasons
        return "unknown", reasons + ["ни доменных полей, ни ссылок на документы"]

    ext = guess_extension(meta["final_url"], "")
    if ext:
        reasons.append(f"есть расширение .{ext} в URL, но заголовки не подтверждают файл — недостаточно")
    return "unknown", reasons or [f"Content-Type: {content_type or 'не указан'}"]


# --------------------------------------------------------------------------
# Один ресурс
# --------------------------------------------------------------------------

def analyze(key: str, name: str, url: str) -> dict:
    result = {"key": key, "name": name, "url": url, "error": None}
    host = urlparse(url).hostname or ""

    if host == GNUMNER_HOST:
        configure_tls()  # системное хранилище ОС; verify=True остаётся включённым
        result["tls"] = "системное хранилище ОС (configure_tls), verify=True"
    else:
        result["tls"] = "стандартная проверка (certifi), verify=True"

    try:
        meta, body = fetch(url)
    except requests.exceptions.SSLError as e:
        result["error"] = f"TLS-ошибка (проверка сертификата НЕ отключалась): {e}"
        return result
    except requests.exceptions.RequestException as e:
        result["error"] = f"{type(e).__name__}: {e}"
        return result

    soup = fields = links = hints = None
    if body is not None:
        soup = BeautifulSoup(body, "html.parser")  # кодировку определяет bs4 (meta/BOM), не requests
        fields = find_fields(soup)
        links = find_links(soup, meta["final_url"])
        hints = dynamic_hints(soup)

    resource_type, reasons = classify(meta, url, fields, links)
    result.update(meta=meta, soup=soup, fields=fields, links=links, hints=hints,
                  resource_type=resource_type, reasons=reasons)
    return result


# --------------------------------------------------------------------------
# Вывод
# --------------------------------------------------------------------------

def print_http(r: dict):
    m = r["meta"]
    print(f"Исходный URL: {r['url']}")
    print(f"Final URL:    {m['final_url']}" + ("" if m["final_url"] != r["url"] else "  (без редиректа)"))
    for step in m["redirects"]:
        print(f"  редирект: {step}")
    print(f"TLS: {r['tls']}")
    print(f"HTTP status: {m['status']}")
    print(f"Content-Type: {m['content_type'] or 'не указан'}")
    print(f"Content-Disposition: {m['content_disposition'] or 'не указан'}")
    length = m["content_length"]
    print(f"Content-Length: {length if length is not None else 'не указан'}")
    if m["body_truncated"]:
        print(f"ВНИМАНИЕ: HTML прочитан не полностью (лимит {MAX_HTML_BYTES} байт)")
    print(f"Тип ресурса: {r['resource_type']}")
    for reason in r["reasons"]:
        print(f"  основание: {reason}")


def print_hints(h: dict):
    print("Признаки динамической загрузки / структуры страницы:")
    print(f"  <html lang>: {h['html_lang'] or 'не указан'}")
    print(f"  видимого текста в <body>: {h['visible_text_len']} символов")
    print(f"  <script>: {h['script_total']} (внешних: {len(h['external_scripts'])}); "
          f"вызовы fetch/XHR/ajax в inline-скриптах: {'да' if h['ajax_calls'] else 'нет'}")
    for src in h["external_scripts"][:10]:
        print(f"    script src: {src}")
    print(f"  <form>: {h['forms']}, <iframe>: {h['iframes']}")


def print_fields(fields: dict):
    print("Поля страницы тендера (по подсказкам-ключевым словам, с HTML-основанием):")
    for key, label, _ in FIELDS:
        matches = fields[key]
        if not matches:
            print(f"  [{label}] — НЕ найдено")
            continue
        print(f"  [{label}] — найдено совпадений: {len(matches)}")
        for m in matches:
            print(f"    via: {m['via']}")
            if m["label"]:
                print(f"    метка: {shorten(m['label'], VALUE_LIMIT)}")
            if m["value"] is not None:
                print(f"    значение: {shorten(m['value'], VALUE_LIMIT) or '(пусто)'}")
            print(f"    HTML: {m['fragment']}")
            print()


def print_links(links: dict, with_others: bool):
    docs = links["documents"]
    print(f"Ссылки на документы в HTML: {len(docs)}")
    for i, d in enumerate(docs, start=1):
        print(f"  [{i}] текст: {shorten(d['text'], VALUE_LIMIT) or '(пусто)'}")
        print(f"      href: {d['href']}")
        print(f"      предполагаемое расширение: {d['ext'] or 'не определено'}  ({d['reason']})")
    print(f"Ссылок-заглушек (#, javascript:, mailto:) — пропущено: {links['non_navigating']}")
    if with_others:
        others = links["others"]
        print(f"Остальные ссылки страницы: {len(others)} (первые {min(MAX_OTHER_LINKS, len(others))}):")
        for text, href in others[:MAX_OTHER_LINKS]:
            print(f"  {shorten(text, 80) or '(пусто)'} -> {href}")


def print_resource(r: dict):
    print("=" * 78)
    print(r["name"])
    if r["error"]:
        print(f"URL: {r['url']}")
        print(f"TLS: {r['tls']}")
        print(f"ОШИБКА: {r['error']}")
        print()
        return

    print_http(r)
    print()

    if r["key"] == "auction" and r["soup"] is not None:
        print_hints(r["hints"])
        print()
        print_fields(r["fields"])
        print_links(r["links"], with_others=True)
    elif r["key"] == "tender_docs" and r["soup"] is not None:
        print(f"<title>: {clean_text(r['soup'].title) if r['soup'].title else 'нет'}")
        h = r["soup"].find(["h1", "h2"])
        print(f"Первый заголовок h1/h2: {clean_text(h) if h else 'нет'}")
        tables = r["soup"].find_all("table")
        print(f"<table> на странице: {len(tables)}")
        print_hints(r["hints"])
        found = [label for key, label, _ in FIELDS if r["fields"][key]]
        print(f"Поля тендера, найденные на этой странице (только названия): {', '.join(found) or 'нет'}")
        print()
        print_links(r["links"], with_others=True)
    elif r["key"] == "direct":
        print("Тело ответа не читалось и не анализировалось (по условию этапа).")
    print()


def print_conclusion(results: dict):
    print("=" * 78)
    print("ТЕХНИЧЕСКИЙ ВЫВОД (только по проверенным примерам)")
    print()

    a, b, c = (results.get(k) for k in ("auction", "tender_docs", "direct"))

    print("1. Данные электронного аукциона:")
    if not a or a["error"]:
        print("   проверить не удалось (см. ошибку выше)")
    elif a["resource_type"] != "html_tender_page":
        print(f"   страница классифицирована как {a['resource_type']}, а не html_tender_page — структура требует ручного изучения")
    else:
        found = [label for key, label, _ in FIELDS if a["fields"][key]]
        missing = [label for key, label, _ in FIELDS if not a["fields"][key]]
        print("   страница отдаётся как HTML; найдены поля: " + (", ".join(found) or "нет"))
        if missing:
            print("   не найдены (по ключевым словам): " + ", ".join(missing))
        print(f"   ссылок на документы в HTML: {len(a['links']['documents'])}")
        h = a["hints"]
        print(f"   видимого текста {h['visible_text_len']} символов, ajax-вызовы в inline-скриптах: {'да' if h['ajax_calls'] else 'нет'}")
        print("   Если нужных полей нет в HTML, они могут подгружаться отдельными запросами — тогда нужен анализ")
        print("   сетевых запросов браузера (этим скриптом не проверялось).")

    print()
    print("2. Документы открытого конкурса:")
    if not b or b["error"]:
        print("   проверить не удалось (см. ошибку выше)")
    else:
        docs = b["links"]["documents"] if b["links"] else []
        if docs:
            exts = sorted({d["ext"] or "?" for d in docs})
            print(f"   в HTML страницы найдено ссылок на документы: {len(docs)} (расширения: {', '.join(exts)})")
            print("   документы можно получить, разобрав HTML страницы и запросив найденные href (в этом скрипте не скачивались)")
        else:
            print(f"   ссылок на документы в HTML не найдено (тип ресурса: {b['resource_type']});")
            print("   возможны динамическая подгрузка, форма или iframe — см. признаки выше")

    print()
    print("3. Как отличать прямой файл от HTML-страницы:")
    print("   по заголовкам ответа: Content-Disposition (attachment/filename) или нетекстовый Content-Type => файл;")
    print("   text/html => HTML-страница. Расширение в URL — только подтверждающий признак.")
    if c and not c["error"]:
        print(f"   в этом запуске C: тип {c['resource_type']}, Content-Type: {c['meta']['content_type'] or 'не указан'}, "
              f"Content-Disposition: {c['meta']['content_disposition'] or 'не указан'}")
    elif c:
        print("   C проверить не удалось (см. ошибку выше)")

    print()
    print("4. Универсальный router по resource_url:")
    types = {r["key"]: r["resource_type"] for r in results.values() if not r["error"]}
    print(f"   получены типы: {types or 'нет данных'}")
    if len(types) == 3 and len(set(types.values())) == 3 and "unknown" not in types.values():
        print("   Три примера дали три разных класса, unknown нет: двухшаговый router возможен:")
        print("   шаг 1 — GET и заголовки (файл или HTML); шаг 2 — для HTML различать страницу тендера и страницу документов.")
    else:
        print("   На этих примерах классы не разделились однозначно (или часть ресурсов не проверена): готового router-а по итогам запуска нет.")
    print("   Ограничения: по одному примеру на тип; ссылки ARMEPS для «запроса котировок» не проверялись;")
    print("   различие tender/documents page основано на эвристике (число найденных полей и ссылок) — её пороги нужно сверить с выводом выше.")
    hosts = ", ".join(f"{urlparse(u).hostname}{urlparse(u).path}" for _, _, u in TARGETS)
    print(f"   Дополнение — грубая маршрутизация по хосту и пути URL (наблюдалось: {hosts}),")
    print("   но надёжна только вместе с проверкой заголовков ответа.")
    print()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    # Порядок важен: внешние домены (A, B) идут до вызова configure_tls(),
    # поэтому проверяются обычным certifi; системное хранилище подключается
    # только перед запросом к gnumner (C).
    results = {}
    for key, name, url in TARGETS:
        try:
            result = analyze(key, name, url)
        except Exception as e:  # ошибка одного ресурса не останавливает остальные
            result = {"key": key, "name": name, "url": url, "tls": "—", "error": f"{type(e).__name__}: {e}"}
        results[key] = result
        print_resource(result)

    print_conclusion(results)


if __name__ == "__main__":
    main()
