"""
Первый этап извлечения данных: разбор уже сохранённого HTML страницы
списка объявлений (data/gnumner_main.html).

Никаких сетевых запросов здесь нет — только парсинг локального файла,
сохранённого ранее с помощью site_probe.py.
"""

import os
import re

from bs4 import BeautifulSoup

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
HTML_FILE = os.path.join(DATA_DIR, "gnumner_main.html")

PUBLISHED_AT_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")


def read_html_file(path: str = HTML_FILE) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse_tenders(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    tenders = []

    for tender_div in soup.find_all("div", class_="tender"):
        link = tender_div.select_one("div.tender_title a")
        if link is None:
            # Блок без ссылки не является объявлением, которое можно обработать.
            continue

        title = link.get_text(strip=True)
        attachment_url = link.get("href", "")
        filename = attachment_url.rsplit("/", 1)[-1]
        file_type = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        time_tag = tender_div.find("p", class_="tender_time")
        tender_time_raw = time_tag.get_text() if time_tag is not None else ""

        match = PUBLISHED_AT_PATTERN.search(tender_time_raw)
        published_at = match.group(0) if match else None

        tenders.append({
            "title": title,
            "attachment_url": attachment_url,
            "filename": filename,
            "file_type": file_type,
            "published_at": published_at,
            "tender_time_raw": tender_time_raw,
        })

    return tenders


def main():
    html = read_html_file()
    tenders = parse_tenders(html)

    print(f"Найдено тендеров: {len(tenders)}")
    print()

    for i, tender in enumerate(tenders, start=1):
        print(f"[{i}] {tender['title']}")
        print(f"    filename:       {tender['filename']}")
        print(f"    file_type:      {tender['file_type']}")
        print(f"    published_at:   {tender['published_at']}")
        print(f"    attachment_url: {tender['attachment_url']}")
        print()


if __name__ == "__main__":
    main()
