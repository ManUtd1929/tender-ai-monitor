"""
Координирующий модуль: связывает scraper (gnumner) с SQLite-хранилищем
(tender_repository).

Собственной логики парсинга, HTTP, SQL или TLS здесь нет — только вызов
уже готовых функций в нужном порядке.
"""

import logging
import sys

from src.database.tender_repository import count_tenders, init_db, save_tenders
from src.scraper.gnumner import configure_tls, fetch_tenders

logger = logging.getLogger(__name__)


def run_monitor(page: int = 1) -> dict:
    """
    Один проход мониторинга: получить тендеры со страницы, сохранить в базу.

    Ошибки сети и базы намеренно не перехватываются — они поднимаются
    вызывающему коду. (Ошибки по отдельным тендерам внутри save_tenders
    логируются и не прерывают сохранение остальных.)
    """
    logger.info("Запуск мониторинга, страница %d", page)

    configure_tls()
    init_db()

    tenders = fetch_tenders(page)
    logger.info("Получено тендеров с сайта: %d", len(tenders))

    new_tenders = save_tenders(tenders)
    total_count = count_tenders()

    logger.info(
        "Мониторинг завершён: получено %d, новых %d, всего в базе %d",
        len(tenders), len(new_tenders), total_count,
    )

    return {
        "fetched_count": len(tenders),
        "new_count": len(new_tenders),
        "total_count": total_count,
        "new_tenders": new_tenders,
    }


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    result = run_monitor(page=1)

    print(f"Получено с сайта: {result['fetched_count']}")
    print(f"Новых тендеров: {result['new_count']}")
    print(f"Всего в базе: {result['total_count']}")

    if not result["new_tenders"]:
        print("Новых тендеров нет.")
        return

    print()
    for i, tender in enumerate(result["new_tenders"], start=1):
        print(f"[{i}] {tender['published_at']} | {tender['file_type']} | {tender['title']}")
        print(f"    {tender['attachment_url']}")


if __name__ == "__main__":
    main()
