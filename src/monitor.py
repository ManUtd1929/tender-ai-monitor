"""
Координирующий модуль: связывает production-сборщик разделов Gnumner
(sections) с SQLite-хранилищем объявлений (announcement_repository).

Собственной логики парсинга, HTTP, SQL здесь нет — только вызов уже готовых
функций в нужном порядке и подсчёт сводки.

Прототип общей страницы (gnumner.fetch_tenders, tender_repository и таблица
tenders) остаётся в проекте как legacy и этим модулем не используется.
"""

import logging
import sys

from src.database.announcement_repository import (
    count_announcements,
    init_db,
    save_announcements,
)
from src.scraper.gnumner import configure_tls
from src.scraper.sections import fetch_all_sections

logger = logging.getLogger(__name__)


def run_monitor(page: int = 1) -> dict:
    """
    Один проход мониторинга: получить объявления всех разделов, сохранить в базу.

    Ошибки сети и базы намеренно не перехватываются — они поднимаются
    вызывающему коду. (Изоляция ошибок отдельных разделов реализована внутри
    fetch_all_sections, отдельных записей — внутри save_announcements.)

    Дубликаты resource_url внутри одного запуска НЕ удаляются до сохранения:
    repository сам определяет new / existing / updated. Здесь они только
    подсчитываются для сводки.
    """
    logger.info("Запуск мониторинга, страница %d", page)

    configure_tls()  # fetch_all_sections сам TLS не настраивает
    init_db()

    announcements = fetch_all_sections(page=page)

    fetched_count = len(announcements)
    unique_resource_count = len({a["resource_url"] for a in announcements if a.get("resource_url")})
    duplicate_count = fetched_count - unique_resource_count
    logger.info(
        "Получено объявлений: %d, уникальных resource_url: %d, дубликатов: %d",
        fetched_count, unique_resource_count, duplicate_count,
    )

    save_result = save_announcements(announcements)
    total_count = count_announcements()

    logger.info(
        "Мониторинг завершён: новых %d, обновлённых %d, существующих %d, ошибок %d, всего в базе %d",
        save_result["new_count"], save_result["updated_count"],
        save_result["existing_count"], save_result["failed_count"], total_count,
    )

    return {
        "fetched_count": fetched_count,
        "unique_resource_count": unique_resource_count,
        "duplicate_count": duplicate_count,
        "new_count": save_result["new_count"],
        "updated_count": save_result["updated_count"],
        "existing_count": save_result["existing_count"],
        "failed_count": save_result["failed_count"],
        "total_count": total_count,
        "new_announcements": save_result["new_announcements"],
        "updated_announcements": save_result["updated_announcements"],
    }


def _print_announcements(title: str, announcements: list[dict]):
    print(title)
    for i, announcement in enumerate(announcements, start=1):
        print()
        print(f"[{i}]")
        print(f"Раздел: {announcement['source_section_name']}")
        print(f"Название: {announcement['title']}")
        print(f"Опубликовано: {announcement.get('published_at')}")
        print(f"Дедлайн: {announcement.get('deadline_at')}")
        print(f"Тип ресурса: {announcement['resource_type']}")
        print(f"URL: {announcement['resource_url']}")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    result = run_monitor(page=1)

    print()
    print("Мониторинг завершён.")
    print()
    print(f"Получено объявлений: {result['fetched_count']}")
    print(f"Уникальных resource_url: {result['unique_resource_count']}")
    print(f"Дубликатов в текущем запуске: {result['duplicate_count']}")
    print()
    print(f"Новых: {result['new_count']}")
    print(f"Обновлённых: {result['updated_count']}")
    print(f"Уже существующих: {result['existing_count']}")
    print(f"Ошибок сохранения: {result['failed_count']}")
    print()
    print(f"Всего в announcements: {result['total_count']}")

    print()
    if result["new_announcements"]:
        _print_announcements("Новые объявления:", result["new_announcements"])
    else:
        print("Новых объявлений нет.")

    if result["updated_announcements"]:
        print()
        _print_announcements("Изменённые объявления:", result["updated_announcements"])


if __name__ == "__main__":
    main()
