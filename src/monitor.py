"""
Координирующий модуль: связывает production-сборщик разделов Gnumner
(sections) с SQLite-хранилищем объявлений (announcement_repository).

Собственной логики парсинга, HTTP, SQL здесь нет — только вызов уже готовых
функций в нужном порядке и подсчёт сводки. Enrichment (enrichment_pipeline)
подключён после сохранения объявлений.

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
from src.database import enrichment_repository
from src.enrichment_pipeline import process_announcements
from src.scraper.gnumner import configure_tls
from src.scraper.sections import fetch_all_sections

logger = logging.getLogger(__name__)

# Сколько старых объявлений без enrichment дообрабатывается за один обычный запуск.
PENDING_ENRICHMENT_RETRY_LIMIT = 3


def merge_enrichment_candidates(
    current_announcements: list[dict],
    pending_announcements: list[dict],
) -> list[dict]:
    """
    Кандидаты на enrichment: сначала объявления текущего запуска, затем pending.
    Дедупликация по непустому resource_url, при дубле остаётся первая запись.
    Записи без resource_url не дедуплицируются. Исходные списки не изменяются.
    """
    candidates = []
    seen_urls = set()
    for announcement in [*current_announcements, *pending_announcements]:
        resource_url = announcement.get("resource_url")
        if resource_url:
            if resource_url in seen_urls:
                continue
            seen_urls.add(resource_url)
        candidates.append(announcement)
    return candidates


def run_monitor(page: int = 1) -> dict:
    """
    Один проход мониторинга: получить объявления всех разделов, сохранить в базу,
    обогатить новые/обновлённые объявления и до PENDING_ENRICHMENT_RETRY_LIMIT
    старых объявлений без enrichment.

    Ошибки сети и базы намеренно не перехватываются — они поднимаются
    вызывающему коду. (Изоляция ошибок отдельных разделов реализована внутри
    fetch_all_sections, отдельных записей — внутри save_announcements,
    отдельных объявлений при enrichment — внутри process_announcements.)

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

    # updated обогащаются всегда: даже при существующем enrichment данные списка
    # (название, дедлайн) могли измениться. new уже pending, но тоже идут сюда —
    # дубли с pending_retry убирает merge_enrichment_candidates.
    current_for_enrichment = save_result["new_announcements"] + save_result["updated_announcements"]

    enrichment_repository.init_db()
    pending_before = enrichment_repository.count_announcements_without_enrichment()
    pending_retry = enrichment_repository.get_announcements_without_enrichment(
        limit=PENDING_ENRICHMENT_RETRY_LIMIT
    )
    enrichment_candidates = merge_enrichment_candidates(current_for_enrichment, pending_retry)
    logger.info(
        "Enrichment: кандидатов %d (текущий запуск %d, pending retry %d), pending до обработки %d",
        len(enrichment_candidates), len(current_for_enrichment), len(pending_retry), pending_before,
    )

    enrichment_result = process_announcements(enrichment_candidates)
    pending_after = enrichment_repository.count_announcements_without_enrichment()

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
        "enrichment_candidate_count": len(enrichment_candidates),
        "pending_enrichment_before": pending_before,
        "pending_retry_selected_count": len(pending_retry),
        "enrichment_result": enrichment_result,
        "pending_enrichment_after": pending_after,
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


def _print_enrichment_summary(result: dict):
    enrichment_result = result["enrichment_result"]

    print()
    print("Обогащение:")
    print()
    print(f"Кандидатов на обработку: {result['enrichment_candidate_count']}")
    print(f"Pending до обработки: {result['pending_enrichment_before']}")
    print(f"Старых pending выбрано для retry: {result['pending_retry_selected_count']}")
    print()
    print(f"Успешно обработано: {enrichment_result['success_count']}")
    print(f"Ошибок enrichment: {enrichment_result['failed_count']}")
    print()
    print(f"Новых enrichment: {enrichment_result['storage_new_count']}")
    print(f"Обновлённых enrichment: {enrichment_result['storage_updated_count']}")
    print(f"Уже существующих enrichment: {enrichment_result['storage_existing_count']}")
    print()
    print(f"Pending после обработки: {result['pending_enrichment_after']}")

    if enrichment_result["failures"]:
        print()
        print("Ошибки enrichment:")
        for i, failure in enumerate(enrichment_result["failures"], start=1):
            print()
            print(f"[{i}]")
            print(f"URL: {failure['resource_url']}")
            print(f"Тип ошибки: {failure['error_type']}")
            print(f"Сообщение: {failure['error_message']}")


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

    _print_enrichment_summary(result)


if __name__ == "__main__":
    main()
