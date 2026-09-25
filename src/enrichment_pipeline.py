"""
Оркестрация обогащения: enrich_announcement (resource_enrichment) ->
save_enrichment (enrichment_repository).

Модуль получает готовый список объявлений и обрабатывает их все. Определять,
какие объявления новые или изменённые, — задача monitor и announcement_repository;
здесь повторной дедупликации нет. Собственного HTTP и SQL тоже нет.

Ошибка обогащения или сохранения одного объявления не останавливает остальные.
"""

import logging

from src.database import enrichment_repository
from src.scraper.resource_enrichment import enrich_announcement

logger = logging.getLogger(__name__)


def process_announcement(announcement: dict, db_path=None) -> dict:
    """
    Обогащает одно объявление и сохраняет enrichment. Исходный announcement
    не изменяется. Ошибки enrich/save не перехватываются — их обрабатывает
    process_announcements.
    """
    enrichment = enrich_announcement(announcement)
    resource_url = announcement.get("resource_url")

    storage_status = enrichment_repository.save_enrichment(
        resource_url, enrichment, db_path=db_path
    )
    logger.info(
        "Объявление обработано: %s, enrichment_status=%s, storage_status=%s",
        resource_url, enrichment.get("enrichment_status"), storage_status,
    )

    return {
        "resource_url": resource_url,
        "enrichment_status": enrichment.get("enrichment_status"),
        "storage_status": storage_status,
        "enrichment": enrichment,
    }


def process_announcements(announcements: list[dict], db_path=None) -> dict:
    """Последовательно обрабатывает объявления; ошибки отдельных объявлений собираются в failures."""
    logger.info("Запуск обогащения: объявлений %d", len(announcements))

    enrichment_repository.init_db(db_path=db_path)

    results = []
    failures = []
    storage_counts = {"new": 0, "updated": 0, "existing": 0}

    for announcement in announcements:
        try:
            result = process_announcement(announcement, db_path=db_path)
        except Exception as error:
            resource_url = announcement.get("resource_url") if isinstance(announcement, dict) else None
            logger.exception("Не удалось обработать объявление: %s", resource_url)
            failures.append({
                "resource_url": resource_url,
                "error_type": type(error).__name__,
                "error_message": str(error),
            })
            continue

        results.append(result)
        storage_counts[result["storage_status"]] = storage_counts.get(result["storage_status"], 0) + 1

    logger.info(
        "Обогащение завершено: всего %d, успешно %d, ошибок %d "
        "(storage: новых %d, обновлённых %d, без изменений %d)",
        len(announcements), len(results), len(failures),
        storage_counts["new"], storage_counts["updated"], storage_counts["existing"],
    )

    return {
        "processed_count": len(announcements),
        "success_count": len(results),
        "failed_count": len(failures),
        "storage_new_count": storage_counts["new"],
        "storage_updated_count": storage_counts["updated"],
        "storage_existing_count": storage_counts["existing"],
        "results": results,
        "failures": failures,
    }


if __name__ == "__main__":
    print("Use process_announcements() from the monitoring pipeline.")
