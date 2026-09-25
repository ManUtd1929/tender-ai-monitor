"""
Контролируемый первый backfill enrichment.

Обрабатывает максимум 3 pending-объявления — по одному каждого типа из
SAMPLE_RESOURCE_TYPES. Все pending за один запуск не берутся. Собственного HTTP
и SQL здесь нет: чтение pending — enrichment_repository, обработка —
enrichment_pipeline (direct_file при этом не скачивается — это решает
resource_enrichment).

Запуск из корня проекта:
    python -m src.backfill_enrichment
"""

import logging
import sys

from src.database import enrichment_repository
from src.enrichment_pipeline import process_announcements
from src.scraper.gnumner import configure_tls

logger = logging.getLogger(__name__)

SAMPLE_RESOURCE_TYPES = (
    "eauction_tender_page",
    "armeps_documents_page",
    "direct_file",
)


def select_sample_announcements(
    announcements: list[dict],
    resource_types=SAMPLE_RESOURCE_TYPES,
) -> list[dict]:
    """
    По одному первому объявлению каждого resource_type, в порядке resource_types.
    Если типа нет в списке — он просто пропускается. Исходный список не изменяется.
    """
    selected = []
    for resource_type in resource_types:
        for announcement in announcements:
            if announcement.get("resource_type") == resource_type:
                selected.append(announcement)
                break
    return selected


def run_sample_backfill(db_path=None) -> dict:
    """Обрабатывает выборку из максимум одного pending-объявления каждого типа."""
    configure_tls()
    enrichment_repository.init_db(db_path=db_path)

    pending_count_before = enrichment_repository.count_announcements_without_enrichment(db_path=db_path)
    pending = enrichment_repository.get_announcements_without_enrichment(db_path=db_path)
    selected = select_sample_announcements(pending)

    logger.info(
        "Backfill: pending до запуска %d, выбрано %d из %d",
        pending_count_before, len(selected), len(pending),
    )

    # process_announcements сам вызывает init_db повторно — он идемпотентен.
    pipeline_result = process_announcements(selected, db_path=db_path)

    pending_count_after = enrichment_repository.count_announcements_without_enrichment(db_path=db_path)

    return {
        "pending_count_before": pending_count_before,
        "selected_count": len(selected),
        "selected": [
            {
                "resource_url": announcement.get("resource_url"),
                "resource_type": announcement.get("resource_type"),
                "title": announcement.get("title"),
            }
            for announcement in selected
        ],
        "pipeline_result": pipeline_result,
        "pending_count_after": pending_count_after,
    }


def _print_summary(result: dict) -> None:
    pipeline_result = result["pipeline_result"]

    print()
    print(f"Pending до запуска: {result['pending_count_before']}")
    print()
    print(f"Выбрано: {result['selected_count']}")

    for number, item in enumerate(result["selected"], start=1):
        print()
        print(f"[{number}]")
        print(f"Тип: {item['resource_type']}")
        print(f"Название: {item['title']}")
        print(f"URL: {item['resource_url']}")

    print()
    print("Результат:")
    print(f"Успешно: {pipeline_result['success_count']}")
    print(f"Ошибок: {pipeline_result['failed_count']}")
    print(f"Новых enrichment: {pipeline_result['storage_new_count']}")
    print(f"Обновлённых enrichment: {pipeline_result['storage_updated_count']}")
    print(f"Существующих enrichment: {pipeline_result['storage_existing_count']}")

    for failure in pipeline_result["failures"]:
        print()
        print(f"URL: {failure['resource_url']}")
        print(f"Тип ошибки: {failure['error_type']}")
        print(f"Сообщение: {failure['error_message']}")

    print()
    print(f"Pending после запуска: {result['pending_count_after']}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    _print_summary(run_sample_backfill())


if __name__ == "__main__":
    main()
