"""
Координирующий модуль: связывает production-сборщик разделов Gnumner
(sections) с SQLite-хранилищем объявлений (announcement_repository).

Собственной логики парсинга, HTTP, SQL здесь нет — только вызов уже готовых
функций в нужном порядке и подсчёт сводки. После сохранения объявлений идут
enrichment (enrichment_pipeline) и обработка документов (document_pipeline).

Прототип общей страницы (gnumner.fetch_tenders, tender_repository и таблица
tenders) остаётся в проекте как legacy и этим модулем не используется.
"""

import logging
import os
import sys
from pathlib import Path

from src.database.announcement_repository import (
    count_announcements,
    init_db,
    save_announcements,
)
from src.database import document_repository, enrichment_repository
from src.document_pipeline import process_enriched_announcements
from src.enrichment_pipeline import process_announcements
from src.scraper.gnumner import configure_tls
from src.scraper.sections import fetch_all_sections

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# AI-анализ выключен по умолчанию: включается только явным AI_ANALYSIS_ENABLED=true.
AI_ANALYSIS_ENV = "AI_ANALYSIS_ENABLED"

# Сколько старых объявлений без enrichment или с неудавшимся refresh дообрабатывается за один запуск.
PENDING_ENRICHMENT_RETRY_LIMIT = 3

# Сколько старых объявлений с незавершённой обработкой документов берётся за один запуск
# (сверх объявлений текущего запуска), чтобы не скачивать весь backlog разом.
PENDING_DOCUMENT_RETRY_LIMIT = 3


def _merge_by_resource_url(current: list[dict], pending: list[dict]) -> list[dict]:
    candidates = []
    seen_urls = set()
    for announcement in [*current, *pending]:
        resource_url = announcement.get("resource_url")
        if resource_url:
            if resource_url in seen_urls:
                continue
            seen_urls.add(resource_url)
        candidates.append(announcement)
    return candidates


def merge_enrichment_candidates(
    current_announcements: list[dict],
    pending_announcements: list[dict],
) -> list[dict]:
    """
    Кандидаты на enrichment: сначала объявления текущего запуска, затем pending.
    Дедупликация по непустому resource_url, при дубле остаётся первая запись.
    Записи без resource_url не дедуплицируются. Исходные списки не изменяются.
    """
    return _merge_by_resource_url(current_announcements, pending_announcements)


def merge_document_candidates(
    current_candidates: list[dict],
    pending_candidates: list[dict],
) -> list[dict]:
    """
    Кандидаты на обработку документов: сначала текущий запуск, затем pending.
    Правила те же, что у merge_enrichment_candidates (первая запись по resource_url
    побеждает, входные списки не изменяются).
    """
    return _merge_by_resource_url(current_candidates, pending_candidates)


def _current_document_urls(enrichment_result: dict, updated_announcements: list[dict]) -> list[str]:
    """
    resource_url объявлений текущего запуска для обработки документов: успешно
    прошедшие enrichment (любой enrichment_status) плюс updated. Объявления с ошибкой
    enrichment исключаются: их enrichment устарел или отсутствует. Порядок сохраняется,
    дубли и пустые значения убираются.
    """
    failed_urls = {failure["resource_url"] for failure in enrichment_result["failures"]}
    urls = []
    for resource_url in (
        [result["resource_url"] for result in enrichment_result["results"]]
        + [announcement.get("resource_url") for announcement in updated_announcements]
    ):
        if resource_url and resource_url not in failed_urls and resource_url not in urls:
            urls.append(resource_url)
    return urls


def _load_document_candidates(resource_urls: list[str]) -> list[dict]:
    """Полные объекты (announcement + enrichment + cpv_codes + documents) для document pipeline."""
    candidates = []
    for resource_url in resource_urls:
        candidate = enrichment_repository.get_announcement_with_enrichment(resource_url)
        if candidate is None:
            logger.warning("Нет announcement/enrichment для обработки документов: %s", resource_url)
            continue
        candidates.append(candidate)
    return candidates


def ai_analysis_enabled() -> bool:
    return os.environ.get(AI_ANALYSIS_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _run_ai_analysis_if_enabled() -> dict | None:
    """
    AI-анализ после enrichment/документов: один batch не более AI_ANALYSIS_BATCH_LIMIT тендеров.
    Выключен (None, никаких импортов analyzers и client) пока AI_ANALYSIS_ENABLED не true. Ошибка AI-стадии не прерывает мониторинг.
    """
    if not ai_analysis_enabled():
        logger.info("AI-анализ выключен (%s не включён)", AI_ANALYSIS_ENV)
        return None
    from src.ai import analysis_pipeline  # ленивый импорт: при выключенном флаге AI-код не загружается

    try:
        return analysis_pipeline.run_monitor_ai_batch()  # батч с AI_ANALYSIS_BATCH_LIMIT, не unlimited
    except Exception as error:
        logger.exception("AI-анализ завершился ошибкой; мониторинг продолжается")
        return {"status": "error", "error_type": type(error).__name__, "error_message": str(error)}


def _run_telegram_if_enabled() -> dict | None:
    """
    Telegram-доставка готовых deep results после AI-стадии (только чтение SQLite, без AI). Выключена
    (None) пока TELEGRAM_NOTIFICATIONS_ENABLED не true. Ошибка стадии не прерывает мониторинг.
    """
    from src.telegram import delivery

    try:
        return delivery.run_delivery_from_env()
    except Exception as error:
        logger.error("Telegram-доставка завершилась ошибкой (%s); мониторинг продолжается", type(error).__name__)
        return {"status": "error", "error_type": type(error).__name__}


def run_monitor(page: int = 1) -> dict:
    """
    Один проход мониторинга: получить объявления всех разделов, сохранить в базу,
    обогатить новые/обновлённые объявления и до PENDING_ENRICHMENT_RETRY_LIMIT
    старых объявлений без enrichment или с неудавшимся refresh, затем обработать документы объявлений текущего
    запуска (успешный enrichment + updated) и до PENDING_DOCUMENT_RETRY_LIMIT старых
    объявлений с незавершённой обработкой документов.

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
    # Кандидаты: без enrichment (первыми) и с failed refresh (у updated старый enrichment остаётся).
    pending_before = enrichment_repository.count_enrichment_processing_candidates()
    pending_retry = enrichment_repository.get_enrichment_processing_candidates(
        limit=PENDING_ENRICHMENT_RETRY_LIMIT
    )
    enrichment_candidates = merge_enrichment_candidates(current_for_enrichment, pending_retry)
    logger.info(
        "Enrichment: кандидатов %d (текущий запуск %d, pending retry %d), pending до обработки %d",
        len(enrichment_candidates), len(current_for_enrichment), len(pending_retry), pending_before,
    )

    enrichment_result = process_announcements(enrichment_candidates)
    pending_after = enrichment_repository.count_enrichment_processing_candidates()

    # Документы updated-объявлений обрабатываются заново даже при state success /
    # no_supported_document: document_url и documents могли измениться.
    document_repository.init_db()
    pending_documents_before = document_repository.count_document_processing_candidates()
    current_document_candidates = _load_document_candidates(
        _current_document_urls(enrichment_result, save_result["updated_announcements"])
    )
    pending_document_retry = document_repository.get_document_processing_candidates(
        limit=PENDING_DOCUMENT_RETRY_LIMIT
    )
    document_candidates = merge_document_candidates(
        current_document_candidates, pending_document_retry
    )
    logger.info(
        "Документы: кандидатов %d (текущий запуск %d, pending retry %d), pending до обработки %d",
        len(document_candidates), len(current_document_candidates),
        len(pending_document_retry), pending_documents_before,
    )

    document_result = process_enriched_announcements(document_candidates)
    pending_documents_after = document_repository.count_document_processing_candidates()

    ai_analysis_result = _run_ai_analysis_if_enabled()
    telegram_result = _run_telegram_if_enabled()

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
        "document_candidate_count": len(document_candidates),
        "pending_documents_before": pending_documents_before,
        "pending_document_retry_selected_count": len(pending_document_retry),
        "document_result": document_result,
        "pending_documents_after": pending_documents_after,
        "ai_analysis": ai_analysis_result,
        "telegram_delivery": telegram_result,
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


def _print_document_summary(result: dict):
    document_result = result["document_result"]

    print()
    print("Документы:")
    print()
    print(f"Кандидатов: {result['document_candidate_count']}")
    print(f"Pending до обработки: {result['pending_documents_before']}")
    print(f"Старых pending выбрано: {result['pending_document_retry_selected_count']}")
    print()
    print(f"Успешно обработано: {document_result['success_count']}")
    print(f"Без поддерживаемого документа: {document_result['no_supported_document_count']}")
    print(f"Ошибок: {document_result['failed_count']}")
    print()
    print(f"Новых downloads: {document_result['download_new_count']}")
    print(f"Существующих downloads: {document_result['download_existing_count']}")
    print()
    print(f"Новых extractions: {document_result['extraction_new_count']}")
    print(f"Обновлённых extractions: {document_result['extraction_updated_count']}")
    print(f"Существующих extractions: {document_result['extraction_existing_count']}")
    print()
    print(f"Pending после обработки: {result['pending_documents_after']}")

    if document_result["failures"]:
        print()
        print("Ошибки обработки документов:")
        for i, failure in enumerate(document_result["failures"], start=1):
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

    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")  # не перезаписывает уже заданные переменные окружения

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
    _print_document_summary(result)

    ai_result = result["ai_analysis"]
    if ai_result is not None:
        print()
        print(f"AI-анализ: {ai_result}")

    telegram_result = result["telegram_delivery"]
    if telegram_result is not None:
        print()
        print(f"Telegram: {telegram_result}")


if __name__ == "__main__":
    main()
