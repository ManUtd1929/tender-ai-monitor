"""
Локальный backfill extraction скачанных документов: пересчитывает текст уже сохранённых
файлов (document_downloads.saved_path) текущим extractor-ом (DOCX, XLSX, ZIP с вложенными
ZIP) БЕЗ HTTP и без повторного download. Модуль не импортирует downloader и requests.

Кандидаты (document_repository.get_document_extraction_backfill_candidates): download без
extraction, с failed extraction или со skipped member .xlsx/.zip, которые раньше не
поддерживались.

DRY RUN по умолчанию: БД открывается только на чтение, файлы не читаются (проверяется лишь
их существование), extraction не запускается, ничего не записывается. Только --apply:
    1. читает файл saved_path и заново считает sha256; при несовпадении с сохранённым
       sha256 download пропускается как failed (сохранённый sha256 не обновляется никогда);
    2. извлекает текст из уже прочитанных (проверенных) bytes;
    3. replace_extractions_for_download — authoritative snapshot одной транзакцией
       (устаревшие строки, например прежний skipped "lot_1.zip", удаляются);
    4. обновляет document_processing_state — только если этот download является последней
       версией своего (resource_url, source_kind, source_ref) и состояние не относится к
       другому источнику. Историческая версия состояние не меняет.
Ошибка одного download не останавливает остальные.

Запуск из корня проекта:
    python -m src.backfill_document_extractions                # dry run
    python -m src.backfill_document_extractions --apply --limit 5
"""

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path

from src.database import document_repository
from src.parser.document_extractor import (
    extract_docx,
    extract_supported_from_zip,
    extract_xlsx,
)

logger = logging.getLogger(__name__)

ITEM_SUCCESS = "success"
ITEM_FAILED = "failed"
ITEM_NO_SUPPORTED = "no_supported"

# Извлечение по расширению локального файла: extension -> (extraction_type, функция).
EXTRACTORS = {
    ".docx": ("docx", extract_docx),
    ".xlsx": ("xlsx", extract_xlsx),
    ".zip": ("zip", extract_supported_from_zip),
}

STATE_UPDATED = "updated"
STATE_UNCHANGED = "unchanged"
STATE_NOT_LATEST = "not_latest_version"
STATE_OTHER_SOURCE = "other_source"


class DocumentHashMismatchError(ValueError):
    """sha256 локального файла не совпадает с document_downloads.sha256."""


# --------------------------------------------------------------------------
# Local file safety
# --------------------------------------------------------------------------

def read_verified_file(saved_path: str, expected_sha256: str) -> dict:
    """
    Читает файл целиком, заново считает размер и sha256 и сравнивает sha256 с сохранённым.
    Возвращает {"data", "size_bytes", "sha256"}. FileNotFoundError — файла нет, OSError —
    это не обычный файл, DocumentHashMismatchError — sha256 не совпал. Сохранённый sha256
    не изменяется.
    """
    path = Path(saved_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл документа не найден: {saved_path}")
    if not path.is_file():
        raise OSError(f"Путь документа не является обычным файлом: {saved_path}")

    data = path.read_bytes()
    actual_sha256 = hashlib.sha256(data).hexdigest()
    if actual_sha256 != expected_sha256:
        raise DocumentHashMismatchError(
            f"sha256 файла не совпадает с сохранённым: {saved_path} "
            f"(в базе {expected_sha256}, на диске {actual_sha256}, {len(data)} bytes)"
        )
    return {"data": data, "size_bytes": len(data), "sha256": actual_sha256}


def _local_extension(saved_path) -> str:
    return os.path.splitext(str(saved_path or ""))[1].lower()


# --------------------------------------------------------------------------
# Processing state
# --------------------------------------------------------------------------

def update_processing_state(
    candidate: dict, status: str, error_type=None, error_message=None, db_path=None,
) -> str:
    """
    Обновляет document_processing_state объявления по итогам backfill. Возвращает
    STATE_UPDATED, STATE_UNCHANGED, STATE_NOT_LATEST или STATE_OTHER_SOURCE.

    Меняет состояние только последняя версия download для (resource_url, source_kind,
    source_ref); историческая версия его не трогает. Если состояние уже привязано к другому
    источнику (source_kind/source_ref заполнены и отличаются), оно тоже не меняется.
    """
    resource_url = candidate["resource_url"]
    source_kind = candidate["source_kind"]
    source_ref = candidate["source_ref"]

    latest = document_repository.get_latest_download(
        resource_url, source_kind, source_ref, db_path=db_path
    )
    if latest is None or latest["id"] != candidate["id"]:
        return STATE_NOT_LATEST

    current = document_repository.get_processing_state(resource_url, db_path=db_path)
    if current is not None:
        has_source = current["source_kind"] is not None or current["source_ref"] is not None
        if has_source and (current["source_kind"], current["source_ref"]) != (
            source_kind, source_ref
        ):
            return STATE_OTHER_SOURCE
        wanted = (status, source_kind, source_ref, error_type, error_message)
        existing = tuple(
            current[name]
            for name in ("status", "source_kind", "source_ref", "error_type", "error_message")
        )
        if existing == wanted:
            return STATE_UNCHANGED

    document_repository.save_processing_state(
        resource_url, status, source_kind=source_kind, source_ref=source_ref,
        error_type=error_type, error_message=error_message, db_path=db_path,
    )
    return STATE_UPDATED


# --------------------------------------------------------------------------
# One download
# --------------------------------------------------------------------------

def _base_item(candidate: dict) -> dict:
    return {
        "download_id": candidate["id"],
        "resource_url": candidate["resource_url"],
        "filename": candidate["filename"],
        "saved_path": candidate["saved_path"],
        "reasons": candidate.get("reasons", []),
    }


def process_candidate(candidate: dict, db_path=None) -> dict:
    """
    Проверка файла -> extraction -> snapshot -> processing state для одного download.
    Исключения не пробрасываются: результат — dict со status success / failed / no_supported.
    """
    item = _base_item(candidate)
    download_id = candidate["id"]
    processing_status = None
    error = None

    extension = _local_extension(candidate["saved_path"])
    if extension not in EXTRACTORS:
        logger.info(
            "Backfill: неподдерживаемый файл пропущен: download_id=%d %s",
            download_id, candidate["saved_path"],
        )
        return {**item, "status": ITEM_NO_SUPPORTED}

    try:
        verified = read_verified_file(candidate["saved_path"], candidate["sha256"])
        extraction_type, extract = EXTRACTORS[extension]
        result = extract(verified["data"])
        records = document_repository.build_extraction_records(extraction_type, result)
        counts = document_repository.replace_extractions_for_download(
            download_id, records, db_path=db_path
        )
    except Exception as exc:
        logger.exception(
            "Backfill: не удалось обработать документ: download_id=%d %s (%s)",
            download_id, candidate["saved_path"], type(exc).__name__,
        )
        item.update(status=ITEM_FAILED, error_type=type(exc).__name__, error_message=str(exc))
        processing_status = document_repository.PROCESSING_FAILED
        error = exc
    else:
        item.update(
            status=ITEM_SUCCESS, extraction_type=extraction_type, extraction_storage=counts,
        )
        processing_status = document_repository.PROCESSING_SUCCESS
        logger.info(
            "Backfill: документ обработан: download_id=%d %s (%s), member %d",
            download_id, candidate["filename"], extraction_type, len(records),
        )

    try:
        item["processing_state"] = update_processing_state(
            candidate, processing_status,
            error_type=type(error).__name__ if error else None,
            error_message=str(error) if error else None,
            db_path=db_path,
        )
    except Exception as exc:
        logger.exception(
            "Backfill: не удалось обновить состояние обработки: download_id=%d", download_id
        )
        # Extraction уже сохранён; сбой состояния виден как failed, исходная ошибка не теряется.
        if item["status"] != ITEM_FAILED:
            item.update(
                status=ITEM_FAILED, error_type=type(exc).__name__,
                error_message=f"processing state update failed: {exc}",
            )
        item["processing_state"] = "error"
    return item


# --------------------------------------------------------------------------
# Batch
# --------------------------------------------------------------------------

def _dry_run_item(candidate: dict) -> dict:
    return {
        **_base_item(candidate),
        "file_exists": Path(candidate["saved_path"]).is_file(),
        "skipped_supported_count": candidate["skipped_supported_count"],
        "failed_extraction_count": candidate["failed_extraction_count"],
    }


def run_backfill(db_path=None, apply: bool = False, limit: int | None = None) -> dict:
    """
    apply=False — dry run: только чтение БД и проверка существования файлов.
    apply=True — обработка каждого кандидата (см. process_candidate); init_db выполняется
    только здесь (создаёт недостающие колонки метрик в document_extractions).
    """
    logger.info("Backfill extraction: режим %s, limit=%s", "APPLY" if apply else "DRY RUN", limit)

    if apply:
        document_repository.init_db(db_path=db_path)

    candidates = document_repository.get_document_extraction_backfill_candidates(
        db_path=db_path, limit=limit
    )
    logger.info("Backfill extraction: кандидатов %d", len(candidates))

    summary = {
        "mode": "apply" if apply else "dry_run",
        "candidate_count": len(candidates),
        "processed_count": 0,
        "success_count": 0,
        "failed_count": 0,
        "no_supported_count": 0,
        "extractions_new_count": 0,
        "extractions_updated_count": 0,
        "extractions_deleted_count": 0,
        "extractions_existing_count": 0,
        "processing_state_updated_count": 0,
        "failures": [],
        "items": [],
    }

    if not apply:
        summary["items"] = [_dry_run_item(candidate) for candidate in candidates]
        return summary

    for candidate in candidates:
        item = process_candidate(candidate, db_path=db_path)
        summary["items"].append(item)
        summary["processed_count"] += 1

        if item["status"] == ITEM_SUCCESS:
            summary["success_count"] += 1
            storage = item["extraction_storage"]
            summary["extractions_new_count"] += storage["new_count"]
            summary["extractions_updated_count"] += storage["updated_count"]
            summary["extractions_deleted_count"] += storage["deleted_count"]
            summary["extractions_existing_count"] += storage["existing_count"]
        elif item["status"] == ITEM_NO_SUPPORTED:
            summary["no_supported_count"] += 1
        else:
            summary["failed_count"] += 1
            summary["failures"].append({
                "download_id": item["download_id"],
                "resource_url": item["resource_url"],
                "error_type": item["error_type"],
                "error_message": item["error_message"],
            })

        if item.get("processing_state") == STATE_UPDATED:
            summary["processing_state_updated_count"] += 1

    logger.info(
        "Backfill extraction завершён: кандидатов %d, успешно %d, ошибок %d, без поддержки %d, "
        "состояний обновлено %d",
        summary["candidate_count"], summary["success_count"], summary["failed_count"],
        summary["no_supported_count"], summary["processing_state_updated_count"],
    )
    return summary


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

REASON_TEXT = {
    document_repository.BACKFILL_REASON_NO_EXTRACTION: "no extraction",
    document_repository.BACKFILL_REASON_FAILED_EXTRACTION: "failed extraction",
    document_repository.BACKFILL_REASON_SKIPPED_SUPPORTED: "skipped supported members",
}


def _print_dry_run(summary: dict) -> None:
    print("DRY RUN: ничего не изменено (для записи используйте --apply)")
    print()
    print(f"Candidates: {summary['candidate_count']}")

    for number, item in enumerate(summary["items"], start=1):
        print()
        print(f"{number}. download_id={item['download_id']}")
        print(f"   {item['filename']}")
        print(f"   reason={'; '.join(REASON_TEXT.get(reason, reason) for reason in item['reasons'])}")
        if item["skipped_supported_count"]:
            print(f"   skipped_supported={item['skipped_supported_count']}")
        if item["failed_extraction_count"]:
            print(f"   failed_members={item['failed_extraction_count']}")
        print(f"   file={'exists' if item['file_exists'] else 'MISSING'}: {item['saved_path']}")


def _print_apply(summary: dict) -> None:
    print("APPLY")
    print()
    for name in (
        "candidate_count", "processed_count", "success_count", "failed_count",
        "no_supported_count", "extractions_new_count", "extractions_updated_count",
        "extractions_deleted_count", "extractions_existing_count",
        "processing_state_updated_count",
    ):
        print(f"{name}: {summary[name]}")

    for failure in summary["failures"]:
        print()
        print(f"download_id: {failure['download_id']}")
        print(f"resource_url: {failure['resource_url']}")
        print(f"error_type: {failure['error_type']}")
        print(f"error_message: {failure['error_message']}")


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("значение должно быть положительным целым")
    return number


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.backfill_document_extractions",
        description="Локальный backfill extraction скачанных документов (без сети). "
        "Без --apply — только dry run.",
    )
    parser.add_argument("--apply", action="store_true", help="выполнить обработку и записать результат")
    parser.add_argument("--limit", type=_positive_int, default=None, help="максимум кандидатов")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    summary = run_backfill(apply=args.apply, limit=args.limit)
    (_print_apply if args.apply else _print_dry_run)(summary)
    return 1 if summary["failed_count"] else 0


if __name__ == "__main__":
    sys.exit(main())
