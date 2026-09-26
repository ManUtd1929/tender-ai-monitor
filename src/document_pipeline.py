"""
Оркестрация обработки документов тендера: выбор источника -> скачивание ->
сохранение download -> извлечение текста -> сохранение extraction.

Вход — enriched announcement (resource_url, resource_type и поля документа из
enrichment). Три вида источников:

    eauction_tender_page  -> document_url (ZIP/DOCX/XLSX приглашения);
    armeps_documents_page -> один документ из documents[] (приоритет языка RU, EN, HY,
                             внутри языка формата .docx, .xlsx, .zip);
    direct_file           -> сам resource_url.

Поддерживаются только .docx, .xlsx и .zip (внутри — DOCX, XLSX и вложенные ZIP). .doc, .pdf,
.xml, .xls, .rar не скачиваются: для них план документа не строится. AI, PDF/.doc parser и OCR здесь
нет. Модуль вызывает monitor.py; ничего не скачивается при импорте и без явного вызова
функций.

Итог обработки объявления записывается в document_processing_state: success,
no_supported_document (плана нет — HTTP не выполняется) или failed (исключение). По этому
состоянию monitor решает, какие объявления обрабатывать повторно.

HTTP выполняют только функции document_downloader. Их ошибки здесь не перехватываются
на уровне одного объявления — их собирает process_enriched_announcements.

Повторный вызов для того же объявления снова скачивает файл: тот же это файл или новая
версия, определяет document_repository по sha256. HEAD/ETag/Last-Modified-кэша нет.
"""

import logging
import posixpath
from urllib.parse import unquote, urlparse

from src.database import document_repository
from src.parser.document_downloader import (
    download_armeps_document,
    download_direct_file,
    download_eauction_document,
)
from src.parser.document_extractor import (
    extract_docx,
    extract_supported_from_zip,
    extract_xlsx,
)

logger = logging.getLogger(__name__)

# Порядок = приоритет формата внутри одного языка ARMEPS.
SUPPORTED_DOCUMENT_EXTENSIONS = (
    ".docx",
    ".xlsx",
    ".zip",
)

# Только для выбора документа ARMEPS; используется поле language из enrichment.
ARMEPS_LANGUAGE_PRIORITY = (
    "RU",
    "EN",
    "HY",
)

RESOURCE_TYPE_EAUCTION = "eauction_tender_page"
RESOURCE_TYPE_ARMEPS = "armeps_documents_page"
RESOURCE_TYPE_DIRECT_FILE = "direct_file"

STATUS_SUCCESS = document_repository.PROCESSING_SUCCESS
STATUS_NO_SUPPORTED_DOCUMENT = document_repository.PROCESSING_NO_SUPPORTED_DOCUMENT
STATUS_FAILED = document_repository.PROCESSING_FAILED


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------

def _url_basename(filename_or_url: str) -> str:
    """Последний сегмент path без query/fragment, URL-decoded ("" если его нет)."""
    if not isinstance(filename_or_url, str):
        return ""
    path = unquote(urlparse(filename_or_url).path).replace("\\", "/")
    return path.rsplit("/", 1)[-1].strip()


def get_file_extension(filename_or_url: str) -> str:
    """'https://example/x.DOCX?foo=1' -> '.docx'; нет расширения -> ''."""
    return posixpath.splitext(_url_basename(filename_or_url))[1].lower()


def _is_supported(filename_or_url: str) -> bool:
    return get_file_extension(filename_or_url) in SUPPORTED_DOCUMENT_EXTENSIONS


def _is_blank(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _normalize_language(language) -> str:
    return language.strip().upper() if isinstance(language, str) else ""


def _best_format(documents: list[dict]) -> dict:
    """Документ с наиболее приоритетным расширением; min берёт первый из равных."""
    return min(
        documents,
        key=lambda document: SUPPORTED_DOCUMENT_EXTENSIONS.index(
            get_file_extension(document["filename"])
        ),
    )


def select_armeps_document(documents: list[dict]) -> dict | None:
    """
    Выбирает один документ ARMEPS. Кандидаты: непустые document_id и filename,
    расширение из SUPPORTED_DOCUMENT_EXTENSIONS (XML и прочее не выбираются).
    Язык важнее формата: сначала первый язык из ARMEPS_LANGUAGE_PRIORITY, у которого
    есть кандидаты, иначе все кандидаты. Внутри группы побеждает формат по порядку
    SUPPORTED_DOCUMENT_EXTENSIONS (.docx, .xlsx, .zip), при равенстве — исходный порядок.
    Список и документы не изменяются.
    """
    candidates = [
        document for document in (documents or [])
        if isinstance(document, dict)
        and not _is_blank(document.get("document_id"))
        and not _is_blank(document.get("filename"))
        and _is_supported(document["filename"])
    ]

    for language in ARMEPS_LANGUAGE_PRIORITY:
        group = [
            document for document in candidates
            if _normalize_language(document.get("language")) == language
        ]
        if group:
            return _best_format(group)

    return _best_format(candidates) if candidates else None


def _build_plan(
    resource_url, source_kind, source_ref, download_type, source_url,
    document_id=None, expected_filename=None,
) -> dict:
    return {
        "resource_url": resource_url,
        "source_kind": source_kind,
        "source_ref": source_ref,
        "download_type": download_type,
        "source_url": source_url,
        "document_id": document_id,
        "expected_filename": expected_filename,
    }


def _plan_eauction(enriched_announcement: dict) -> dict | None:
    document_url = enriched_announcement.get("document_url")
    if _is_blank(document_url) or not _is_supported(document_url):
        return None
    return _build_plan(
        enriched_announcement.get("resource_url"),
        "eauction_document", document_url, "eauction", document_url,
    )


def _plan_armeps(enriched_announcement: dict) -> dict | None:
    document = select_armeps_document(enriched_announcement.get("documents"))
    if document is None:
        return None
    resource_url = enriched_announcement.get("resource_url")
    # source_url — страница listContractDocuments: именно она нужна download_armeps_document.
    return _build_plan(
        resource_url, "armeps_document", document["document_id"], "armeps", resource_url,
        document_id=document["document_id"], expected_filename=document["filename"],
    )


def _plan_direct_file(enriched_announcement: dict) -> dict | None:
    resource_url = enriched_announcement.get("resource_url")
    if _is_blank(resource_url) or not _is_supported(resource_url):
        return None
    return _build_plan(
        resource_url, "direct_file", resource_url, "direct", resource_url,
        expected_filename=_url_basename(resource_url) or None,
    )


def plan_document_source(enriched_announcement: dict) -> dict | None:
    """
    План одного источника документа или None, если поддерживаемого документа нет
    (нет документа, неподдерживаемое расширение, неизвестный resource_type).
    """
    resource_type = enriched_announcement.get("resource_type")
    if resource_type == RESOURCE_TYPE_EAUCTION:
        return _plan_eauction(enriched_announcement)
    if resource_type == RESOURCE_TYPE_ARMEPS:
        return _plan_armeps(enriched_announcement)
    if resource_type == RESOURCE_TYPE_DIRECT_FILE:
        return _plan_direct_file(enriched_announcement)
    return None


# --------------------------------------------------------------------------
# Download / extract
# --------------------------------------------------------------------------

def download_planned_source(plan: dict, root_dir=None) -> dict:
    """Скачивает документ по плану. HTTP-ошибки downloader-а не перехватываются."""
    download_type = plan["download_type"]

    if download_type == "eauction":
        return download_eauction_document(
            document_url=plan["source_url"],
            resource_url=plan["resource_url"],
            root_dir=root_dir,
        )
    if download_type == "armeps":
        return download_armeps_document(
            resource_url=plan["source_url"],
            document_id=plan["document_id"],
            expected_filename=plan["expected_filename"],
            root_dir=root_dir,
        )
    if download_type == "direct":
        return download_direct_file(
            url=plan["source_url"],
            resource_url=plan["resource_url"],
            root_dir=root_dir,
        )
    raise ValueError(f"Unknown download_type: {download_type!r}")


def extract_downloaded_file(download_result: dict) -> dict:
    """Извлекает текст из скачанного .docx, .xlsx или .zip; тип определяется по saved_path / filename."""
    saved_path = download_result.get("saved_path")

    for name in (saved_path, download_result.get("filename")):
        extension = get_file_extension(name)
        if extension == ".docx":
            return {"extraction_type": "docx", "result": extract_docx(saved_path)}
        if extension == ".xlsx":
            return {"extraction_type": "xlsx", "result": extract_xlsx(saved_path)}
        if extension == ".zip":
            return {"extraction_type": "zip", "result": extract_supported_from_zip(saved_path)}

    raise ValueError(
        f"Unsupported file for extraction: {saved_path!r} "
        f"(filename={download_result.get('filename')!r})"
    )


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def process_enriched_announcement(
    enriched_announcement: dict, db_path=None, root_dir=None,
) -> dict:
    """
    План -> скачивание -> save_download -> извлечение -> save_*_extraction.
    Без поддерживаемого документа возвращает status="no_supported_document" без HTTP и
    без записи download/extraction. Итог (success / no_supported_document) сохраняется в
    document_processing_state. Ошибки downloader/extractor/repository не перехватываются.
    """
    resource_url = enriched_announcement.get("resource_url")

    plan = plan_document_source(enriched_announcement)
    if plan is None:
        logger.info(
            "Нет поддерживаемого документа: %s (resource_type=%s)",
            resource_url, enriched_announcement.get("resource_type"),
        )
        document_repository.save_processing_state(
            resource_url, STATUS_NO_SUPPORTED_DOCUMENT, db_path=db_path,
        )
        return {
            "resource_url": resource_url,
            "status": STATUS_NO_SUPPORTED_DOCUMENT,
            "plan": None,
        }

    if _is_blank(plan["resource_url"]):
        raise ValueError("У объявления отсутствует resource_url, документ не скачивается")

    logger.info(
        "План документа: %s [%s %s]", resource_url, plan["source_kind"], plan["source_ref"],
    )

    download_result = download_planned_source(plan, root_dir=root_dir)

    saved = document_repository.save_download(
        resource_url=plan["resource_url"],
        source_kind=plan["source_kind"],
        source_ref=plan["source_ref"],
        download_result=download_result,
        document_id=plan["document_id"],
        db_path=db_path,
    )
    download_id = saved["download_id"]

    extraction = extract_downloaded_file(download_result)
    extraction_type = extraction["extraction_type"]
    if extraction_type == "docx":
        extraction_storage = document_repository.save_docx_extraction(
            download_id, extraction["result"], db_path=db_path,
        )
    elif extraction_type == "xlsx":
        extraction_storage = document_repository.save_xlsx_extraction(
            download_id, extraction["result"], db_path=db_path,
        )
    else:
        extraction_storage = document_repository.save_zip_extraction(
            download_id, extraction["result"], db_path=db_path,
        )

    document_repository.save_processing_state(
        plan["resource_url"], STATUS_SUCCESS,
        source_kind=plan["source_kind"], source_ref=plan["source_ref"], db_path=db_path,
    )

    logger.info(
        "Документ обработан: %s, download_id=%s (%s), extraction=%s",
        resource_url, download_id, saved["status"], extraction_type,
    )
    return {
        "resource_url": resource_url,
        "status": STATUS_SUCCESS,
        "plan": plan,
        "download": download_result,
        "download_storage_status": saved["status"],
        "download_id": download_id,
        "extraction_type": extraction_type,
        "extraction_storage": extraction_storage,
    }


def _save_failed_state(resource_url, error: Exception, db_path=None) -> None:
    """Записывает status=failed; сбой самой записи только логируется (не прерывает цикл)."""
    if _is_blank(resource_url):
        return
    try:
        document_repository.save_processing_state(
            resource_url, STATUS_FAILED,
            error_type=type(error).__name__, error_message=str(error), db_path=db_path,
        )
    except Exception:
        logger.exception("Не удалось сохранить состояние failed: %s", resource_url)


def process_enriched_announcements(
    enriched_announcements: list[dict], db_path=None, root_dir=None,
) -> dict:
    """
    Последовательно обрабатывает объявления; ошибка одного попадает в failures и
    сохраняется как status=failed в document_processing_state. Ошибка init_db не
    перехватывается и состояний не создаёт.
    """
    logger.info("Запуск обработки документов: объявлений %d", len(enriched_announcements))

    document_repository.init_db(db_path=db_path)

    results = []
    failures = []
    extraction_totals = {"new_count": 0, "updated_count": 0, "existing_count": 0}
    download_new = 0
    download_existing = 0
    no_supported = 0

    for enriched_announcement in enriched_announcements:
        try:
            result = process_enriched_announcement(
                enriched_announcement, db_path=db_path, root_dir=root_dir,
            )
        except Exception as error:
            resource_url = (
                enriched_announcement.get("resource_url")
                if isinstance(enriched_announcement, dict) else None
            )
            logger.exception("Не удалось обработать документ объявления: %s", resource_url)
            failures.append({
                "resource_url": resource_url,
                "error_type": type(error).__name__,
                "error_message": str(error),
            })
            _save_failed_state(resource_url, error, db_path=db_path)
            continue

        results.append(result)

        if result["status"] == STATUS_NO_SUPPORTED_DOCUMENT:
            no_supported += 1
            continue

        if result["download_storage_status"] == document_repository.STATUS_NEW:
            download_new += 1
        else:
            download_existing += 1
        for key in extraction_totals:
            extraction_totals[key] += result["extraction_storage"].get(key, 0)

    success_count = sum(1 for result in results if result["status"] == STATUS_SUCCESS)

    logger.info(
        "Обработка документов завершена: всего %d, успешно %d, без документа %d, ошибок %d "
        "(download: новых %d, без изменений %d; extraction: новых %d, обновлённых %d, "
        "без изменений %d)",
        len(enriched_announcements), success_count, no_supported, len(failures),
        download_new, download_existing,
        extraction_totals["new_count"], extraction_totals["updated_count"],
        extraction_totals["existing_count"],
    )

    return {
        "processed_count": len(enriched_announcements),
        "success_count": success_count,
        "no_supported_document_count": no_supported,
        "failed_count": len(failures),
        "download_new_count": download_new,
        "download_existing_count": download_existing,
        "extraction_new_count": extraction_totals["new_count"],
        "extraction_updated_count": extraction_totals["updated_count"],
        "extraction_existing_count": extraction_totals["existing_count"],
        "results": results,
        "failures": failures,
    }


def main():
    print("Document pipeline module. Use explicit processing functions.")


if __name__ == "__main__":
    main()
