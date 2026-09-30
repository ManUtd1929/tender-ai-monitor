"""
Provider-independent контекст тендера для AI relevance-анализа.

Читает уже собранные production-данные (announcements, enrichment, document
downloads/extractions) через существующие repository-модули и строит из них
неизменяемые dict-структуры для двух стадий анализа:

    build_tender_context()        -> полный контекст (announcement + enrichment +
                                      document_coverage + latest-версии documents);
    build_triage_context()        -> компактная выжимка для STAGE 1 (triage) с
                                      ограниченным по символам preview документов;
    build_deep_analysis_context() -> детерминированные текстовые chunks документов
                                      для STAGE 2 (пока без embeddings/vector DB —
                                      только чанкинг, LLM map/reduce здесь нет);
    compute_input_hash()          -> детерминированный hash контекста для
                                      analysis_repository (пересчёт при изменении
                                      announcement/enrichment/extraction).

Здесь нет HTTP и AI-вызовов: только чтение БД (document_repository, enrichment_repository)
и чистые функции. document_coverage — Python-derived факт (не AI): coverage_complete
учитывает failed/skipped extraction И документы, известные из enrichment, для которых
document_pipeline вообще не строит план (расширение не входит в
document_pipeline.SUPPORTED_DOCUMENT_EXTENSIONS, например .doc технические спецификации) —
такие документы никогда не скачиваются и не попадают в document_extractions, поэтому
единственный способ их заметить — enrichment["documents"] / document_url / resource_url.
"""

import hashlib
import json

from src.database import document_repository, enrichment_repository
from src.document_pipeline import SUPPORTED_DOCUMENT_EXTENSIONS, get_file_extension

# Метрики extraction, которые могут быть непустыми (лишние None не попадают в контекст).
METRIC_FIELDS = ("char_count", "paragraph_count", "table_count", "sheet_count", "row_count", "cell_count")

DEFAULT_TRIAGE_CHAR_BUDGET = 6000
DEFAULT_DEEP_CHUNK_SIZE = 4000


# --------------------------------------------------------------------------
# document coverage (Python-derived факт)
# --------------------------------------------------------------------------

def _known_document_names(combined: dict) -> list:
    """
    Документы, о существовании которых известно из enrichment, независимо от того,
    скачивались ли они (document_pipeline выбирает и скачивает только один документ
    на объявление). ARMEPS может анонсировать несколько документов сразу.
    """
    resource_type = combined.get("resource_type")
    if resource_type == "armeps_documents_page":
        return [
            document.get("filename")
            for document in (combined.get("documents") or [])
            if document.get("filename")
        ]
    if resource_type == "eauction_tender_page":
        document_url = combined.get("document_url")
        return [document_url] if document_url else []
    if resource_type == "direct_file":
        resource_url = combined.get("resource_url")
        return [resource_url] if resource_url else []
    return []


def compute_document_coverage(
    combined: dict, downloads_with_extractions: list, processing_status: str | None = None,
) -> dict:
    """
    Возвращает {successful_extractions, failed_extractions, skipped_extractions,
    successful_docx, successful_xlsx, unsupported_extensions, total_extracted_chars,
    coverage_complete}. coverage_complete = False, если:
        - есть extraction со статусом failed или skipped;
        - среди известных (enrichment) документов есть неподдерживаемое расширение
          (document_pipeline такой документ никогда не скачивает);
        - последняя попытка обработки документов завершилась failed
          (processing_status == "failed");
        - известен хотя бы один документ, но ни один extraction не succeeded
          (обработка ещё не выполнялась или ничего не дала).
    Пустой список known-документов (нет ни одного объявленного документа) считается
    complete: извлекать нечего. combined и downloads_with_extractions не изменяются.
    """
    extraction_rows = [row for item in downloads_with_extractions for row in item["extractions"]]

    successful = [row for row in extraction_rows if row["extraction_status"] == "success"]
    failed_count = sum(1 for row in extraction_rows if row["extraction_status"] == "failed")
    skipped_rows = [row for row in extraction_rows if row["extraction_status"] == "skipped"]

    unsupported = set()
    for row in skipped_rows:
        unsupported.add(get_file_extension(row["member_name"]) or "(no extension)")
    for name in _known_document_names(combined):
        extension = get_file_extension(name)
        if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
            unsupported.add(extension or "(no extension)")

    has_known_documents = bool(_known_document_names(combined))

    coverage_complete = (
        failed_count == 0
        and not skipped_rows
        and not unsupported
        and processing_status != "failed"
        and (not has_known_documents or len(successful) > 0)
    )

    return {
        "successful_extractions": len(successful),
        "failed_extractions": failed_count,
        "skipped_extractions": len(skipped_rows),
        "successful_docx": sum(1 for row in successful if row["file_type"] == "docx"),
        "successful_xlsx": sum(1 for row in successful if row["file_type"] == "xlsx"),
        "unsupported_extensions": sorted(unsupported),
        "total_extracted_chars": sum(row.get("char_count") or 0 for row in successful),
        "coverage_complete": coverage_complete,
    }


# --------------------------------------------------------------------------
# tender context
# --------------------------------------------------------------------------

def _latest_downloads_with_extractions(resource_url: str, db_path=None) -> list:
    """Только последняя версия (максимальный id) на каждый (source_kind, source_ref)."""
    downloads = document_repository.get_downloads_for_resource(resource_url, db_path=db_path)

    latest_by_key = {}
    for download in downloads:
        key = (download["source_kind"], download["source_ref"])
        current = latest_by_key.get(key)
        if current is None or download["id"] > current["id"]:
            latest_by_key[key] = download

    result = []
    for download in latest_by_key.values():
        extractions = document_repository.get_extractions_for_download(download["id"], db_path=db_path)
        result.append({"download": download, "extractions": extractions})
    return result


def _document_entries(downloads_with_extractions: list) -> list:
    entries = []
    for item in downloads_with_extractions:
        download = item["download"]
        for row in item["extractions"]:
            metrics = {name: row[name] for name in METRIC_FIELDS if row.get(name) is not None}
            entries.append({
                "download_id": download["id"],
                "member_name": row["member_name"],
                "file_type": row["file_type"],
                "extraction_status": row["extraction_status"],
                "text": row["text"],
                "metrics": metrics,
            })
    return entries


def build_tender_context(resource_url: str, db_path=None) -> dict:
    """
    Полный AI-контекст тендера: announcement + enrichment + document_coverage +
    documents (только latest download на logical source_kind/source_ref; extracted_at /
    downloaded_at / enriched_at — volatile-поля, сюда намеренно не попадают, см.
    compute_input_hash). ValueError — нет announcement/enrichment для resource_url.
    """
    combined = enrichment_repository.get_announcement_with_enrichment(resource_url, db_path=db_path)
    if combined is None:
        raise ValueError(f"Нет announcement/enrichment для resource_url: {resource_url}")

    announcement = {
        "title": combined.get("title"),
        "section": combined.get("source_section_name"),
        "resource_type": combined.get("resource_type"),
        "published_at": combined.get("published_at"),
        "deadline_at": combined.get("deadline_at"),
        "resource_url": combined.get("resource_url"),
    }

    enrichment = {
        "enrichment_status": combined.get("enrichment_status"),
        "procedure_code": combined.get("procedure_code"),
        "contracting_authority": combined.get("contracting_authority"),
        "detail_titles": {
            "default": combined.get("detail_title"),
            "ru": combined.get("detail_title_ru"),
            "en": combined.get("detail_title_en"),
        },
        "procurement_type": combined.get("procurement_type"),
        "procedure_type": combined.get("procedure_type"),
        "description": combined.get("description"),
        "estimated_value_amd": combined.get("estimated_value_amd"),
        "dates": {
            "published_at_detail": combined.get("published_at_detail"),
            "deadline_at_detail": combined.get("deadline_at_detail"),
        },
        "number_of_lots": combined.get("number_of_lots"),
        "cpv_codes": combined.get("cpv_codes") or [],
    }

    downloads_with_extractions = _latest_downloads_with_extractions(resource_url, db_path=db_path)
    processing_state = document_repository.get_processing_state(resource_url, db_path=db_path)
    processing_status = processing_state["status"] if processing_state is not None else None

    return {
        "resource_url": resource_url,
        "announcement": announcement,
        "enrichment": enrichment,
        "document_coverage": compute_document_coverage(
            combined, downloads_with_extractions, processing_status,
        ),
        "documents": _document_entries(downloads_with_extractions),
    }


# --------------------------------------------------------------------------
# triage context (STAGE 1) — компактный, с бюджетом символов
# --------------------------------------------------------------------------

def build_triage_context(tender_context: dict, char_budget: int = DEFAULT_TRIAGE_CHAR_BUDGET) -> dict:
    """
    Компактная выжимка tender_context для triage: без полного текста документов, только
    метаданные + ограниченный по char_budget preview успешных extraction (в порядке
    tender_context["documents"], до исчерпания бюджета). Provenance (download_id,
    member_name, file_type) сохраняется у каждого preview. tender_context не изменяется.
    ValueError — char_budget не положительное целое.
    """
    if isinstance(char_budget, bool) or not isinstance(char_budget, int) or char_budget <= 0:
        raise ValueError(f"char_budget должен быть положительным целым: {char_budget!r}")

    announcement = tender_context["announcement"]
    enrichment = tender_context["enrichment"]
    documents = tender_context["documents"]

    document_summaries = [
        {
            "download_id": document["download_id"],
            "member_name": document["member_name"],
            "file_type": document["file_type"],
            "extraction_status": document["extraction_status"],
        }
        for document in documents
    ]

    previews = []
    remaining = char_budget
    for document in documents:
        if remaining <= 0:
            break
        if document["extraction_status"] != "success":
            continue
        text = document.get("text") or ""
        if not text:
            continue
        take = min(len(text), remaining)
        previews.append({
            "download_id": document["download_id"],
            "member_name": document["member_name"],
            "file_type": document["file_type"],
            "preview_text": text[:take],
            "truncated": take < len(text),
        })
        remaining -= take

    return {
        "resource_url": tender_context["resource_url"],
        "title": announcement.get("title"),
        "section": announcement.get("section"),
        "resource_type": announcement.get("resource_type"),
        "published_at": announcement.get("published_at"),
        "deadline_at": announcement.get("deadline_at"),
        "detail_titles": dict(enrichment.get("detail_titles") or {}),
        "description": enrichment.get("description"),
        "procurement_type": enrichment.get("procurement_type"),
        "procedure_type": enrichment.get("procedure_type"),
        "contracting_authority": enrichment.get("contracting_authority"),
        "estimated_value_amd": enrichment.get("estimated_value_amd"),
        "dates": dict(enrichment.get("dates") or {}),
        "cpv_codes": list(enrichment.get("cpv_codes") or []),
        "document_coverage": dict(tender_context["document_coverage"]),
        "documents": document_summaries,
        "document_previews": previews,
        "char_budget": char_budget,
    }


# --------------------------------------------------------------------------
# deep analysis context (STAGE 2) — детерминированные chunks, без embeddings
# --------------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int) -> list:
    """
    Строгое разбиение text на куски длиной <= chunk_size: "".join(chunks) == text
    всегда (текст не теряется и не дублируется). Предпочитает резать по последнему
    "\\n" внутри окна [start, start+chunk_size), чтобы не резать строку посередине;
    если в окне нет "\\n" (одна длинная строка) — режет по chunk_size (иначе не избежать).
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError(f"chunk_size должен быть положительным целым: {chunk_size!r}")

    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            newline_pos = text.rfind("\n", start, end)
            if newline_pos > start:
                end = newline_pos + 1
        chunks.append(text[start:end])
        start = end
    return chunks


def build_deep_analysis_context(tender_context: dict, chunk_size: int = DEFAULT_DEEP_CHUNK_SIZE) -> dict:
    """
    Детерминированные текстовые chunks успешных document extraction для STAGE 2.
    chunk_id стабилен при неизменном входе: f"{download_id}:{member_name}:{index}".
    Только chunking — semantic embeddings/vector DB и LLM map/reduce здесь нет.
    tender_context не изменяется.
    """
    documents_with_text = [
        document for document in tender_context["documents"]
        if document["extraction_status"] == "success" and (document.get("text") or "")
    ]

    # Exact content dedup: sha256 от точного extracted text (без fuzzy/semantic сравнения,
    # имена файлов в identity не участвуют). Canonical source группы — минимальный
    # (str(download_id), member_name): не зависит от порядка documents.
    groups_by_hash = {}
    for document in documents_with_text:
        content_hash = hashlib.sha256(document["text"].encode("utf-8")).hexdigest()
        groups_by_hash.setdefault(content_hash, []).append(document)

    def _source_key(document: dict) -> tuple:
        return (str(document["download_id"]), str(document["member_name"]))

    canonical_by_hash = {
        content_hash: min(group, key=_source_key) for content_hash, group in groups_by_hash.items()
    }
    group_id_by_hash = {content_hash: f"content-{content_hash[:16]}" for content_hash in groups_by_hash}

    chunks = []
    for document in documents_with_text:
        content_hash = hashlib.sha256(document["text"].encode("utf-8")).hexdigest()
        if canonical_by_hash[content_hash] is not document:
            continue
        for index, piece in enumerate(_chunk_text(document["text"], chunk_size)):
            chunks.append({
                "chunk_id": f"{document['download_id']}:{document['member_name']}:{index}",
                "content_group_id": group_id_by_hash[content_hash],
                "download_id": document["download_id"],
                "member_name": document["member_name"],
                "file_type": document["file_type"],
                "text": piece,
            })

    content_groups = []
    for content_hash in sorted(groups_by_hash, key=lambda h: _source_key(canonical_by_hash[h])):
        canonical = canonical_by_hash[content_hash]
        content_groups.append({
            "content_group_id": group_id_by_hash[content_hash],
            "content_sha256": content_hash,
            "canonical_source": {
                "download_id": canonical["download_id"],
                "member_name": canonical["member_name"],
                "file_type": canonical["file_type"],
            },
            "represented_sources": [
                {
                    "download_id": document["download_id"],
                    "member_name": document["member_name"],
                    "file_type": document["file_type"],
                }
                for document in sorted(groups_by_hash[content_hash], key=_source_key)
            ],
        })

    return {
        "resource_url": tender_context["resource_url"],
        "chunk_size": chunk_size,
        "chunks": chunks,
        "content_groups": content_groups,
        "document_content_stats": {
            "total_documents_with_text": len(documents_with_text),
            "unique_content_groups": len(content_groups),
            "duplicate_documents": len(documents_with_text) - len(content_groups),
        },
    }


# --------------------------------------------------------------------------
# input hash
# --------------------------------------------------------------------------

def compute_input_hash(tender_context: dict) -> str:
    """
    Детерминированный sha256 от AI-relevant части tender_context. tender_context,
    построенный build_tender_context(), уже не содержит volatile-полей (enriched_at,
    downloaded_at, extracted_at, first_seen_at, last_seen_at, analyzed_at) — только их
    и добавляет repository при чтении, а не build_tender_context. Одинаковый по
    содержанию tender_context (в т.ч. при другом порядке ключей dict) даёт одинаковый
    hash благодаря sort_keys; изменение title/enrichment/extraction text меняет hash.
    """
    payload = json.dumps(tender_context, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
