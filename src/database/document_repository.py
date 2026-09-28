"""
Production-хранилище скачанных документов и извлечённого текста (таблицы
document_downloads и document_extractions) на стандартном sqlite3.

document_downloads — metadata скачанных файлов. Логический документ определяется
тройкой (resource_url, source_kind, source_ref); source_ref репозиторий не придумывает,
его передаёт вызывающий код. Если bytes изменились (другой sha256), добавляется новая
строка-версия, старая сохраняется. Тот же sha256 — дубль не создаётся.

document_extractions — результат извлечения текста: по одной строке на
(download_id, member_name). Для обычного DOCX member_name = "", для ZIP — имя member.
Статусы: success, failed, skipped (SQL CHECK намеренно нет). Метрики DOCX — paragraph_count,
table_count; метрики XLSX — sheet_count, row_count, cell_count (у чужого типа NULL).
Для member вложенного ZIP member_name имеет вид "inner.zip!/file.xlsx".

document_processing_state — итог обработки документов объявления (по одной строке на
resource_url): success, no_supported_document или failed (SQL CHECK намеренно нет). По ней
get_document_processing_candidates решает, какие объявления ещё нужно обработать.

replace_extractions_for_download — authoritative snapshot extraction одного (неизменяемого по
sha256) download: сохраняет текущие строки и удаляет устаревшие одной транзакцией; её использует
локальный backfill. get_document_extraction_backfill_candidates выбирает download, extraction
которых нужно пересчитать (открывает БД только на чтение).

Таблица announcements здесь только читается (проверка resource_url); её схема
и данные не изменяются. Общий у модулей только файл БД. Запросы кандидатов читают также
announcement_enrichment и enrichment_processing_state, поэтому перед ними должен быть
вызван enrichment_repository.init_db.
"""

import logging
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path

from src.database import announcement_repository, enrichment_repository
from src.database.tender_repository import DEFAULT_DB_PATH
from src.parser.document_extractor import DEFAULT_MAX_NESTED_DEPTH

logger = logging.getLogger(__name__)

STATUS_NEW = announcement_repository.STATUS_NEW
STATUS_UPDATED = announcement_repository.STATUS_UPDATED
STATUS_EXISTING = announcement_repository.STATUS_EXISTING

EXTRACTION_SUCCESS = "success"
EXTRACTION_FAILED = "failed"
EXTRACTION_SKIPPED = "skipped"

PROCESSING_SUCCESS = "success"
PROCESSING_NO_SUPPORTED_DOCUMENT = "no_supported_document"
PROCESSING_FAILED = "failed"

# Поля download_result из document_downloader; size_bytes проверяется отдельно (0 допустим).
REQUIRED_DOWNLOAD_RESULT_FIELDS = ("source_url", "saved_path", "filename", "sha256")

# Поля extraction, изменение которых считается обновлением (extracted_at сюда не входит).
EXTRACTION_FIELDS = (
    "file_type",
    "extraction_status",
    "text",
    "char_count",
    "paragraph_count",
    "table_count",
    "sheet_count",
    "row_count",
    "cell_count",
    "error_type",
    "error_message",
)

# Колонки, которых нет в БД, созданных до поддержки XLSX: добавляются в init_db.
EXTRACTION_MIGRATION_COLUMNS = (
    ("sheet_count", "INTEGER"),
    ("row_count", "INTEGER"),
    ("cell_count", "INTEGER"),
)

CREATE_DOWNLOADS_TABLE = """
CREATE TABLE IF NOT EXISTS document_downloads (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    resource_url  TEXT NOT NULL,
    source_kind   TEXT NOT NULL,
    source_ref    TEXT NOT NULL,
    source_url    TEXT NOT NULL,
    document_id   TEXT,
    filename      TEXT NOT NULL,
    saved_path    TEXT NOT NULL,
    content_type  TEXT,
    size_bytes    INTEGER NOT NULL,
    sha256        TEXT NOT NULL,
    downloaded_at TEXT NOT NULL,
    FOREIGN KEY (resource_url) REFERENCES announcements (resource_url),
    UNIQUE (resource_url, source_kind, source_ref, sha256)
)
"""

CREATE_EXTRACTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS document_extractions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    download_id       INTEGER NOT NULL,
    member_name       TEXT NOT NULL DEFAULT '',
    file_type         TEXT,
    extraction_status TEXT NOT NULL,
    text              TEXT,
    char_count        INTEGER,
    paragraph_count   INTEGER,
    table_count       INTEGER,
    sheet_count       INTEGER,
    row_count         INTEGER,
    cell_count        INTEGER,
    error_type        TEXT,
    error_message     TEXT,
    extracted_at      TEXT NOT NULL,
    FOREIGN KEY (download_id) REFERENCES document_downloads (id) ON DELETE CASCADE,
    UNIQUE (download_id, member_name)
)
"""

CREATE_PROCESSING_STATE_TABLE = """
CREATE TABLE IF NOT EXISTS document_processing_state (
    resource_url    TEXT PRIMARY KEY,
    status          TEXT NOT NULL,
    source_kind     TEXT,
    source_ref      TEXT,
    last_attempt_at TEXT NOT NULL,
    error_type      TEXT,
    error_message   TEXT,
    FOREIGN KEY (resource_url) REFERENCES announcements (resource_url)
)
"""

UPSERT_PROCESSING_STATE = """
INSERT INTO document_processing_state (
    resource_url, status, source_kind, source_ref, last_attempt_at, error_type, error_message
) VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (resource_url) DO UPDATE SET
    status = excluded.status,
    source_kind = excluded.source_kind,
    source_ref = excluded.source_ref,
    last_attempt_at = excluded.last_attempt_at,
    error_type = excluded.error_type,
    error_message = excluded.error_message
"""

# Кандидаты: enrichment уже есть, а состояния обработки нет или оно failed.
# Объявления с failed refresh enrichment исключаются: их enrichment устарел, документы
# нужно обрабатывать только после успешного повторного enrichment.
FROM_PROCESSING_CANDIDATES = """
FROM announcements
JOIN announcement_enrichment
    ON announcement_enrichment.resource_url = announcements.resource_url
LEFT JOIN document_processing_state
    ON document_processing_state.resource_url = announcements.resource_url
LEFT JOIN enrichment_processing_state
    ON enrichment_processing_state.resource_url = announcements.resource_url
WHERE (document_processing_state.resource_url IS NULL
        OR document_processing_state.status = 'failed')
    AND (enrichment_processing_state.status IS NULL
        OR enrichment_processing_state.status != 'failed')
"""

# Сначала ни разу не обработанные, затем failed: постоянно падающие документы
# не должны блокировать новые (limit берёт из начала списка).
SELECT_PROCESSING_CANDIDATE_URLS = (
    f"SELECT announcements.resource_url {FROM_PROCESSING_CANDIDATES} "
    "ORDER BY CASE WHEN document_processing_state.resource_url IS NULL THEN 0 ELSE 1 END ASC, "
    "announcements.first_seen_at ASC, announcements.id ASC"
)

COUNT_PROCESSING_CANDIDATES = f"SELECT COUNT(*) {FROM_PROCESSING_CANDIDATES}"

SELECT_EXISTING_DOWNLOAD = """
SELECT id FROM document_downloads
WHERE resource_url = ? AND source_kind = ? AND source_ref = ? AND sha256 = ?
"""

INSERT_DOWNLOAD = """
INSERT INTO document_downloads (
    resource_url, source_kind, source_ref, source_url, document_id, filename,
    saved_path, content_type, size_bytes, sha256, downloaded_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

SELECT_EXTRACTION = (
    f"SELECT id, {', '.join(EXTRACTION_FIELDS)} FROM document_extractions "
    "WHERE download_id = ? AND member_name = ?"
)

INSERT_EXTRACTION = f"""
INSERT INTO document_extractions (
    download_id, member_name, {', '.join(EXTRACTION_FIELDS)}, extracted_at
) VALUES ({', '.join('?' * (len(EXTRACTION_FIELDS) + 3))})
"""

UPDATE_EXTRACTION = f"""
UPDATE document_extractions
SET {', '.join(f'{name} = ?' for name in EXTRACTION_FIELDS)}, extracted_at = ?
WHERE id = ?
"""


def _resolve_path(db_path) -> Path:
    return Path(db_path) if db_path is not None else DEFAULT_DB_PATH


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _connect(db_path=None):
    """Соединение с foreign_keys = ON; commit/rollback и закрытие гарантированы."""
    path = _resolve_path(db_path)
    with closing(sqlite3.connect(path)) as conn:
        # PRAGMA действует только вне транзакции, поэтому выполняется сразу после connect.
        conn.execute("PRAGMA foreign_keys = ON")
        with conn:
            yield conn


def _add_missing_extraction_columns(conn: sqlite3.Connection) -> None:
    """Добавляет метрики XLSX в document_extractions старой схемы; строки не изменяются."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(document_extractions)")}
    for name, column_type in EXTRACTION_MIGRATION_COLUMNS:
        if name not in existing:
            conn.execute(f"ALTER TABLE document_extractions ADD COLUMN {name} {column_type}")
            logger.info("Колонка добавлена в document_extractions: %s", name)


def init_db(db_path=None) -> Path:
    """
    Создаёт announcements (если нет) и таблицы документов и состояния; добавляет в
    document_extractions недостающие колонки метрик XLSX. Существующие строки не изменяет.
    """
    path = announcement_repository.init_db(db_path)

    with _connect(path) as conn:
        conn.execute(CREATE_DOWNLOADS_TABLE)
        conn.execute(CREATE_EXTRACTIONS_TABLE)
        _add_missing_extraction_columns(conn)
        conn.execute(CREATE_PROCESSING_STATE_TABLE)

    logger.info("Таблицы документов готовы: %s", path)
    return path


def _is_blank(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _validate_limit(limit) -> None:
    """limit — None или положительное целое (bool — подкласс int, но это почти наверняка ошибка)."""
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
    ):
        raise ValueError(f"limit должен быть положительным целым или None: {limit!r}")


# --- downloads ---

def _save_download(
    conn: sqlite3.Connection,
    resource_url: str,
    source_kind: str,
    source_ref: str,
    download_result: dict,
    document_id: str | None,
) -> dict:
    missing = [
        name for name, value in (
            ("resource_url", resource_url),
            ("source_kind", source_kind),
            ("source_ref", source_ref),
        )
        if _is_blank(value)
    ]
    missing += [
        name for name in REQUIRED_DOWNLOAD_RESULT_FIELDS
        if _is_blank(download_result.get(name))
    ]
    if download_result.get("size_bytes") is None:
        missing.append("size_bytes")
    if missing:
        raise ValueError(f"Для скачанного документа отсутствуют поля: {', '.join(missing)}")

    exists = conn.execute(
        "SELECT 1 FROM announcements WHERE resource_url = ?", (resource_url,)
    ).fetchone()
    if exists is None:
        raise ValueError(f"Объявление не найдено в announcements: {resource_url}")

    sha256 = download_result["sha256"]
    row = conn.execute(
        SELECT_EXISTING_DOWNLOAD, (resource_url, source_kind, source_ref, sha256)
    ).fetchone()
    if row is not None:
        logger.info(
            "Скачанный документ уже есть в базе (id=%d): %s [%s %s]",
            row[0], download_result["filename"], source_kind, source_ref,
        )
        return {"status": STATUS_EXISTING, "download_id": row[0]}

    cursor = conn.execute(
        INSERT_DOWNLOAD,
        (
            resource_url,
            source_kind,
            source_ref,
            download_result["source_url"],
            document_id,
            download_result["filename"],
            download_result["saved_path"],
            download_result.get("content_type"),
            download_result["size_bytes"],
            sha256,
            _utc_now(),
        ),
    )
    logger.info(
        "Скачанный документ сохранён (id=%d): %s [%s %s]",
        cursor.lastrowid, download_result["filename"], source_kind, source_ref,
    )
    return {"status": STATUS_NEW, "download_id": cursor.lastrowid}


def save_download(
    resource_url: str,
    source_kind: str,
    source_ref: str,
    download_result: dict,
    document_id: str | None = None,
    db_path=None,
) -> dict:
    """
    Возвращает {"status": "new" | "existing", "download_id": int}.
    Тот же (resource_url, source_kind, source_ref, sha256) — "existing" без новой строки;
    другой sha256 при том же source_ref — новая версия ("new").
    ValueError — не хватает обязательных полей или нет объявления в announcements.
    """
    with _connect(db_path) as conn:
        return _save_download(
            conn, resource_url, source_kind, source_ref, download_result, document_id
        )


# --- extractions ---

def _save_extraction(conn: sqlite3.Connection, download_id: int, extraction: dict) -> str:
    """Сохраняет одну строку extraction на переданном соединении (без commit)."""
    status = extraction.get("extraction_status")
    if _is_blank(status):
        raise ValueError(f"У extraction отсутствует extraction_status (download_id={download_id})")

    exists = conn.execute(
        "SELECT 1 FROM document_downloads WHERE id = ?", (download_id,)
    ).fetchone()
    if exists is None:
        raise ValueError(f"Скачанный документ не найден в document_downloads: {download_id}")

    # NULL в member_name сломал бы UNIQUE (NULL не равен NULL), поэтому всегда строка.
    member_name = extraction.get("member_name") or ""
    values = tuple(extraction.get(name) for name in EXTRACTION_FIELDS)
    now = _utc_now()

    row = conn.execute(SELECT_EXTRACTION, (download_id, member_name)).fetchone()

    if row is None:
        conn.execute(INSERT_EXTRACTION, (download_id, member_name, *values, now))
        logger.info(
            "Extraction сохранён (new, %s): download_id=%d member=%r",
            status, download_id, member_name,
        )
        return STATUS_NEW

    row_id, old_values = row[0], tuple(row[1:])
    if old_values == values:
        logger.info(
            "Extraction без изменений: download_id=%d member=%r", download_id, member_name
        )
        return STATUS_EXISTING

    changed = [name for name, old, new in zip(EXTRACTION_FIELDS, old_values, values) if old != new]
    conn.execute(UPDATE_EXTRACTION, (*values, now, row_id))
    logger.info(
        "Extraction обновлён (%s): download_id=%d member=%r",
        ", ".join(changed), download_id, member_name,
    )
    return STATUS_UPDATED


def save_extraction(download_id: int, extraction: dict, db_path=None) -> str:
    """
    Возвращает "new", "updated" или "existing" (ключ — download_id + member_name).
    ValueError — нет extraction_status или download_id не существует.
    """
    with _connect(db_path) as conn:
        return _save_extraction(conn, download_id, extraction)


def _counts(statuses: list[str]) -> dict:
    return {
        "new_count": statuses.count(STATUS_NEW),
        "updated_count": statuses.count(STATUS_UPDATED),
        "existing_count": statuses.count(STATUS_EXISTING),
    }


def _success_extraction(extraction_result: dict, member_name: str) -> dict:
    """Успешный результат extract_docx() / extract_xlsx(); метрики чужого типа остаются None."""
    row = {
        "member_name": member_name,
        "extraction_status": EXTRACTION_SUCCESS,
        "error_type": None,
        "error_message": None,
    }
    for name in EXTRACTION_FIELDS:
        if name not in row:
            row[name] = extraction_result.get(name)
    return row


def save_docx_extraction(download_id: int, extraction_result: dict, db_path=None) -> dict:
    """Результат extract_docx() -> одна строка с member_name="" и статусом success."""
    with _connect(db_path) as conn:
        status = _save_extraction(conn, download_id, _success_extraction(extraction_result, ""))
    return _counts([status])


def save_xlsx_extraction(download_id: int, extraction_result: dict, db_path=None) -> dict:
    """Результат extract_xlsx() -> одна строка с member_name="" и статусом success."""
    with _connect(db_path) as conn:
        status = _save_extraction(conn, download_id, _success_extraction(extraction_result, ""))
    return _counts([status])


def _zip_extraction_rows(extraction_result: dict) -> list[dict]:
    """Результат extract_supported_from_zip() -> строки success, затем skipped, затем failed."""
    rows = [
        _success_extraction(item, item["member_name"])
        for item in extraction_result.get("documents", [])
    ]
    rows += [
        {
            "member_name": name,
            "extraction_status": EXTRACTION_SKIPPED,
        }
        for name in extraction_result.get("skipped_members", [])
    ]
    rows += [
        {
            "member_name": item["member_name"],
            "extraction_status": EXTRACTION_FAILED,
            "error_type": item.get("error_type"),
            "error_message": item.get("error_message"),
        }
        for item in extraction_result.get("failures", [])
    ]
    return rows


def build_extraction_records(extraction_type: str, extraction_result: dict) -> list[dict]:
    """
    Результат extract_docx / extract_xlsx / extract_supported_from_zip -> список строк
    extraction (success / skipped / failed) с метриками. extraction_type: "docx", "xlsx", "zip".
    Обычный документ — одна строка с member_name="", ZIP — по строке на member (member_name
    вложенного ZIP вида "outer.zip!/inner.xlsx"). ValueError — неизвестный extraction_type.
    """
    if extraction_type in ("docx", "xlsx"):
        return [_success_extraction(extraction_result, "")]
    if extraction_type == "zip":
        return _zip_extraction_rows(extraction_result)
    raise ValueError(f"Неизвестный extraction_type: {extraction_type!r}")


def replace_extractions_for_download(
    download_id: int, extraction_records: list[dict], db_path=None
) -> dict:
    """
    Authoritative snapshot extraction для download_id: сохраняет все extraction_records
    (как save_extraction: new / updated / existing) и удаляет строки этого download, которых
    в snapshot нет (например, прежний skipped "lot_1.zip", вместо которого теперь
    "lot_1.zip!/spec.xlsx"). Одна транзакция: при любой ошибке откатывается всё. Безопасно,
    потому что download неизменяем по sha256; document_downloads не изменяется. Пустой список
    удаляет все extraction этого download. ValueError — download_id не существует или у
    записи нет extraction_status.
    Возвращает {"new_count", "updated_count", "existing_count", "deleted_count"}.
    """
    with _connect(db_path) as conn:
        exists = conn.execute(
            "SELECT 1 FROM document_downloads WHERE id = ?", (download_id,)
        ).fetchone()
        if exists is None:
            raise ValueError(f"Скачанный документ не найден в document_downloads: {download_id}")

        statuses = [_save_extraction(conn, download_id, record) for record in extraction_records]
        kept = {record.get("member_name") or "" for record in extraction_records}

        obsolete = [
            (row_id, member_name)
            for row_id, member_name in conn.execute(
                "SELECT id, member_name FROM document_extractions WHERE download_id = ?",
                (download_id,),
            ).fetchall()
            if member_name not in kept
        ]
        for row_id, member_name in obsolete:
            conn.execute("DELETE FROM document_extractions WHERE id = ?", (row_id,))
            logger.info(
                "Устаревший extraction удалён: download_id=%d member=%r", download_id, member_name
            )

    counts = {**_counts(statuses), "deleted_count": len(obsolete)}
    logger.info(
        "Extraction snapshot сохранён (download_id=%d): записей %d, новых %d, обновлённых %d, "
        "без изменений %d, удалено %d",
        download_id, len(extraction_records), counts["new_count"], counts["updated_count"],
        counts["existing_count"], counts["deleted_count"],
    )
    return counts


def save_zip_extraction(download_id: int, extraction_result: dict, db_path=None) -> dict:
    """
    Результат extract_supported_from_zip() -> строки documents (success; DOCX и XLSX со
    своими метриками; member_name вложенного ZIP вида "inner.zip!/a.xlsx"),
    skipped_members (skipped) и failures (failed). Весь результат сохраняется одной
    транзакцией: при ошибке откатываются все member.
    """
    rows = _zip_extraction_rows(extraction_result)

    with _connect(db_path) as conn:
        statuses = [_save_extraction(conn, download_id, row) for row in rows]

    counts = _counts(statuses)
    logger.info(
        "ZIP extraction сохранён (download_id=%d): member %d, новых %d, обновлённых %d, "
        "без изменений %d",
        download_id, len(rows), counts["new_count"], counts["updated_count"],
        counts["existing_count"],
    )
    return counts


# --- состояние обработки ---

def save_processing_state(
    resource_url: str,
    status: str,
    source_kind: str | None = None,
    source_ref: str | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    db_path=None,
) -> None:
    """
    Upsert по resource_url; last_attempt_at — текущее UTC-время (timezone-aware).
    Все поля строки перезаписываются: успешная попытка очищает прежнюю ошибку.
    ValueError — пустой resource_url/status или нет объявления в announcements.
    """
    if _is_blank(resource_url):
        raise ValueError("resource_url не может быть пустым")
    if _is_blank(status):
        raise ValueError(f"У состояния обработки отсутствует status: {resource_url}")

    with _connect(db_path) as conn:
        exists = conn.execute(
            "SELECT 1 FROM announcements WHERE resource_url = ?", (resource_url,)
        ).fetchone()
        if exists is None:
            raise ValueError(f"Объявление не найдено в announcements: {resource_url}")

        conn.execute(
            UPSERT_PROCESSING_STATE,
            (resource_url, status, source_kind, source_ref, _utc_now(), error_type, error_message),
        )

    logger.info("Состояние обработки документов сохранено (%s): %s", status, resource_url)


def get_processing_state(resource_url: str, db_path=None) -> dict | None:
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM document_processing_state WHERE resource_url = ?", (resource_url,)
        ).fetchone()
    return dict(row) if row is not None else None


def get_document_processing_candidates(db_path=None, limit: int | None = None) -> list[dict]:
    """
    Объявления с enrichment, обработка документов которых не завершена: состояния нет
    или оно failed (success и no_supported_document не возвращаются). Сначала ни разу не
    обработанные, затем failed; внутри группы самые старые первыми (first_seen_at, id).
    Объявления, у которых последний refresh enrichment failed, не возвращаются.
    Каждый dict — результат enrichment_repository.get_announcement_with_enrichment,
    то есть готов для document_pipeline.process_enriched_announcement.
    limit=None — все; limit — положительное целое, иначе ValueError.
    """
    _validate_limit(limit)
    query = SELECT_PROCESSING_CANDIDATE_URLS
    params: tuple = ()
    if limit is not None:
        query += " LIMIT ?"
        params = (limit,)

    with _connect(db_path) as conn:
        urls = [row[0] for row in conn.execute(query, params).fetchall()]

    candidates = []
    for resource_url in urls:
        candidate = enrichment_repository.get_announcement_with_enrichment(
            resource_url, db_path=db_path
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def count_document_processing_candidates(db_path=None) -> int:
    with _connect(db_path) as conn:
        return conn.execute(COUNT_PROCESSING_CANDIDATES).fetchone()[0]


# --- кандидаты backfill extraction ---

BACKFILL_REASON_NO_EXTRACTION = "no_extraction"
BACKFILL_REASON_FAILED_EXTRACTION = "failed_extraction"
BACKFILL_REASON_SKIPPED_SUPPORTED = "skipped_supported_members"

NESTED_MEMBER_DELIMITER = "!/"

SELECT_BACKFILL_DOWNLOADS = """
SELECT
    document_downloads.id AS id,
    document_downloads.resource_url AS resource_url,
    document_downloads.source_kind AS source_kind,
    document_downloads.source_ref AS source_ref,
    document_downloads.filename AS filename,
    document_downloads.saved_path AS saved_path,
    document_downloads.sha256 AS sha256,
    document_downloads.downloaded_at AS downloaded_at,
    (SELECT COUNT(*) FROM document_extractions
        WHERE document_extractions.download_id = document_downloads.id) AS extraction_count,
    (SELECT COUNT(*) FROM document_extractions
        WHERE document_extractions.download_id = document_downloads.id
            AND document_extractions.extraction_status = 'failed') AS failed_extraction_count
FROM document_downloads
ORDER BY document_downloads.id ASC
"""

SELECT_SKIPPED_MEMBERS = (
    "SELECT download_id, member_name FROM document_extractions "
    "WHERE extraction_status = 'skipped' ORDER BY id ASC"
)


@contextmanager
def _connect_read_only(db_path=None):
    """Соединение mode=ro: записать через него нельзя; несуществующая БД не создаётся."""
    path = _resolve_path(db_path).resolve()
    with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as conn:
        yield conn


def _is_supported_skipped_member(member_name: str) -> bool:
    """
    Skipped member, который текущий extractor уже обрабатывает: любой .xlsx и .zip не глубже
    предела вложенности. Прежний extractor пропускал все .xlsx и .zip; .doc, .pdf, .rar, .xml
    по-прежнему не поддерживаются, поэтому причиной backfill не являются. ZIP на предельной
    глубине ("a.zip!/b.zip" при глубине 2) skipped и сейчас: он не должен делать download
    кандидатом навсегда.
    """
    lowered = member_name.lower()
    if lowered.endswith(".xlsx"):
        return True
    if lowered.endswith(".zip"):
        level = lowered.count(NESTED_MEMBER_DELIMITER) + 1
        return level < DEFAULT_MAX_NESTED_DEPTH
    return False


def get_document_extraction_backfill_candidates(db_path=None, limit: int | None = None) -> list[dict]:
    """
    Скачанные документы, extraction которых нужно пересчитать локально (id по возрастанию):
      - нет ни одной extraction (reason "no_extraction");
      - есть extraction_status = "failed" ("failed_extraction");
      - есть skipped member .xlsx или .zip, которые теперь поддерживаются
        ("skipped_supported_members").
    Каждый dict: id, resource_url, source_kind, source_ref, filename, saved_path, sha256,
    downloaded_at, extraction_count, failed_extraction_count, skipped_supported_count,
    reasons (список причин). БД открывается только на чтение; ничего не изменяется.
    limit=None — все; limit — положительное целое, иначе ValueError.
    """
    _validate_limit(limit)

    with _connect_read_only(db_path) as conn:
        conn.row_factory = sqlite3.Row
        downloads = [dict(row) for row in conn.execute(SELECT_BACKFILL_DOWNLOADS).fetchall()]
        skipped = conn.execute(SELECT_SKIPPED_MEMBERS).fetchall()

    skipped_supported: dict[int, int] = {}
    for download_id, member_name in skipped:
        if _is_supported_skipped_member(member_name):
            skipped_supported[download_id] = skipped_supported.get(download_id, 0) + 1

    candidates = []
    for download in downloads:
        download["skipped_supported_count"] = skipped_supported.get(download["id"], 0)
        reasons = []
        if download["extraction_count"] == 0:
            reasons.append(BACKFILL_REASON_NO_EXTRACTION)
        if download["failed_extraction_count"] > 0:
            reasons.append(BACKFILL_REASON_FAILED_EXTRACTION)
        if download["skipped_supported_count"] > 0:
            reasons.append(BACKFILL_REASON_SKIPPED_SUPPORTED)
        if reasons:
            download["reasons"] = reasons
            candidates.append(download)

    return candidates if limit is None else candidates[:limit]


# --- чтение ---

def get_downloads_for_resource(resource_url: str, db_path=None) -> list[dict]:
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM document_downloads WHERE resource_url = ? ORDER BY id ASC",
            (resource_url,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_extractions_for_download(download_id: int, db_path=None) -> list[dict]:
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM document_extractions WHERE download_id = ? ORDER BY id ASC",
            (download_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_latest_download(
    resource_url: str, source_kind: str, source_ref: str, db_path=None
) -> dict | None:
    """Последняя версия логического документа (максимальный id) или None."""
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM document_downloads "
            "WHERE resource_url = ? AND source_kind = ? AND source_ref = ? "
            "ORDER BY id DESC LIMIT 1",
            (resource_url, source_kind, source_ref),
        ).fetchone()
    return dict(row) if row is not None else None


def count_downloads(db_path=None) -> int:
    with _connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM document_downloads").fetchone()[0]


def count_extractions(db_path=None) -> int:
    with _connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM document_extractions").fetchone()[0]
