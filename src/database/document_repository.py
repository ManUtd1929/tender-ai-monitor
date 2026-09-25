"""
Production-хранилище скачанных документов и извлечённого текста (таблицы
document_downloads и document_extractions) на стандартном sqlite3.

document_downloads — metadata скачанных файлов. Логический документ определяется
тройкой (resource_url, source_kind, source_ref); source_ref репозиторий не придумывает,
его передаёт вызывающий код. Если bytes изменились (другой sha256), добавляется новая
строка-версия, старая сохраняется. Тот же sha256 — дубль не создаётся.

document_extractions — результат извлечения текста: по одной строке на
(download_id, member_name). Для обычного DOCX member_name = "", для ZIP — имя member.
Статусы: success, failed, skipped (SQL CHECK намеренно нет).

Таблица announcements здесь только читается (проверка resource_url); её схема
и данные не изменяются. Общий у модулей только файл БД.
"""

import logging
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path

from src.database import announcement_repository
from src.database.tender_repository import DEFAULT_DB_PATH

logger = logging.getLogger(__name__)

STATUS_NEW = announcement_repository.STATUS_NEW
STATUS_UPDATED = announcement_repository.STATUS_UPDATED
STATUS_EXISTING = announcement_repository.STATUS_EXISTING

EXTRACTION_SUCCESS = "success"
EXTRACTION_FAILED = "failed"
EXTRACTION_SKIPPED = "skipped"

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
    "error_type",
    "error_message",
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
    error_type        TEXT,
    error_message     TEXT,
    extracted_at      TEXT NOT NULL,
    FOREIGN KEY (download_id) REFERENCES document_downloads (id) ON DELETE CASCADE,
    UNIQUE (download_id, member_name)
)
"""

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


def init_db(db_path=None) -> Path:
    """Создаёт announcements (если нет) и таблицы документов. Данные не изменяет."""
    path = announcement_repository.init_db(db_path)

    with _connect(path) as conn:
        conn.execute(CREATE_DOWNLOADS_TABLE)
        conn.execute(CREATE_EXTRACTIONS_TABLE)

    logger.info("Таблицы документов готовы: %s", path)
    return path


def _is_blank(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


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


def _docx_extraction(extraction_result: dict, member_name: str) -> dict:
    return {
        "member_name": member_name,
        "file_type": extraction_result.get("file_type"),
        "extraction_status": EXTRACTION_SUCCESS,
        "text": extraction_result.get("text"),
        "char_count": extraction_result.get("char_count"),
        "paragraph_count": extraction_result.get("paragraph_count"),
        "table_count": extraction_result.get("table_count"),
        "error_type": None,
        "error_message": None,
    }


def save_docx_extraction(download_id: int, extraction_result: dict, db_path=None) -> dict:
    """Результат extract_docx() -> одна строка с member_name="" и статусом success."""
    with _connect(db_path) as conn:
        status = _save_extraction(conn, download_id, _docx_extraction(extraction_result, ""))
    return _counts([status])


def save_zip_extraction(download_id: int, extraction_result: dict, db_path=None) -> dict:
    """
    Результат extract_docx_from_zip() -> строки documents (success), skipped_members
    (skipped) и failures (failed). Весь результат сохраняется одной транзакцией:
    при ошибке откатываются все member.
    """
    rows = [
        _docx_extraction(item, item["member_name"])
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
