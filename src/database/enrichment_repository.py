"""
Production-хранилище enrichment для объявлений (таблицы announcement_enrichment,
announcement_cpv, announcement_documents) на стандартном sqlite3.

Данные приходят из src/scraper/resource_enrichment.py (enrich_announcement).
Таблица announcements здесь только читается: модуль проверяет, что объявление
существует, но не изменяет его. Общий у модулей только файл БД.

enrichment_processing_state — итог последней попытки enrichment объявления (по одной строке
на resource_url): success или failed (SQL CHECK намеренно нет). Она нужна потому, что при
неудачном повторном enrichment обновлённого объявления старая валидная строка
announcement_enrichment остаётся, и по одной этой таблице «refresh не удался» не понять.
По состоянию get_enrichment_processing_candidates находит объявления для повторной попытки.

Семантика необязательных списков:
- ключа "cpv_codes" / "documents" нет (или значение None) -> старые строки не трогаем;
- пустой список []                                         -> старые строки удаляем;
- непустой список                                          -> заменяем старые строки.
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

PROCESSING_SUCCESS = "success"
PROCESSING_FAILED = "failed"

# Скалярные поля enrichment, которые сохраняются как есть.
SCALAR_FIELDS = (
    "enrichment_status",
    "procedure_code",
    "contracting_authority",
    "detail_title",
    "detail_title_ru",
    "detail_title_en",
    "procurement_type",
    "procedure_type",
    "description",
    "estimated_value_amd",
    "published_at_detail",
    "deadline_at_detail",
    "number_of_lots",
    "resource_id",
    "document_url",
)

# Поля из enrichment["consistency"]; в БД хранятся как 1 / 0 / NULL.
MATCH_FIELDS = ("published_at_match", "deadline_at_match")

# Поля, изменение которых считается обновлением (enriched_at сюда не входит).
COMPARED_FIELDS = SCALAR_FIELDS + MATCH_FIELDS

DOCUMENT_FIELDS = ("document_id", "filename", "language", "title", "description")

# Поля announcement, которые ожидает enrich_announcement (id и first/last_seen_at не нужны).
ANNOUNCEMENT_FIELDS = (
    "title",
    "source_section",
    "source_section_name",
    "source_page_url",
    "resource_url",
    "resource_type",
    "published_at",
    "deadline_at",
    "tender_time_raw",
)

CREATE_ENRICHMENT_TABLE = """
CREATE TABLE IF NOT EXISTS announcement_enrichment (
    resource_url          TEXT PRIMARY KEY,
    enrichment_status     TEXT NOT NULL,
    procedure_code        TEXT,
    contracting_authority TEXT,
    detail_title          TEXT,
    detail_title_ru       TEXT,
    detail_title_en       TEXT,
    procurement_type      TEXT,
    procedure_type        TEXT,
    description           TEXT,
    estimated_value_amd   TEXT,
    published_at_detail   TEXT,
    deadline_at_detail    TEXT,
    number_of_lots        INTEGER,
    resource_id           TEXT,
    document_url          TEXT,
    published_at_match    INTEGER,
    deadline_at_match     INTEGER,
    enriched_at           TEXT NOT NULL,
    FOREIGN KEY (resource_url) REFERENCES announcements (resource_url)
)
"""

CREATE_CPV_TABLE = """
CREATE TABLE IF NOT EXISTS announcement_cpv (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    resource_url TEXT NOT NULL,
    code         TEXT NOT NULL,
    name         TEXT,
    FOREIGN KEY (resource_url) REFERENCES announcements (resource_url),
    UNIQUE (resource_url, code, name)
)
"""

CREATE_DOCUMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS announcement_documents (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    resource_url TEXT NOT NULL,
    document_id  TEXT,
    filename     TEXT NOT NULL,
    language     TEXT,
    title        TEXT,
    description  TEXT,
    FOREIGN KEY (resource_url) REFERENCES announcements (resource_url),
    UNIQUE (resource_url, document_id, filename)
)
"""

SELECT_SCALARS = f"SELECT {', '.join(COMPARED_FIELDS)} FROM announcement_enrichment WHERE resource_url = ?"

INSERT_ENRICHMENT = f"""
INSERT INTO announcement_enrichment (
    resource_url, {', '.join(COMPARED_FIELDS)}, enriched_at
) VALUES ({', '.join('?' * (len(COMPARED_FIELDS) + 2))})
"""

UPDATE_ENRICHMENT = f"""
UPDATE announcement_enrichment
SET {', '.join(f'{name} = ?' for name in COMPARED_FIELDS)}, enriched_at = ?
WHERE resource_url = ?
"""

CREATE_PROCESSING_STATE_TABLE = """
CREATE TABLE IF NOT EXISTS enrichment_processing_state (
    resource_url    TEXT PRIMARY KEY,
    status          TEXT NOT NULL,
    last_attempt_at TEXT NOT NULL,
    error_type      TEXT,
    error_message   TEXT,
    FOREIGN KEY (resource_url) REFERENCES announcements (resource_url)
)
"""

UPSERT_PROCESSING_STATE = """
INSERT INTO enrichment_processing_state (
    resource_url, status, last_attempt_at, error_type, error_message
) VALUES (?, ?, ?, ?, ?)
ON CONFLICT (resource_url) DO UPDATE SET
    status = excluded.status,
    last_attempt_at = excluded.last_attempt_at,
    error_type = excluded.error_type,
    error_message = excluded.error_message
"""

SELECT_CPV = "SELECT code, name FROM announcement_cpv WHERE resource_url = ? ORDER BY id"
DELETE_CPV = "DELETE FROM announcement_cpv WHERE resource_url = ?"
INSERT_CPV = "INSERT INTO announcement_cpv (resource_url, code, name) VALUES (?, ?, ?)"

SELECT_DOCUMENTS = (
    f"SELECT {', '.join(DOCUMENT_FIELDS)} FROM announcement_documents "
    "WHERE resource_url = ? ORDER BY id"
)
FROM_WITHOUT_ENRICHMENT = """
FROM announcements
LEFT JOIN announcement_enrichment
    ON announcement_enrichment.resource_url = announcements.resource_url
WHERE announcement_enrichment.resource_url IS NULL
"""

SELECT_WITHOUT_ENRICHMENT = (
    f"SELECT {', '.join(f'announcements.{name}' for name in ANNOUNCEMENT_FIELDS)} "
    f"{FROM_WITHOUT_ENRICHMENT} "
    "ORDER BY announcements.first_seen_at ASC, announcements.id ASC"
)

COUNT_WITHOUT_ENRICHMENT = f"SELECT COUNT(*) {FROM_WITHOUT_ENRICHMENT}"

# Кандидаты на enrichment: enrichment нет вовсе (state не важен) ИЛИ последняя попытка
# failed (обновлённое объявление с прежним enrichment). state = success с существующим
# enrichment кандидатом не является.
FROM_PROCESSING_CANDIDATES = """
FROM announcements
LEFT JOIN announcement_enrichment
    ON announcement_enrichment.resource_url = announcements.resource_url
LEFT JOIN enrichment_processing_state
    ON enrichment_processing_state.resource_url = announcements.resource_url
WHERE announcement_enrichment.resource_url IS NULL
    OR enrichment_processing_state.status = 'failed'
"""

# Сначала без enrichment, затем failed refresh: старые failed не блокируют новые объявления.
SELECT_PROCESSING_CANDIDATES = (
    f"SELECT {', '.join(f'announcements.{name}' for name in ANNOUNCEMENT_FIELDS)} "
    f"{FROM_PROCESSING_CANDIDATES} "
    "ORDER BY CASE WHEN announcement_enrichment.resource_url IS NULL THEN 0 ELSE 1 END ASC, "
    "announcements.first_seen_at ASC, announcements.id ASC"
)

COUNT_PROCESSING_CANDIDATES = f"SELECT COUNT(*) {FROM_PROCESSING_CANDIDATES}"

DELETE_DOCUMENTS = "DELETE FROM announcement_documents WHERE resource_url = ?"
INSERT_DOCUMENT = f"""
INSERT INTO announcement_documents (resource_url, {', '.join(DOCUMENT_FIELDS)})
VALUES ({', '.join('?' * (len(DOCUMENT_FIELDS) + 1))})
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
    """Создаёт announcements (если нет), три таблицы enrichment и таблицу состояния. Данные не изменяет."""
    path = announcement_repository.init_db(db_path)

    with _connect(path) as conn:
        conn.execute(CREATE_ENRICHMENT_TABLE)
        conn.execute(CREATE_CPV_TABLE)
        conn.execute(CREATE_DOCUMENTS_TABLE)
        conn.execute(CREATE_PROCESSING_STATE_TABLE)

    logger.info("Таблицы enrichment готовы: %s", path)
    return path


def _bool_to_db(value) -> int | None:
    return None if value is None else int(bool(value))


def _bool_from_db(value) -> bool | None:
    return None if value is None else bool(value)


def _scalar_values(enrichment: dict) -> tuple:
    """Значения COMPARED_FIELDS в порядке COMPARED_FIELDS (совпадения -> 1/0/NULL)."""
    consistency = enrichment.get("consistency") or {}
    scalars = [enrichment.get(name) for name in SCALAR_FIELDS]
    matches = [_bool_to_db(consistency.get(name)) for name in MATCH_FIELDS]
    return tuple(scalars + matches)


def _is_blank(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _prepare_cpv(items: list, resource_url: str) -> list[tuple]:
    """Список (code, name) без дублей и без записей без code (code NOT NULL)."""
    prepared = []
    for item in items:
        code = item.get("code")
        if _is_blank(code):
            logger.warning("CPV без code пропущен (%s): %r", resource_url, item)
            continue
        entry = (code, item.get("name"))
        if entry not in prepared:
            prepared.append(entry)
    return prepared


def _prepare_documents(items: list, resource_url: str) -> list[tuple]:
    """Список кортежей DOCUMENT_FIELDS без дублей и без документов без filename."""
    prepared = []
    for item in items:
        if _is_blank(item.get("filename")):
            logger.warning("Документ без filename пропущен (%s): %r", resource_url, item)
            continue
        entry = tuple(item.get(name) for name in DOCUMENT_FIELDS)
        # UNIQUE в SQLite не считает NULL равными, поэтому дубли отсекаются здесь.
        if not any(entry[0] == other[0] and entry[1] == other[1] for other in prepared):
            prepared.append(entry)
    return prepared


def _save(conn: sqlite3.Connection, resource_url: str, enrichment: dict) -> str:
    if _is_blank(resource_url):
        raise ValueError("resource_url не может быть пустым")
    if _is_blank(enrichment.get("enrichment_status")):
        raise ValueError(f"У enrichment отсутствует enrichment_status: {resource_url}")

    exists = conn.execute(
        "SELECT 1 FROM announcements WHERE resource_url = ?", (resource_url,)
    ).fetchone()
    if exists is None:
        raise ValueError(f"Объявление не найдено в announcements: {resource_url}")

    values = _scalar_values(enrichment)
    now = _utc_now()

    # None вместо списка (как у парсера, когда CPV не найдены) — «не передано», а не «пусто».
    cpv = enrichment.get("cpv_codes")
    cpv = _prepare_cpv(cpv, resource_url) if cpv is not None else None
    documents = enrichment.get("documents")
    documents = _prepare_documents(documents, resource_url) if documents is not None else None

    row = conn.execute(SELECT_SCALARS, (resource_url,)).fetchone()
    changed = []

    if row is None:
        conn.execute(INSERT_ENRICHMENT, (resource_url, *values, now))
    else:
        changed = [name for name, old, new in zip(COMPARED_FIELDS, row, values) if old != new]
        conn.execute(UPDATE_ENRICHMENT, (*values, now, resource_url))

    if cpv is not None:
        old_cpv = [tuple(r) for r in conn.execute(SELECT_CPV, (resource_url,))]
        if old_cpv != cpv:
            conn.execute(DELETE_CPV, (resource_url,))
            conn.executemany(INSERT_CPV, [(resource_url, code, name) for code, name in cpv])
            changed.append("cpv_codes")

    if documents is not None:
        old_documents = [tuple(r) for r in conn.execute(SELECT_DOCUMENTS, (resource_url,))]
        if old_documents != documents:
            conn.execute(DELETE_DOCUMENTS, (resource_url,))
            conn.executemany(INSERT_DOCUMENT, [(resource_url, *doc) for doc in documents])
            changed.append("documents")

    if row is None:
        logger.info("Enrichment сохранён (new): %s", resource_url)
        return STATUS_NEW
    if changed:
        logger.info("Enrichment обновлён (%s): %s", ", ".join(changed), resource_url)
        return STATUS_UPDATED

    logger.info("Enrichment без изменений: %s", resource_url)
    return STATUS_EXISTING


def save_enrichment(resource_url: str, enrichment: dict, db_path=None) -> str:
    """
    Возвращает "new", "updated" или "existing". Скаляры, CPV и документы
    сохраняются одной транзакцией; при ошибке откатывается всё.
    ValueError — пустой resource_url, нет enrichment_status или нет объявления.
    """
    with _connect(db_path) as conn:
        return _save(conn, resource_url, enrichment)


def get_enrichment(resource_url: str, db_path=None) -> dict | None:
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM announcement_enrichment WHERE resource_url = ?", (resource_url,)
        ).fetchone()
        if row is None:
            return None

        cpv_rows = conn.execute(SELECT_CPV, (resource_url,)).fetchall()
        document_rows = conn.execute(SELECT_DOCUMENTS, (resource_url,)).fetchall()

    result = {"resource_url": row["resource_url"]}
    result.update({name: row[name] for name in SCALAR_FIELDS})
    result["enriched_at"] = row["enriched_at"]
    result["consistency"] = {name: _bool_from_db(row[name]) for name in MATCH_FIELDS}
    result["cpv_codes"] = [{"code": r["code"], "name": r["name"]} for r in cpv_rows]
    result["documents"] = [{name: r[name] for name in DOCUMENT_FIELDS} for r in document_rows]
    return result


def get_announcement_with_enrichment(resource_url: str, db_path=None) -> dict | None:
    """
    Объединённый объект для document pipeline: поля announcements (в т.ч. resource_type,
    которого нет в get_enrichment) + скалярные поля enrichment + consistency, cpv_codes,
    documents. Без enrichment (или без объявления) возвращает None.
    Поля announcement и enrichment не пересекаются, кроме resource_url (значение одно).
    """
    enrichment = get_enrichment(resource_url, db_path=db_path)
    if enrichment is None:
        return None

    announcement = announcement_repository.get_announcement_by_resource_url(
        resource_url, db_path=db_path
    )
    if announcement is None:
        return None

    result = {name: announcement[name] for name in ANNOUNCEMENT_FIELDS}
    result.update({name: enrichment[name] for name in SCALAR_FIELDS})
    result["consistency"] = enrichment["consistency"]
    result["cpv_codes"] = enrichment["cpv_codes"]
    result["documents"] = enrichment["documents"]
    return result


def count_enrichments(db_path=None) -> int:
    with _connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM announcement_enrichment").fetchone()[0]


def _select_announcements(query: str, limit: int | None, db_path) -> list[dict]:
    """Выполняет запрос с необязательным parameterized LIMIT; dict совместим с enrich_announcement."""
    params: tuple = ()
    if limit is not None:
        # bool — подкласс int, но True/False как лимит — почти наверняка ошибка.
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError(f"limit должен быть положительным целым или None: {limit!r}")
        query += " LIMIT ?"
        params = (limit,)

    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()

    return [{name: row[name] for name in ANNOUNCEMENT_FIELDS} for row in rows]


def get_announcements_without_enrichment(db_path=None, limit: int | None = None) -> list[dict]:
    """
    Объявления, для которых строки в announcement_enrichment ещё нет (самые старые первыми).
    Любой статус enrichment (success, partial, not_required, unsupported) — не pending.
    Каждый dict совместим с enrich_announcement(announcement).
    limit=None — все; limit — положительное целое, иначе ValueError.
    """
    return _select_announcements(SELECT_WITHOUT_ENRICHMENT, limit, db_path)


def count_announcements_without_enrichment(db_path=None) -> int:
    with _connect(db_path) as conn:
        return conn.execute(COUNT_WITHOUT_ENRICHMENT).fetchone()[0]


def save_enrichment_processing_state(
    resource_url: str,
    status: str,
    error_type: str | None = None,
    error_message: str | None = None,
    db_path=None,
) -> None:
    """
    Upsert по resource_url; last_attempt_at — текущее UTC-время (timezone-aware).
    Все поля строки перезаписываются: успешная попытка очищает прежнюю ошибку.
    announcement_enrichment не изменяется (старый валидный enrichment сохраняется).
    ValueError — пустой resource_url/status или нет объявления в announcements.
    """
    if _is_blank(resource_url):
        raise ValueError("resource_url не может быть пустым")
    if _is_blank(status):
        raise ValueError(f"У состояния enrichment отсутствует status: {resource_url}")

    with _connect(db_path) as conn:
        exists = conn.execute(
            "SELECT 1 FROM announcements WHERE resource_url = ?", (resource_url,)
        ).fetchone()
        if exists is None:
            raise ValueError(f"Объявление не найдено в announcements: {resource_url}")

        conn.execute(
            UPSERT_PROCESSING_STATE,
            (resource_url, status, _utc_now(), error_type, error_message),
        )

    logger.info("Состояние enrichment сохранено (%s): %s", status, resource_url)


def get_enrichment_processing_state(resource_url: str, db_path=None) -> dict | None:
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM enrichment_processing_state WHERE resource_url = ?", (resource_url,)
        ).fetchone()
    return dict(row) if row is not None else None


def get_enrichment_processing_candidates(db_path=None, limit: int | None = None) -> list[dict]:
    """
    Объявления, которым нужен (повторный) enrichment: enrichment отсутствует (любое состояние)
    или последняя попытка failed. Сначала без enrichment, затем failed refresh; внутри группы
    самые старые первыми (first_seen_at, id). Объявления со state = success и существующим
    enrichment не возвращаются. Каждый dict совместим с enrich_announcement(announcement).
    limit=None — все; limit — положительное целое, иначе ValueError.
    """
    return _select_announcements(SELECT_PROCESSING_CANDIDATES, limit, db_path)


def count_enrichment_processing_candidates(db_path=None) -> int:
    with _connect(db_path) as conn:
        return conn.execute(COUNT_PROCESSING_CANDIDATES).fetchone()[0]
