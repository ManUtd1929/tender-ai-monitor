"""
Хранилище тендеров на стандартном sqlite3.

Уникальность объявления на текущем этапе определяется по attachment_url
(прямая ссылка на прикреплённый файл, есть у каждого объявления на сайте).
"""

import logging
import sqlite3
import sys
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "tenders.db"

REQUIRED_FIELDS = ("title", "attachment_url", "filename")

CREATE_TENDERS_TABLE = """
CREATE TABLE IF NOT EXISTS tenders (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    title           TEXT NOT NULL,
    attachment_url  TEXT NOT NULL UNIQUE,
    filename        TEXT NOT NULL,
    file_type       TEXT,
    published_at    TEXT,
    tender_time_raw TEXT,
    first_seen_at   TEXT NOT NULL
)
"""

INSERT_TENDER = """
INSERT OR IGNORE INTO tenders (
    title, attachment_url, filename, file_type,
    published_at, tender_time_raw, first_seen_at
) VALUES (?, ?, ?, ?, ?, ?, ?)
"""


def _resolve_path(db_path) -> Path:
    return Path(db_path) if db_path is not None else DEFAULT_DB_PATH


@contextmanager
def _connect(db_path=None):
    """Открывает соединение, делает commit/rollback и гарантированно закрывает его."""
    path = _resolve_path(db_path)
    with closing(sqlite3.connect(path)) as conn:
        with conn:
            yield conn


def init_db(db_path=None) -> Path:
    path = _resolve_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with _connect(path) as conn:
        conn.execute(CREATE_TENDERS_TABLE)

    logger.info("База данных инициализирована: %s", path)
    return path


def _insert_tender(conn: sqlite3.Connection, tender: dict) -> bool:
    # INSERT OR IGNORE молча пропускает и нарушение NOT NULL, поэтому
    # обязательные поля проверяем явно, чтобы не принять ошибку за дубликат.
    missing = [name for name in REQUIRED_FIELDS if not tender.get(name)]
    if missing:
        raise ValueError(f"У тендера отсутствуют обязательные поля: {', '.join(missing)}")

    first_seen_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    cursor = conn.execute(
        INSERT_TENDER,
        (
            tender["title"],
            tender["attachment_url"],
            tender["filename"],
            tender.get("file_type"),
            tender.get("published_at"),
            tender.get("tender_time_raw"),
            first_seen_at,
        ),
    )
    is_new = cursor.rowcount == 1

    if is_new:
        logger.info("Новый тендер сохранён: %s", tender["attachment_url"])
    else:
        logger.info("Тендер уже есть в базе, пропуск: %s", tender["attachment_url"])
    return is_new


def save_tender(tender: dict, db_path=None) -> bool:
    with _connect(db_path) as conn:
        return _insert_tender(conn, tender)


def save_tenders(tenders: list[dict], db_path=None) -> list[dict]:
    new_tenders = []

    with _connect(db_path) as conn:
        for tender in tenders:
            try:
                if _insert_tender(conn, tender):
                    new_tenders.append(tender)
            except (ValueError, sqlite3.Error):
                logger.exception(
                    "Не удалось сохранить тендер: %s", tender.get("attachment_url")
                )

    logger.info("Сохранено новых тендеров: %d из %d", len(new_tenders), len(tenders))
    return new_tenders


def count_tenders(db_path=None) -> int:
    with _connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM tenders").fetchone()[0]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    path = init_db()
    print(f"База данных: {path}")
    print(f"Записей в таблице tenders: {count_tenders()}")


if __name__ == "__main__":
    main()
