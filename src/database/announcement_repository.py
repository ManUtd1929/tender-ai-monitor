"""
Production-хранилище объявлений разделов Gnumner (таблица announcements)
на стандартном sqlite3.

Уникальность объявления определяется по resource_url. Если у уже известного
resource_url изменились данные (например, deadline_at), запись обновляется,
а first_seen_at сохраняется.

Таблица tenders (прототип общей страницы, tender_repository.py) здесь не
создаётся, не читается и не изменяется. Общий у модулей только файл БД.
"""

import logging
import sqlite3
import sys
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path

from src.database.tender_repository import DEFAULT_DB_PATH

logger = logging.getLogger(__name__)

STATUS_NEW = "new"
STATUS_UPDATED = "updated"
STATUS_EXISTING = "existing"

REQUIRED_FIELDS = (
    "resource_url",
    "source_section",
    "source_section_name",
    "source_page_url",
    "resource_type",
    "title",
)

# Поля, изменение которых считается обновлением объявления.
COMPARED_FIELDS = (
    "title",
    "source_section",
    "source_section_name",
    "source_page_url",
    "resource_type",
    "published_at",
    "deadline_at",
    "tender_time_raw",
)

CREATE_ANNOUNCEMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS announcements (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    resource_url        TEXT NOT NULL UNIQUE,
    source_section      TEXT NOT NULL,
    source_section_name TEXT NOT NULL,
    source_page_url     TEXT NOT NULL,
    resource_type       TEXT NOT NULL,
    title               TEXT NOT NULL,
    published_at        TEXT,
    deadline_at         TEXT,
    tender_time_raw     TEXT,
    first_seen_at       TEXT NOT NULL,
    last_seen_at        TEXT NOT NULL
)
"""

SELECT_COMPARED = f"SELECT {', '.join(COMPARED_FIELDS)} FROM announcements WHERE resource_url = ?"

INSERT_ANNOUNCEMENT = """
INSERT INTO announcements (
    resource_url, source_section, source_section_name, source_page_url,
    resource_type, title, published_at, deadline_at, tender_time_raw,
    first_seen_at, last_seen_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

UPDATE_ANNOUNCEMENT = f"""
UPDATE announcements
SET {', '.join(f'{name} = ?' for name in COMPARED_FIELDS)}, last_seen_at = ?
WHERE resource_url = ?
"""

TOUCH_ANNOUNCEMENT = "UPDATE announcements SET last_seen_at = ? WHERE resource_url = ?"


def _resolve_path(db_path) -> Path:
    return Path(db_path) if db_path is not None else DEFAULT_DB_PATH


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
        conn.execute(CREATE_ANNOUNCEMENTS_TABLE)

    logger.info("Таблица announcements готова: %s", path)
    return path


def _validate(announcement: dict) -> None:
    missing = [
        name for name in REQUIRED_FIELDS
        if not announcement.get(name)
        or (isinstance(announcement[name], str) and not announcement[name].strip())
    ]
    if missing:
        raise ValueError(f"У объявления отсутствуют обязательные поля: {', '.join(missing)}")


def _save(conn: sqlite3.Connection, announcement: dict) -> str:
    _validate(announcement)

    resource_url = announcement["resource_url"]
    values = tuple(announcement.get(name) for name in COMPARED_FIELDS)
    now = _utc_now()

    row = conn.execute(SELECT_COMPARED, (resource_url,)).fetchone()

    if row is None:
        conn.execute(
            INSERT_ANNOUNCEMENT,
            (
                resource_url,
                announcement["source_section"],
                announcement["source_section_name"],
                announcement["source_page_url"],
                announcement["resource_type"],
                announcement["title"],
                announcement.get("published_at"),
                announcement.get("deadline_at"),
                announcement.get("tender_time_raw"),
                now,
                now,
            ),
        )
        logger.info("Новое объявление сохранено: %s", resource_url)
        return STATUS_NEW

    if tuple(row) == values:
        conn.execute(TOUCH_ANNOUNCEMENT, (now, resource_url))
        logger.info("Объявление уже есть в базе, без изменений: %s", resource_url)
        return STATUS_EXISTING

    changed = [name for name, old, new in zip(COMPARED_FIELDS, row, values) if old != new]
    conn.execute(UPDATE_ANNOUNCEMENT, (*values, now, resource_url))
    logger.info("Объявление обновлено (%s): %s", ", ".join(changed), resource_url)
    return STATUS_UPDATED


def save_announcement(announcement: dict, db_path=None) -> str:
    """Возвращает "new", "updated" или "existing". ValueError — нет обязательного поля."""
    with _connect(db_path) as conn:
        return _save(conn, announcement)


def save_announcements(announcements: list[dict], db_path=None) -> dict:
    new_announcements = []
    updated_announcements = []
    existing_count = 0
    failed_count = 0

    with _connect(db_path) as conn:
        for announcement in announcements:
            try:
                status = _save(conn, announcement)
            except (ValueError, sqlite3.Error):
                failed_count += 1
                logger.exception(
                    "Не удалось сохранить объявление: %s", announcement.get("resource_url")
                )
                continue

            if status == STATUS_NEW:
                new_announcements.append(announcement)
            elif status == STATUS_UPDATED:
                updated_announcements.append(announcement)
            else:
                existing_count += 1

    logger.info(
        "Сохранение объявлений: всего %d, новых %d, обновлённых %d, без изменений %d, ошибок %d",
        len(announcements), len(new_announcements), len(updated_announcements),
        existing_count, failed_count,
    )

    return {
        "new_count": len(new_announcements),
        "updated_count": len(updated_announcements),
        "existing_count": existing_count,
        "failed_count": failed_count,
        "new_announcements": new_announcements,
        "updated_announcements": updated_announcements,
    }


def count_announcements(db_path=None) -> int:
    with _connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM announcements").fetchone()[0]


def get_announcement_by_resource_url(resource_url: str, db_path=None) -> dict | None:
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM announcements WHERE resource_url = ?", (resource_url,)
        ).fetchone()
    return dict(row) if row is not None else None


def _count_legacy_tenders(db_path=None) -> int | None:
    """Только чтение: число строк старой таблицы tenders или None, если её нет."""
    with _connect(db_path) as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'tenders'"
        ).fetchone()
        if exists is None:
            return None
        return conn.execute("SELECT COUNT(*) FROM tenders").fetchone()[0]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    path = init_db()
    print(f"База данных: {path}")
    print(f"Записей в таблице announcements: {count_announcements()}")

    legacy_count = _count_legacy_tenders()
    if legacy_count is None:
        print("Старая таблица tenders: отсутствует (модуль её не создаёт и не изменяет).")
    else:
        print(
            f"Старая таблица tenders: не изменялась, записей: {legacy_count} "
            "(модуль её только читает для этого отчёта)."
        )


if __name__ == "__main__":
    main()
