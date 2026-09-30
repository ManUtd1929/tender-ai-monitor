"""
Append-only локальный ledger фактического AI usage (таблица ai_usage_events) на sqlite3.

Отдельная таблица без связей с business-таблицами анализа: ledger нужен Budget Guard'у и
не должен зависеть от того, сохранился ли результат анализа. Только INSERT: триггеры БД
запрещают UPDATE и DELETE строк (исправление = новая компенсирующая запись).

Одна строка = один завершённый API-вызов, который дал billable usage (в т.ч. если ответ потом
не прошёл validation: success=0 + error_kind). Вызов без usage (сеть, auth, timeout) — не
строка: фейковая стоимость не записывается (record_usage возвращает None).

Деньги: estimated_cost_usd хранится TEXT (str(Decimal)), суммы считаются в Python Decimal,
не float. Стоимость считает pricing.cost_from_usage при записи; pricing_version сохраняется,
чтобы старые записи оставались объяснимыми после смены тарифов. API key никогда не хранится:
в таблице нет ни такого поля, ни произвольных payload'ов.

Схема: колонка cache_write_tokens добавлена позже; _migrate идемпотентно дописывает её в старую
таблицу (старые строки = 0) при каждом подключении. Колонка cached_input_tokens хранит
usage["cached_tokens"] (имя не менялось, чтобы не ломать append-only таблицу).

Время — UTC (ISO, "+00:00"); месяц бюджета = календарный месяц UTC, rollover автоматический
(запросы фильтруют по [начало месяца, начало следующего)).

Durable pre-call reservation (таблица ai_call_reservations): ПЕРЕД каждым платным вызовом в БД
фиксируется резерв с консервативной оценкой стоимости (reserve_call); без успешной записи резерва вызов
не начинается. После ответа settle_reservation в ОДНОЙ транзакции пишет фактический usage в
ai_usage_events и переводит резерв в settled (оценка перестаёт учитываться — двойного счёта нет).
Если запись/завершение не удалось, резерв остаётся reserved/unresolved в БД и переживает перезапуск:
любой нерешённый резерв блокирует новые платные вызовы и входит в effective_committed_spend.
Статусы: reserved (вызов начат/завершение не подтверждено), unresolved (ответ был, учёт не завершён
или биллинг неоднозначен), settled (usage записан), released (известно, что биллинга не было).
Снимается только settle (транзакция с usage), либо явным решением оператора release_reservation.

Fail-closed учёт (таблица ai_accounting_blocks): если платный ответ получен, а его usage НЕ удалось
записать в ledger, оркестратор открывает блок с реальным usage (токены не выдумываются). Пока есть
нерешённый блок, платные вызовы не начинаются (см. analysis_pipeline). Блок снимается ТОЛЬКО
reconcile_accounting_blocks: он записывает сохранённый usage в ledger и помечает блок решённым в одной
транзакции (CLI: python -m src.ai.accounting_reconcile). Автоматического снятия нет.

Evaluator'ы и тесты в ledger не пишут. Любая будущая запись из evaluator/теста обязана
передавать db_path временной БД — production data/tenders.db трогает только production-код.
"""

import json
import logging
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from src.ai import pricing
from src.database.tender_repository import DEFAULT_DB_PATH

logger = logging.getLogger(__name__)

# Значения analysis_type совпадают с src.ai.budget_guard.ANALYSIS_*; ledger их не ограничивает.
ANALYSIS_TRIAGE = "triage"
ANALYSIS_DEEP = "deep"

CREATE_USAGE_TABLE = """
CREATE TABLE IF NOT EXISTS ai_usage_events (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT NOT NULL,
    analysis_type       TEXT NOT NULL,
    model               TEXT NOT NULL,
    reference           TEXT,
    response_id         TEXT,
    input_tokens        INTEGER NOT NULL,
    cached_input_tokens INTEGER NOT NULL,
    cache_write_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens       INTEGER NOT NULL,
    reasoning_tokens    INTEGER NOT NULL,
    estimated_cost_usd  TEXT NOT NULL,
    pricing_version     TEXT NOT NULL,
    prompt_version      TEXT,
    success             INTEGER NOT NULL,
    error_kind          TEXT
)
"""

CREATE_NO_UPDATE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS ai_usage_events_no_update
BEFORE UPDATE ON ai_usage_events
BEGIN SELECT RAISE(ABORT, 'ai_usage_events is append-only'); END
"""

CREATE_NO_DELETE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS ai_usage_events_no_delete
BEFORE DELETE ON ai_usage_events
BEGIN SELECT RAISE(ABORT, 'ai_usage_events is append-only'); END
"""

CREATE_MONTH_INDEX = "CREATE INDEX IF NOT EXISTS idx_ai_usage_events_created_at ON ai_usage_events (created_at)"

CREATE_BLOCKS_TABLE = """
CREATE TABLE IF NOT EXISTS ai_accounting_blocks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL,
    analysis_type   TEXT NOT NULL,
    model           TEXT NOT NULL,
    reference       TEXT,
    response_id     TEXT,
    prompt_version  TEXT,
    success         INTEGER NOT NULL,
    error_kind      TEXT,
    usage_json      TEXT NOT NULL,
    failure         TEXT NOT NULL,
    resolved_at     TEXT,
    resolution      TEXT,
    reservation_id  INTEGER
)
"""

STATUS_RESERVED = "reserved"
STATUS_UNRESOLVED = "unresolved"
STATUS_SETTLED = "settled"
STATUS_RELEASED = "released"
OUTSTANDING_STATUSES = (STATUS_RESERVED, STATUS_UNRESOLVED)

CREATE_RESERVATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS ai_call_reservations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT NOT NULL,
    analysis_type       TEXT NOT NULL,
    reference           TEXT,
    model               TEXT NOT NULL,
    prompt_version      TEXT,
    input_hash          TEXT,
    estimated_cost_usd  TEXT NOT NULL,
    status              TEXT NOT NULL,
    settled_at          TEXT,
    actual_cost_usd     TEXT,
    resolution          TEXT
)
"""

# Не более одного нерешённого резерва на одну и ту же логическую работу (защита от дублей при ретрае).
CREATE_RESERVATION_UNIQUE_INDEX = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_ai_call_reservations_outstanding
ON ai_call_reservations (analysis_type, reference, input_hash, model, prompt_version)
WHERE status IN ('reserved', 'unresolved')
"""


class ReservationBlocked(Exception):
    """Резерв создать нельзя: есть нерешённый резерв (в т.ч. дубль той же работы). Платный вызов запрещён."""

INSERT_EVENT = """
INSERT INTO ai_usage_events (
    created_at, analysis_type, model, reference, response_id, input_tokens,
    cached_input_tokens, cache_write_tokens, output_tokens, reasoning_tokens, estimated_cost_usd,
    pricing_version, prompt_version, success, error_kind
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

SELECT_MONTH = """
SELECT analysis_type, model, input_tokens, cached_input_tokens, output_tokens,
       reasoning_tokens, estimated_cost_usd, cache_write_tokens
FROM ai_usage_events
WHERE created_at >= ? AND created_at < ?
"""

USAGE_TOKEN_FIELDS = ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens")
CACHE_WRITE_ROW_INDEX = 7  # позиция cache_write_tokens в SELECT_MONTH (добавлена в конец)


def _migrate(conn) -> None:
    """
    Идемпотентная миграция старой схемы: добавляет cache_write_tokens (существующие строки = 0).
    Если таблицы ещё нет — ничего не делает (её создаст init_db). ALTER ADD COLUMN не срабатывает
    как UPDATE, поэтому append-only триггеры не мешают и старые строки не переписываются.
    """
    columns = [row[1] for row in conn.execute("PRAGMA table_info(ai_usage_events)")]
    if columns and "cache_write_tokens" not in columns:
        conn.execute("ALTER TABLE ai_usage_events ADD COLUMN cache_write_tokens INTEGER NOT NULL DEFAULT 0")
        logger.info("AI usage ledger: добавлена колонка cache_write_tokens (миграция схемы)")
    block_columns = [row[1] for row in conn.execute("PRAGMA table_info(ai_accounting_blocks)")]
    if block_columns and "reservation_id" not in block_columns:
        conn.execute("ALTER TABLE ai_accounting_blocks ADD COLUMN reservation_id INTEGER")


def _resolve_path(db_path) -> Path:
    return Path(db_path) if db_path is not None else DEFAULT_DB_PATH


@contextmanager
def _connect(db_path=None):
    path = _resolve_path(db_path)
    with closing(sqlite3.connect(path)) as conn:
        with conn:
            _migrate(conn)
            yield conn


def init_db(db_path=None) -> Path:
    path = _resolve_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(path) as conn:
        conn.execute(CREATE_USAGE_TABLE)
        conn.execute(CREATE_NO_UPDATE_TRIGGER)
        conn.execute(CREATE_NO_DELETE_TRIGGER)
        conn.execute(CREATE_MONTH_INDEX)
        conn.execute(CREATE_BLOCKS_TABLE)
        conn.execute(CREATE_RESERVATIONS_TABLE)
        conn.execute(CREATE_RESERVATION_UNIQUE_INDEX)
    return path


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc_iso(moment: datetime) -> str:
    if moment.tzinfo is None:
        raise ValueError("Время события должно быть timezone-aware (UTC)")
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


def _count(value) -> int:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _insert_event(
    conn, analysis_type, model, usage, created_at, reference, response_id, prompt_version, success, error_kind,
    pricing_overrides,
) -> Decimal:
    cost = pricing.cost_from_usage(model, usage, overrides=pricing_overrides)
    conn.execute(INSERT_EVENT, (
        created_at, analysis_type, model, reference, response_id,
        _count(usage.get("input_tokens")), _count(usage.get("cached_tokens")),
        _count(usage.get("cache_write_tokens")),
        _count(usage.get("output_tokens")), _count(usage.get("reasoning_tokens")),
        str(cost), pricing.PRICING_VERSION, prompt_version, 1 if success else 0, error_kind,
    ))
    return cost


def record_usage(
    analysis_type: str,
    model: str,
    usage: dict | None,
    *,
    reference: str | None = None,
    response_id: str | None = None,
    prompt_version: str | None = None,
    success: bool = True,
    error_kind: str | None = None,
    now: datetime | None = None,
    pricing_overrides: dict | None = None,
    db_path=None,
) -> Decimal | None:
    """
    Добавляет одно usage-событие; возвращает его стоимость (Decimal) или None, если usage нет
    (тогда ничего не пишется). usage — dict формата openai_triage.usage_dict.
    pricing.CostEstimationError (неизвестная модель без override) НЕ подавляется: ledger не
    записывает событие с выдуманной или нулевой стоимостью.
    """
    if not analysis_type or not model:
        raise ValueError("analysis_type и model обязательны")
    if usage is None or usage.get("input_tokens") is None and usage.get("output_tokens") is None:
        logger.info("AI usage ledger: usage отсутствует (%s, %s) — событие не записано", analysis_type, model)
        return None

    with _connect(db_path) as conn:
        cost = _insert_event(
            conn, analysis_type, model, usage, _as_utc_iso(now or _utc_now()), reference, response_id,
            prompt_version, success, error_kind, pricing_overrides,
        )
    logger.info("AI usage ledger: %s %s cost=$%s success=%s", analysis_type, model, cost, success)
    return cost


def has_response_id(response_id: str | None, db_path=None) -> bool:
    """Есть ли уже строка ledger с этим response_id (защита от двойной записи одного ответа)."""
    if not response_id:
        return False
    with _connect(db_path) as conn:
        return conn.execute(
            "SELECT 1 FROM ai_usage_events WHERE response_id = ? LIMIT 1", (response_id,)
        ).fetchone() is not None


# --------------------------------------------------------------------------
# fail-closed accounting: платный ответ получен, а ledger его не принял
# --------------------------------------------------------------------------

def open_accounting_block(
    analysis_type: str, model: str, usage: dict, failure: str, *, reference=None, response_id=None,
    prompt_version=None, success=True, error_kind=None, reservation_id=None, db_path=None,
) -> int:
    """
    Фиксирует неучтённый платный usage (реальный, как пришёл от API) для reconcile. Это best-effort
    ДОПОЛНЕНИЕ к резерву: устойчивость обеспечивает сам резерв, а не этот блок. Возвращает id блока.
    """
    with _connect(db_path) as conn:
        conn.execute(CREATE_BLOCKS_TABLE)
        cursor = conn.execute(
            "INSERT INTO ai_accounting_blocks (created_at, analysis_type, model, reference, response_id,"
            " prompt_version, success, error_kind, usage_json, failure, reservation_id)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _as_utc_iso(_utc_now()), analysis_type, model, reference, response_id, prompt_version,
                1 if success else 0, error_kind, json.dumps(usage, sort_keys=True, default=str), failure[:2000],
                reservation_id,
            ),
        )
        return cursor.lastrowid


def unresolved_accounting_blocks(db_path=None) -> list:
    """Нерешённые блоки (dict), старые первыми. Пока список не пуст — платные вызовы запрещены."""
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute(CREATE_BLOCKS_TABLE)
        rows = conn.execute("SELECT * FROM ai_accounting_blocks WHERE resolved_at IS NULL ORDER BY id").fetchall()
    return [dict(row, usage=json.loads(row["usage_json"])) for row in rows]


def reconcile_accounting_blocks(note: str, *, pricing_overrides: dict | None = None, db_path=None) -> list:
    """
    Единственный способ снять блок: для каждого нерешённого блока в ОДНОЙ транзакции записать его
    сохранённый usage в ledger (событие с created_at блока; если response_id уже есть в ledger —
    повторно не пишется) и пометить блок решённым. Любая ошибка откатывает всю транзакцию: блок
    остаётся. Возвращает [{id, cost_usd (Decimal|None), duplicate}].
    """
    if not note or not note.strip():
        raise ValueError("Для reconcile нужна непустая пометка (note)")
    results = []
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute(CREATE_BLOCKS_TABLE)
        rows = conn.execute("SELECT * FROM ai_accounting_blocks WHERE resolved_at IS NULL ORDER BY id").fetchall()
        for row in rows:
            duplicate = bool(row["response_id"]) and conn.execute(
                "SELECT 1 FROM ai_usage_events WHERE response_id = ? LIMIT 1", (row["response_id"],)
            ).fetchone() is not None
            cost = None
            if not duplicate:
                cost = _insert_event(
                    conn, row["analysis_type"], row["model"], json.loads(row["usage_json"]), row["created_at"],
                    row["reference"], row["response_id"], row["prompt_version"], bool(row["success"]),
                    row["error_kind"], pricing_overrides,
                )
            resolved_at = _as_utc_iso(_utc_now())
            if row["reservation_id"] is not None:
                conn.execute(
                    "UPDATE ai_call_reservations SET status = ?, settled_at = ?, actual_cost_usd = ?, resolution = ?"
                    " WHERE id = ? AND status IN ('reserved', 'unresolved')",
                    (STATUS_SETTLED, resolved_at, str(cost) if cost is not None else None,
                     f"reconcile: {note.strip()}", row["reservation_id"]),
                )
            conn.execute(
                "UPDATE ai_accounting_blocks SET resolved_at = ?, resolution = ? WHERE id = ?",
                (resolved_at, note.strip(), row["id"]),
            )
            results.append({"id": row["id"], "cost_usd": cost, "duplicate": duplicate})
    logger.warning("AI accounting: reconcile снял блоков: %d (%s)", len(results), note)
    return results


# --------------------------------------------------------------------------
# durable pre-call reservations
# --------------------------------------------------------------------------

def reserve_call(
    analysis_type: str, model: str, estimated_cost_usd, *, reference=None, prompt_version=None,
    input_hash=None, db_path=None,
) -> int:
    """
    Атомарно (BEGIN IMMEDIATE): если нет нерешённых резервов — записывает новый, статус reserved.
    Возвращает reservation_id (deterministic DB id). ReservationBlocked — есть нерешённый резерв
    (последовательность «проверка + вставка» неделима, поэтому два процесса не пройдут одновременно);
    любая другая ошибка БД пробрасывается. Вызов API разрешён ТОЛЬКО после успешного возврата.
    """
    estimate = Decimal(estimated_cost_usd)
    if estimate < 0:
        raise ValueError("estimated_cost_usd не может быть отрицательной")
    path = _resolve_path(db_path)
    with closing(sqlite3.connect(path)) as conn:
        _migrate(conn)
        conn.execute(CREATE_RESERVATIONS_TABLE)
        conn.execute(CREATE_RESERVATION_UNIQUE_INDEX)
        conn.execute("BEGIN IMMEDIATE")
        try:
            outstanding = conn.execute(
                "SELECT id FROM ai_call_reservations WHERE status IN ('reserved', 'unresolved') ORDER BY id LIMIT 1"
            ).fetchone()
            if outstanding is not None:
                conn.rollback()
                raise ReservationBlocked(f"нерешённый резерв #{outstanding[0]}")
            cursor = conn.execute(
                "INSERT INTO ai_call_reservations (created_at, analysis_type, reference, model, prompt_version,"
                " input_hash, estimated_cost_usd, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (_as_utc_iso(_utc_now()), analysis_type, reference, model, prompt_version, input_hash,
                 str(estimate), STATUS_RESERVED),
            )
            conn.commit()
        except ReservationBlocked:
            raise
        except sqlite3.IntegrityError as error:
            conn.rollback()
            raise ReservationBlocked(f"дубль нерешённого резерва: {error}") from error
        except BaseException:
            conn.rollback()
            raise
        return cursor.lastrowid


def outstanding_reservations(db_path=None) -> list:
    """Нерешённые резервы (reserved / unresolved), старые первыми. Не пусто -> платные вызовы запрещены."""
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute(CREATE_RESERVATIONS_TABLE)
        rows = conn.execute(
            "SELECT * FROM ai_call_reservations WHERE status IN ('reserved', 'unresolved') ORDER BY id"
        ).fetchall()
    return [dict(row) for row in rows]


def outstanding_reserved_cost(now: datetime | None = None, db_path=None) -> Decimal:
    """Сумма консервативных оценок нерешённых резервов, созданных в текущем месяце UTC."""
    moment = (now or _utc_now()).astimezone(timezone.utc)
    start, end = month_bounds(moment.year, moment.month)
    with _connect(db_path) as conn:
        conn.execute(CREATE_RESERVATIONS_TABLE)
        rows = conn.execute(
            "SELECT estimated_cost_usd FROM ai_call_reservations"
            " WHERE status IN ('reserved', 'unresolved') AND created_at >= ? AND created_at < ?",
            (start, end),
        ).fetchall()
    return sum((Decimal(row[0]) for row in rows), Decimal(0))


def effective_committed_spend(now: datetime | None = None, db_path=None) -> Decimal:
    """
    Фактический расход месяца (ledger) + оценки нерешённых резервов месяца. Рассчитанный резерв
    (settled/released) в сумму не входит: после settle учитывается только фактический usage.
    """
    return current_month_cost(now, db_path) + outstanding_reserved_cost(now, db_path)


def settle_reservation(
    reservation_id: int, analysis_type: str, model: str, usage: dict, *, reference=None, response_id=None,
    prompt_version=None, success=True, error_kind=None, pricing_overrides: dict | None = None, db_path=None,
) -> Decimal | None:
    """
    ОДНА транзакция: фактический usage -> ai_usage_events (если такого response_id ещё нет) + резерв ->
    settled. Возвращает стоимость события или None, если response_id уже был в ledger (дубликат:
    повторно не пишется, резерв всё равно закрывается). При любой ошибке (в т.ч. usage без токенов,
    неизвестная цена) откат: резерв остаётся нерешённым. Резерв должен быть нерешённым.
    """
    if not usage or usage.get("input_tokens") is None and usage.get("output_tokens") is None:
        raise ValueError("settle_reservation требует фактический usage (иначе — release/оставить нерешённым)")
    with _connect(db_path) as conn:
        row = conn.execute("SELECT status FROM ai_call_reservations WHERE id = ?", (reservation_id,)).fetchone()
        if row is None or row[0] not in OUTSTANDING_STATUSES:
            raise ValueError(f"Резерв #{reservation_id} не найден или уже закрыт")
        duplicate = bool(response_id) and conn.execute(
            "SELECT 1 FROM ai_usage_events WHERE response_id = ? LIMIT 1", (response_id,)
        ).fetchone() is not None
        cost = None
        if not duplicate:
            cost = _insert_event(
                conn, analysis_type, model, usage, _as_utc_iso(_utc_now()), reference, response_id,
                prompt_version, success, error_kind, pricing_overrides,
            )
        conn.execute(
            "UPDATE ai_call_reservations SET status = ?, settled_at = ?, actual_cost_usd = ? WHERE id = ?",
            (STATUS_SETTLED, _as_utc_iso(_utc_now()), str(cost) if cost is not None else None, reservation_id),
        )
    return cost


def mark_reservation_unresolved(reservation_id: int, note: str, db_path=None) -> None:
    """Best-effort пометка «ответ был, учёт не завершён». Не обязательна: reserved блокирует так же."""
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE ai_call_reservations SET status = ?, resolution = ? WHERE id = ? AND status = ?",
            (STATUS_UNRESOLVED, note[:500], reservation_id, STATUS_RESERVED),
        )


def release_reservation(reservation_id: int, note: str, db_path=None) -> None:
    """
    Снять резерв БЕЗ usage: только когда биллинга точно не было (известный тип ошибки без ответа) или
    по явному решению оператора (accounting_reconcile --release). note обязателен и сохраняется.
    """
    if not note or not note.strip():
        raise ValueError("Для release нужна непустая пометка (note)")
    with _connect(db_path) as conn:
        cursor = conn.execute(
            "UPDATE ai_call_reservations SET status = ?, settled_at = ?, actual_cost_usd = '0', resolution = ?"
            " WHERE id = ? AND status IN ('reserved', 'unresolved')",
            (STATUS_RELEASED, _as_utc_iso(_utc_now()), note.strip(), reservation_id),
        )
        if cursor.rowcount != 1:
            raise ValueError(f"Резерв #{reservation_id} не найден или уже закрыт")


# --------------------------------------------------------------------------
# monthly aggregates (UTC)
# --------------------------------------------------------------------------

def month_bounds(year: int, month: int) -> tuple[str, str]:
    """[начало месяца, начало следующего) в UTC ISO-формате ledger'а."""
    if not 1 <= month <= 12:
        raise ValueError(f"month должен быть 1..12: {month!r}")
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    end = datetime(year + 1, 1, 1, tzinfo=timezone.utc) if month == 12 else datetime(year, month + 1, 1, tzinfo=timezone.utc)
    return _as_utc_iso(start), _as_utc_iso(end)


def _month_rows(year: int, month: int, db_path) -> list:
    start, end = month_bounds(year, month)
    with _connect(db_path) as conn:
        return conn.execute(SELECT_MONTH, (start, end)).fetchall()


def monthly_cost(year: int, month: int, db_path=None) -> Decimal:
    return sum((Decimal(row[6]) for row in _month_rows(year, month, db_path)), Decimal(0))


def monthly_usage(year: int, month: int, db_path=None) -> dict:
    """{events, input_tokens, cached_input_tokens, cache_write_tokens, output_tokens, reasoning_tokens, cost_usd}."""
    rows = _month_rows(year, month, db_path)
    totals = {"events": len(rows)}
    for offset, name in enumerate(USAGE_TOKEN_FIELDS):
        totals[name] = sum(row[2 + offset] for row in rows)
    totals["cache_write_tokens"] = sum(row[CACHE_WRITE_ROW_INDEX] for row in rows)
    totals["cost_usd"] = sum((Decimal(row[6]) for row in rows), Decimal(0))
    return totals


def _cost_grouped_by(year: int, month: int, column: int, db_path) -> dict:
    grouped: dict = {}
    for row in _month_rows(year, month, db_path):
        grouped[row[column]] = grouped.get(row[column], Decimal(0)) + Decimal(row[6])
    return grouped


def cost_by_model(year: int, month: int, db_path=None) -> dict:
    return _cost_grouped_by(year, month, 1, db_path)


def cost_by_analysis_type(year: int, month: int, db_path=None) -> dict:
    return _cost_grouped_by(year, month, 0, db_path)


def current_month_cost(now: datetime | None = None, db_path=None) -> Decimal:
    """Расход текущего месяца по UTC (rollover автоматический)."""
    moment = (now or _utc_now()).astimezone(timezone.utc)
    return monthly_cost(moment.year, moment.month, db_path)
