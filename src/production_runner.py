"""
Production runner для одного persistent service (Railway): один процесс, одна SQLite БД.

    python -m src.production_runner

MAIN THREAD: Telegram callback long-poll worker (callback_worker.run_worker).
BACKGROUND THREAD (не daemon): scheduler, вызывающий monitor.run_monitor сразу после старта
и затем каждые MONITOR_INTERVAL_MINUTES (фиксированная сетка тиков). Если run ещё идёт, когда
наступил следующий тик, этот тик пропускается (очереди пропущенных запусков нет).

Монитор, AI и Telegram-доставка не меняются: runner только вызывает существующие функции.
Секреты (токен, ключи) в логи не попадают.
"""

import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from src import monitor
from src.database import (
    announcement_repository,
    document_repository,
    enrichment_repository,
    telegram_delivery_repository,
)
from src.database import tender_repository
from src.telegram import callback_worker, delivery
from src.telegram.client import TelegramClient

logger = logging.getLogger(__name__)

INTERVAL_ENV = "MONITOR_INTERVAL_MINUTES"
OVERLAP_WARNING = "Предыдущий monitor run ещё выполняется, запуск пропущен"
SHUTDOWN_JOIN_SECONDS = 60

EXIT_OK = 0
EXIT_FATAL = 1
EXIT_CONFIG_ERROR = 2


class ConfigError(Exception):
    """Ошибка конфигурации runner; сообщение не содержит значений секретов."""


@dataclass(frozen=True)
class RunnerConfig:
    interval_minutes: int
    db_path: Path
    bot_token: str
    chat_id: str

    def __repr__(self) -> str:  # токен не попадает в логи/traceback
        return (
            f"RunnerConfig(interval_minutes={self.interval_minutes}, db_path={str(self.db_path)!r}, "
            f"bot_token='***', chat_id={self.chat_id!r})"
        )


def parse_interval_minutes(raw) -> int:
    text = (raw or "").strip()
    try:
        value = int(text)
    except ValueError:
        raise ConfigError(f"{INTERVAL_ENV} должен быть целым числом > 0") from None
    if value <= 0:
        raise ConfigError(f"{INTERVAL_ENV} должен быть целым числом > 0")
    return value


def load_config(environ=None, monitor_db_path=tender_repository.DEFAULT_DB_PATH) -> RunnerConfig:
    """
    Проверка конфигурации до любых сетевых вызовов. Без молчаливых значений по умолчанию.

    monitor_db_path — БД, в которую реально пишут репозитории monitor/AI/Telegram (они пока используют
    DEFAULT_DB_PATH и не читают DATABASE_PATH). Если DATABASE_PATH указывает на другой файл, callback
    worker и monitor работали бы с разными БД, поэтому запуск отклоняется.
    """
    env = os.environ if environ is None else environ
    problems = []

    interval = None
    try:
        interval = parse_interval_minutes(env.get(INTERVAL_ENV))
    except ConfigError as error:
        problems.append(str(error))

    db_path = tender_repository.resolve_db_path(environ=env)
    if Path(db_path) != Path(monitor_db_path):
        problems.append(
            f"DATABASE_PATH ({db_path}) не совпадает с БД, которую использует monitor ({monitor_db_path})"
        )

    telegram = delivery.load_config(env)
    if telegram.missing:
        problems.append(f"не заданы: {', '.join(telegram.missing)}")

    if problems:
        raise ConfigError("; ".join(problems))
    return RunnerConfig(interval, Path(db_path), telegram.bot_token, telegram.chat_id)


def init_databases(db_path) -> None:
    """Схемы через существующие init_db (до запуска потоков, чтобы не гонять CREATE TABLE)."""
    announcement_repository.init_db(db_path)
    enrichment_repository.init_db(db_path)
    document_repository.init_db(db_path)
    telegram_delivery_repository.init_db(db_path)  # включает pipeline_state и analysis


def _compact_result(result) -> str:
    if not isinstance(result, dict):
        return "n/a"
    keys = ("fetched_count", "new_count", "updated_count", "failed_count", "total_count")
    return ", ".join(f"{key}={result[key]}" for key in keys if key in result) or "n/a"


class MonitorScheduler:
    """Периодический запуск monitor_fn; один процесс-локальный Lock исключает пересечение run'ов."""

    def __init__(self, monitor_fn, interval_seconds: float, stop_event: threading.Event,
                 clock=time.monotonic, wait=None):
        self._monitor_fn = monitor_fn
        self._interval = interval_seconds
        self._stop = stop_event
        self._clock = clock
        self._wait = wait or stop_event.wait  # прерываемый sleep: возвращает True, если stop установлен
        self._lock = threading.Lock()

    def run_once(self) -> bool:
        """True, если run выполнился; False, если предыдущий ещё идёт. Исключения monitor не пробрасываются."""
        if not self._lock.acquire(blocking=False):
            logger.warning(OVERLAP_WARNING)
            return False
        started = self._clock()
        try:
            logger.info("Monitor run started")
            try:
                result = self._monitor_fn()
            except Exception:
                logger.exception("Monitor run завершился ошибкой; следующий запуск по расписанию")
                return True
            logger.info(
                "Monitor run finished: duration=%.1fs, %s", self._clock() - started, _compact_result(result)
            )
            return True
        finally:
            self._lock.release()

    def run_forever(self) -> None:
        next_tick = self._clock()  # первый run сразу после старта
        while not self._stop.is_set():
            self.run_once()
            next_tick += self._interval
            while next_tick <= self._clock():  # тики, наступившие во время run'а, пропускаются без очереди
                logger.warning(OVERLAP_WARNING)
                next_tick += self._interval
            if self._wait(max(0.0, next_tick - self._clock())):
                break
        logger.info("Monitor scheduler остановлен")


def run(config: RunnerConfig, monitor_fn=monitor.run_monitor, worker_fn=callback_worker.run_worker,
        client_factory=TelegramClient, init_fn=init_databases, stop_event=None, clock=time.monotonic,
        wait=None, shutdown_join_seconds=SHUTDOWN_JOIN_SECONDS) -> int:
    """Возвращает exit code: 0 — штатная остановка, 1 — фатальная ошибка callback worker."""
    stop = stop_event or threading.Event()
    init_fn(config.db_path)

    logger.info(
        "Production runner started: monitor interval=%d min, database=%s, Telegram callback worker enabled",
        config.interval_minutes, config.db_path,
    )
    scheduler = MonitorScheduler(monitor_fn, config.interval_minutes * 60, stop, clock=clock, wait=wait)
    thread = threading.Thread(target=scheduler.run_forever, name="monitor-scheduler")  # не daemon
    thread.start()

    exit_code = EXIT_OK
    try:
        client = client_factory(config.bot_token, config.chat_id)
        worker_fn(client, config.chat_id, db_path=config.db_path, should_continue=lambda: not stop.is_set())
    except KeyboardInterrupt:
        logger.info("Получен сигнал остановки")
    except Exception:
        logger.exception("Telegram callback worker завершился фатальной ошибкой")
        exit_code = EXIT_FATAL
    finally:
        stop.set()
        thread.join(timeout=shutdown_join_seconds)
        if thread.is_alive():
            logger.warning("Monitor run не завершился за %ds; ожидание завершения без обрыва записи", shutdown_join_seconds)
            thread.join()
    logger.info("Production runner остановлен (exit code %d)", exit_code)
    return exit_code


def main(environ=None, **run_kwargs) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if environ is None:
        from dotenv import load_dotenv

        load_dotenv(monitor.PROJECT_ROOT / ".env")  # не перезаписывает уже заданные переменные окружения
    try:
        config = load_config(environ)
    except ConfigError as error:
        logger.error("Production runner не запущен: %s", error)
        return EXIT_CONFIG_ERROR

    stop = run_kwargs.setdefault("stop_event", threading.Event())
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, lambda signum, frame: stop.set())
    return run(config, **run_kwargs)


if __name__ == "__main__":
    sys.exit(main())
