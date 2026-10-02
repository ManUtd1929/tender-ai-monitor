"""
Отдельный лёгкий процесс интерактивных Telegram-кнопок: long polling getUpdates -> callback_query.

    python -m src.telegram.callback_worker

Не запускает monitor и не использует AI/OpenAI; читает SQLite. Использует TELEGRAM_BOT_TOKEN и
TELEGRAM_CHAT_ID (без них — config error до любых сетевых вызовов). Обслуживает только этот чат.
Offset = update_id + 1 после каждого update, чтобы callback не обрабатывался повторно.
"""

import logging
import sys
import time

from src.database import telegram_delivery_repository as repo
from src.telegram import callbacks, delivery
from src.telegram.client import TelegramClient

logger = logging.getLogger(__name__)

POLL_TIMEOUT_SECONDS = 30
RETRY_BACKOFF_START_SECONDS = 2
RETRY_BACKOFF_MAX_SECONDS = 60


def process_update(client, update: dict, chat_id: str, db_path=None) -> None:
    """Ошибка обработки одного update не должна останавливать worker."""
    callback_query = update.get("callback_query")
    if not isinstance(callback_query, dict):
        return
    try:
        callbacks.handle_callback_query(client, callback_query, chat_id, db_path=db_path)
    except Exception:
        logger.exception("Telegram worker: ошибка обработки update_id=%s", update.get("update_id"))


def run_worker(client, chat_id: str, db_path=None, sleep=time.sleep, should_continue=lambda: True) -> None:
    """Бесконечный long-polling цикл (should_continue — только для тестов)."""
    repo.init_db(db_path)
    offset = None
    backoff = RETRY_BACKOFF_START_SECONDS
    logger.info("Telegram callback worker запущен")
    while should_continue():
        result = client.get_updates(offset=offset, timeout=POLL_TIMEOUT_SECONDS)
        if not result.ok:
            logger.error("Telegram worker: ошибка getUpdates (повтор через %ds): %s", backoff, result.error)
            sleep(backoff)
            backoff = min(backoff * 2, RETRY_BACKOFF_MAX_SECONDS)
            continue
        backoff = RETRY_BACKOFF_START_SECONDS
        for update in result.updates:
            update_id = update.get("update_id")
            if isinstance(update_id, int):
                offset = update_id + 1 if offset is None else max(offset, update_id + 1)
            process_update(client, update, chat_id, db_path=db_path)


def main(environ=None, client_factory=TelegramClient, db_path=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = delivery.load_config(environ)
    if config.missing:
        logger.error("Callback worker не запущен: не заданы: %s", ", ".join(config.missing))
        return 2
    client = client_factory(config.bot_token, config.chat_id)
    try:
        run_worker(client, config.chat_id, db_path=db_path)
    except KeyboardInterrupt:
        logger.info("Telegram callback worker остановлен")
    return 0


if __name__ == "__main__":
    sys.exit(main())
