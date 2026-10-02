"""
Telegram delivery stage: читает из SQLite уже готовые deep results и отправляет карточки.

Никакого AI/OpenAI: модуль не импортирует src.ai.analysis_pipeline и не создаёт analyzers; из src.ai
берутся только чистые детерминированные функции (tender_context, operational_eligibility, value_facts).

Правило отбора: pipeline state = deep_completed; сохранённый deep result существует, не пуст и его
input_hash совпадает и с состоянием, и с ТЕКУЩИМ input_hash tender_context (иначе результат устарел);
версия (resource_url, input_hash) ещё не sent; срок однозначно не истёк (Eligibility.expired).
Каждая версия отправляется максимум один раз за запуск; failed повторяется только в следующем запуске.
"""

import logging
import os
from dataclasses import dataclass

from src.ai import operational_eligibility, tender_context as tender_context_module, value_facts
from src.database import analysis_repository, telegram_delivery_repository as repo
from src.telegram import message

logger = logging.getLogger(__name__)

ENABLED_ENV = "TELEGRAM_NOTIFICATIONS_ENABLED"
TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
CHAT_ID_ENV = "TELEGRAM_CHAT_ID"
_TRUE_VALUES = ("1", "true", "yes", "on")


@dataclass(frozen=True)
class TelegramConfig:
    enabled: bool
    bot_token: str = ""
    chat_id: str = ""

    @property
    def missing(self) -> list[str]:
        return [name for name, value in ((TOKEN_ENV, self.bot_token), (CHAT_ID_ENV, self.chat_id)) if not value]

    def __repr__(self) -> str:  # токен не попадает в логи/traceback
        return f"TelegramConfig(enabled={self.enabled}, bot_token={'***' if self.bot_token else ''!r}, chat_id={self.chat_id!r})"


def load_config(environ=None) -> TelegramConfig:
    env = os.environ if environ is None else environ
    return TelegramConfig(
        enabled=(env.get(ENABLED_ENV) or "").strip().lower() in _TRUE_VALUES,
        bot_token=(env.get(TOKEN_ENV) or "").strip(),
        chat_id=(env.get(CHAT_ID_ENV) or "").strip(),
    )


def _format_deadline(eligibility) -> str | None:
    if eligibility.resolved_deadline is None:
        return None
    return eligibility.resolved_deadline.astimezone(operational_eligibility.LOCAL_TZ).strftime("%d.%m.%Y %H:%M")


def _official_value(context: dict) -> int | None:
    facts = value_facts.extract_value_facts(context)
    return facts["estimated_value_amd"]  # только official_tender_value / sum_of_official_lot_values


def _deliver_one(client, url: str, state_hash: str, db_path, clock, summary: dict) -> None:
    deep = analysis_repository.get_deep_analysis(url, db_path=db_path)
    result = deep["result"] if deep else None
    if not isinstance(result, dict) or not message._clean(result.get("summary")):
        logger.info("Telegram: нет полного deep result, пропуск: %s", url)
        summary["skipped_incomplete"] += 1
        return
    input_hash = deep["input_hash"]
    if input_hash != state_hash:
        logger.info("Telegram: deep result не соответствует состоянию пайплайна, пропуск: %s", url)
        summary["skipped_stale"] += 1
        return

    existing = repo.get_delivery(url, input_hash, db_path=db_path)
    if existing and existing["status"] == repo.STATUS_SENT:
        logger.info("Telegram: версия уже отправлена, пропуск: %s", url)
        summary["skipped_already_sent"] += 1
        return

    context = tender_context_module.build_tender_context(url, db_path=db_path)
    if tender_context_module.compute_input_hash(context) != input_hash:
        logger.info("Telegram: анализ устарел относительно текущих данных тендера, пропуск: %s", url)
        summary["skipped_stale"] += 1
        return
    eligibility = operational_eligibility.evaluate_context(context, clock())
    if eligibility.expired:
        logger.info("Telegram: срок подачи истёк, пропуск: %s", url)
        summary["skipped_expired"] += 1
        return

    summary["eligible_for_delivery"] += 1
    text = message.build_card(result, url, _format_deadline(eligibility), _official_value(context))
    repo.begin_attempt(url, input_hash, db_path=db_path)
    delivery_id = repo.get_delivery(url, input_hash, db_path=db_path)["id"]
    markup = message.build_reply_markup(delivery_id, url)  # callback_data = full:<delivery id>
    send = client.send_message(text, reply_markup=markup)  # одна попытка на версию за запуск
    if send.ok:
        repo.mark_sent(url, input_hash, send.message_id, db_path=db_path)
        summary["sent"] += 1
        logger.info("Telegram: отправлено (message_id=%s): %s", send.message_id, url)
    else:
        repo.mark_failed(url, input_hash, send.error or "unknown error", db_path=db_path)
        summary["failed"] += 1
        logger.error("Telegram: ошибка отправки %s: %s", url, send.error)


def run_delivery(client, db_path=None, clock=operational_eligibility.utc_now) -> dict:
    """Один проход доставки. Ошибка одного тендера не прерывает остальные."""
    repo.init_db(db_path)
    summary = {
        "status": "ok", "telegram_enabled": True, "eligible_for_delivery": 0, "sent": 0, "failed": 0,
        "skipped_already_sent": 0, "skipped_expired": 0, "skipped_stale": 0, "skipped_incomplete": 0,
    }
    for row in repo.list_deep_completed(db_path=db_path):
        try:
            _deliver_one(client, row["resource_url"], row["input_hash"], db_path, clock, summary)
        except Exception as error:
            logger.exception("Telegram: необработанная ошибка тендера %s", row["resource_url"])
            summary["failed"] += 1
            try:
                repo.mark_failed(row["resource_url"], row["input_hash"], f"{type(error).__name__}: {error}", db_path=db_path)
            except Exception:
                logger.exception("Telegram: не удалось сохранить ошибку доставки: %s", row["resource_url"])
    logger.info("Telegram delivery: %s", summary)
    return summary


def run_delivery_from_env(db_path=None, environ=None, client_factory=None) -> dict | None:
    """
    Production-вход для monitor. None — выключено (клиент не создаётся). enabled без token/chat_id ->
    {"status": "config_error"} без отправок и без записи sent-состояний.
    """
    config = load_config(environ)
    if not config.enabled:
        logger.info("Telegram-уведомления выключены (%s не включён)", ENABLED_ENV)
        return None
    if config.missing:
        error = f"не заданы: {', '.join(config.missing)}"
        logger.error("Telegram-доставка не запущена: %s", error)
        return {"status": "config_error", "telegram_enabled": True, "error_message": error}
    if client_factory is None:
        from src.telegram.client import TelegramClient as client_factory
    return run_delivery(client_factory(config.bot_token, config.chat_id), db_path=db_path)
