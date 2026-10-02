"""
Обработка Telegram callback_query «📋 Полный анализ» (callback_data = full:<telegram_deliveries.id>).

Только чтение SQLite и детерминированное форматирование сохранённого deep result: никакого AI/OpenAI
(модуль не импортирует analysis_pipeline), monitor не запускается, состояние telegram_deliveries не меняется.
Версия анализа определяется строкой доставки: (resource_url, analysis_input_hash); если сохранённый deep
result уже другой версии — "Эта версия анализа больше недоступна." (старая кнопка не подменяется новым анализом).
"""

import logging

from src.ai import operational_eligibility, tender_context as tender_context_module, value_facts
from src.database import analysis_repository, telegram_delivery_repository as repo
from src.telegram import delivery, full_analysis, message

logger = logging.getLogger(__name__)

ANSWER_OPENING = "Открываю полный анализ"
ANSWER_FORBIDDEN = "Недоступно в этом чате."
ANSWER_BAD_BUTTON = "Некорректная кнопка."
ANSWER_NOT_FOUND = "Анализ не найден."
VERSION_UNAVAILABLE = "Эта версия анализа больше недоступна."


def _deadline_and_value(url: str, input_hash: str, db_path, clock) -> tuple[str | None, int | None]:
    """Дедлайн и официальная стоимость — только если текущие данные тендера совпадают с версией анализа."""
    try:
        context = tender_context_module.build_tender_context(url, db_path=db_path)
        if tender_context_module.compute_input_hash(context) != input_hash:
            return None, None
        eligibility = operational_eligibility.evaluate_context(context, clock())
        return delivery._format_deadline(eligibility), value_facts.extract_value_facts(context)["estimated_value_amd"]
    except Exception:
        logger.exception("Telegram callback: не удалось получить дедлайн/стоимость: %s", url)
        return None, None


def _send_unavailable(client, callback_id: str) -> None:
    client.answer_callback_query(callback_id, VERSION_UNAVAILABLE)
    client.send_message(VERSION_UNAVAILABLE)


def handle_callback_query(client, callback_query: dict, allowed_chat_id: str, db_path=None,
                          clock=operational_eligibility.utc_now) -> str:
    """Обрабатывает один callback_query; возвращает короткий код исхода (для логов/тестов)."""
    callback_id = callback_query.get("id")
    if not callback_id:
        return "ignored_no_id"
    chat = ((callback_query.get("message") or {}).get("chat") or {})
    if str(chat.get("id")) != str(allowed_chat_id):
        logger.warning("Telegram callback: чужой чат %r, данные не отправлены", chat.get("id"))
        client.answer_callback_query(callback_id, ANSWER_FORBIDDEN)
        return "forbidden_chat"

    data = callback_query.get("data")
    if not isinstance(data, str) or not data.startswith(message.FULL_CALLBACK_PREFIX):
        logger.info("Telegram callback: неизвестный callback_data, пропуск")
        client.answer_callback_query(callback_id)
        return "unknown_prefix"
    delivery_id = message.parse_full_callback(data)
    if delivery_id is None:
        client.answer_callback_query(callback_id, ANSWER_BAD_BUTTON)
        return "bad_callback_data"

    row = repo.get_delivery_by_id(delivery_id, db_path=db_path)
    if row is None or row["status"] != repo.STATUS_SENT:
        logger.info("Telegram callback: доставка %s не найдена или не sent", delivery_id)
        client.answer_callback_query(callback_id, ANSWER_NOT_FOUND)
        return "delivery_not_found"

    url, input_hash = row["resource_url"], row["analysis_input_hash"]
    deep = analysis_repository.get_deep_analysis(url, db_path=db_path)
    result = deep["result"] if deep else None
    if deep is None or deep["input_hash"] != input_hash or not isinstance(result, dict):
        logger.info("Telegram callback: версия анализа недоступна (delivery %s): %s", delivery_id, url)
        _send_unavailable(client, callback_id)
        return "version_unavailable"

    deadline, value_amd = _deadline_and_value(url, input_hash, db_path, clock)
    pages = full_analysis.build_full_analysis_messages(result, url, deadline, value_amd)
    client.answer_callback_query(callback_id, ANSWER_OPENING)
    for number, text in enumerate(pages, 1):
        send = client.send_message(text)
        if not send.ok:
            logger.error("Telegram callback: ошибка отправки страницы %d/%d (delivery %s): %s",
                         number, len(pages), delivery_id, send.error)
            return "send_failed"
    logger.info("Telegram callback: полный анализ отправлен (delivery %s, страниц %d)", delivery_id, len(pages))
    return "sent"
