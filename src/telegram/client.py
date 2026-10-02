"""
Минимальный Telegram Bot API клиент (POST /<method> через requests): sendMessage, getUpdates,
answerCallbackQuery. Любая ошибка (сеть, не-200, ok=false) возвращается как результат с ok=False,
исключений наружу не бросается. Токен нигде не логируется и вырезается из текста ошибок
(URL с токеном попадает в сообщения requests).
"""

import logging
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)

API_URL = "https://api.telegram.org/bot{token}/{method}"
DEFAULT_TIMEOUT_SECONDS = 15
GET_UPDATES_MARGIN_SECONDS = 10  # HTTP-таймаут getUpdates = long-poll timeout + запас


@dataclass(frozen=True)
class SendResult:
    ok: bool
    message_id: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class UpdatesResult:
    ok: bool
    updates: list = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class AnswerResult:
    ok: bool
    error: str | None = None


class TelegramClient:
    """Тесты подменяют класс любым объектом с такими же методами."""

    def __init__(self, bot_token: str, chat_id: str, timeout: float = DEFAULT_TIMEOUT_SECONDS, post=None):
        self._token = bot_token
        self._chat_id = chat_id
        self._timeout = timeout
        self._post = post or requests.post

    def __repr__(self) -> str:
        return "TelegramClient(chat_id=%r, token=***)" % (self._chat_id,)

    def _redact(self, text: str) -> str:
        return text.replace(self._token, "***") if self._token else text

    def _call(self, method: str, payload: dict, timeout: float) -> tuple[dict | None, str | None]:
        """(result_json, None) при успехе (ok=true, HTTP 200), иначе (None, redacted error)."""
        try:
            response = self._post(API_URL.format(token=self._token, method=method), json=payload, timeout=timeout)
        except requests.RequestException as error:
            return None, self._redact(f"network_error: {type(error).__name__}: {error}")
        except Exception as error:  # fail-safe: клиент не должен ронять вызывающий код
            return None, self._redact(f"unexpected_error: {type(error).__name__}")

        try:
            data = response.json()
        except ValueError:
            data = None
        description = data.get("description") if isinstance(data, dict) else None

        if response.status_code != 200:
            return None, self._redact(f"http_{response.status_code}: {description or ''}".strip())
        if not isinstance(data, dict) or data.get("ok") is not True:
            return None, self._redact(f"telegram_not_ok: {description or 'invalid response'}")
        return data, None

    def send_message(self, text: str, reply_markup: dict | None = None) -> SendResult:
        payload = {
            "chat_id": self._chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True,
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup
        data, error = self._call("sendMessage", payload, self._timeout)
        if data is None:
            return SendResult(False, error=error)
        message_id = (data.get("result") or {}).get("message_id") if isinstance(data.get("result"), dict) else None
        return SendResult(True, message_id=message_id if isinstance(message_id, int) else None)

    def get_updates(self, offset: int | None = None, timeout: int = 30) -> UpdatesResult:
        """Long polling: только callback_query. timeout — секунды ожидания на стороне Telegram."""
        payload = {"timeout": timeout, "allowed_updates": ["callback_query"]}
        if offset is not None:
            payload["offset"] = offset
        data, error = self._call("getUpdates", payload, timeout + GET_UPDATES_MARGIN_SECONDS)
        if data is None:
            return UpdatesResult(False, error=error)
        result = data.get("result")
        return UpdatesResult(True, updates=[u for u in result if isinstance(u, dict)] if isinstance(result, list) else [])

    def answer_callback_query(self, callback_query_id: str, text: str | None = None) -> AnswerResult:
        payload = {"callback_query_id": callback_query_id}
        if text:
            payload["text"] = text
        data, error = self._call("answerCallbackQuery", payload, self._timeout)
        return AnswerResult(data is not None, error=error)
