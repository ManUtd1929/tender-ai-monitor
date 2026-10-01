"""
Минимальный Telegram Bot API клиент (POST /sendMessage через requests). Любая ошибка (сеть, не-200,
ok=false) возвращается как SendResult(ok=False), исключений наружу не бросается. Токен нигде не
логируется и вырезается из текста ошибок (URL с токеном попадает в сообщения requests).
"""

import logging
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

API_URL = "https://api.telegram.org/bot{token}/sendMessage"
DEFAULT_TIMEOUT_SECONDS = 15


@dataclass(frozen=True)
class SendResult:
    ok: bool
    message_id: int | None = None
    error: str | None = None


class TelegramClient:
    """send_message(text) -> SendResult. Тесты подменяют класс любым объектом с таким же методом."""

    def __init__(self, bot_token: str, chat_id: str, timeout: float = DEFAULT_TIMEOUT_SECONDS, post=None):
        self._token = bot_token
        self._chat_id = chat_id
        self._timeout = timeout
        self._post = post or requests.post

    def __repr__(self) -> str:
        return "TelegramClient(chat_id=%r, token=***)" % (self._chat_id,)

    def _redact(self, text: str) -> str:
        return text.replace(self._token, "***") if self._token else text

    def send_message(self, text: str) -> SendResult:
        payload = {
            "chat_id": self._chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True,
        }
        try:
            response = self._post(API_URL.format(token=self._token), json=payload, timeout=self._timeout)
        except requests.RequestException as error:
            return SendResult(False, error=self._redact(f"network_error: {type(error).__name__}: {error}"))
        except Exception as error:  # fail-safe: клиент не должен ронять delivery stage
            return SendResult(False, error=self._redact(f"unexpected_error: {type(error).__name__}"))

        try:
            data = response.json()
        except ValueError:
            data = None
        description = data.get("description") if isinstance(data, dict) else None

        if response.status_code != 200:
            return SendResult(False, error=self._redact(f"http_{response.status_code}: {description or ''}".strip()))
        if not isinstance(data, dict) or data.get("ok") is not True:
            return SendResult(False, error=self._redact(f"telegram_not_ok: {description or 'invalid response'}"))
        message_id = (data.get("result") or {}).get("message_id")
        return SendResult(True, message_id=message_id if isinstance(message_id, int) else None)
