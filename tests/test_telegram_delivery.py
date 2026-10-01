"""
Тесты Telegram delivery layer: fake client, временная SQLite БД, ZERO сети, ZERO OpenAI.

    python -m unittest tests.test_telegram_delivery -v
"""

import hashlib
import io
import logging
import sqlite3
import unittest
from contextlib import closing
from datetime import timedelta
from pathlib import Path
from unittest import mock

import requests

from src import monitor
from src.database import analysis_repository, enrichment_repository, pipeline_state_repository as state_repo
from src.database import telegram_delivery_repository as repo
from src.telegram import delivery, message
from src.telegram.client import SendResult, TelegramClient
from tests.test_analysis_pipeline import triage_result
from tests.test_monitor import make_combined, make_document_result, make_enrichment_result, make_save_result
from tests.test_operational_eligibility import FUTURE, NOW, DeadlineCase

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
TOKEN = "123456:SECRET-TOKEN-VALUE"


class FakeTelegramClient:
    def __init__(self, results=None):
        self.texts = []
        self.results = list(results or [])

    def send_message(self, text):
        self.texts.append(text)
        return self.results.pop(0) if self.results else SendResult(True, message_id=1000 + len(self.texts))


FAIL = SendResult(False, error="http_500: boom")


class DeliveryCase(DeadlineCase):
    def setUp(self):
        super().setUp()
        repo.init_db(self.db_path)
        self._env_before = hashlib.sha256(ENV_FILE.read_bytes()).hexdigest() if ENV_FILE.exists() else None
        self.addCleanup(lambda: self.assertEqual(
            self._env_before, hashlib.sha256(ENV_FILE.read_bytes()).hexdigest() if ENV_FILE.exists() else None))
        for analyzer in (self.triage, self.deep):
            analyzer._client = object()

    def analyzed(self, number=1, deadline=FUTURE) -> str:
        url = self.add_deadline(number, deadline)
        self.make().process_announcement(url)
        return url

    def deliver(self, client, clock=None):
        return delivery.run_delivery(client, db_path=self.db_path, clock=clock or self.clock)

    def delivery_row(self, url):
        return repo.get_delivery(url, analysis_repository.get_deep_analysis(url, self.db_path)["input_hash"], self.db_path)

    def ai_calls(self):
        return (len(self.triage.calls), len(self.deep.calls))


class SelectionTests(DeliveryCase):
    def test_disabled_makes_zero_client_calls(self):
        self.analyzed()
        factory = mock.Mock()
        for env in ({}, {"TELEGRAM_NOTIFICATIONS_ENABLED": "false", "TELEGRAM_BOT_TOKEN": TOKEN, "TELEGRAM_CHAT_ID": "1"}):
            self.assertIsNone(delivery.run_delivery_from_env(self.db_path, env, factory))
        factory.assert_not_called()

    def test_deep_completed_unsent_is_sent_with_durable_state(self):
        url = self.analyzed()
        client = FakeTelegramClient()
        summary = self.deliver(client)
        self.assertEqual((summary["eligible_for_delivery"], summary["sent"], summary["failed"]), (1, 1, 0))
        self.assertEqual(len(client.texts), 1)
        row = self.delivery_row(url)
        self.assertEqual((row["status"], row["telegram_message_id"], row["attempt_count"]), ("sent", 1001, 1))
        self.assertIsNotNone(row["sent_at"])

    def test_already_sent_same_version_not_duplicated(self):
        self.analyzed()
        client = FakeTelegramClient()
        self.deliver(client)
        summary = self.deliver(client)
        self.assertEqual(len(client.texts), 1)
        self.assertEqual((summary["sent"], summary["skipped_already_sent"], summary["eligible_for_delivery"]), (0, 1, 0))

    def test_changed_analysis_version_is_sent_again(self):
        url = self.analyzed()
        client = FakeTelegramClient()
        self.deliver(client)
        enrichment_repository.save_enrichment(
            url, {"enrichment_status": "success", "description": "Изменённое описание", "documents": [],
                  "deadline_at_detail": None}, self.db_path)
        self.make().process_announcement(url)  # fake AI: новый input_hash -> новый deep result
        summary = self.deliver(client)
        self.assertEqual((summary["sent"], len(client.texts)), (1, 2))
        with closing(sqlite3.connect(self.db_path)) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM telegram_deliveries").fetchone()[0], 2)

    def test_stale_deep_result_not_sent(self):
        url = self.analyzed()
        enrichment_repository.save_enrichment(
            url, {"enrichment_status": "success", "description": "Новое", "documents": [], "deadline_at_detail": None},
            self.db_path)  # данные изменились, AI ещё не пересчитал
        client = FakeTelegramClient()
        summary = self.deliver(client)
        self.assertEqual((client.texts, summary["skipped_stale"]), ([], 1))

    def test_not_relevant_not_sent(self):
        url = self.add_deadline(1, FUTURE)
        self.triage.outcomes[url] = triage_result("not_relevant")
        self.make().process_announcement(url)
        client = FakeTelegramClient()
        self.deliver(client)
        self.assertEqual(client.texts, [])

    def test_other_states_not_sent(self):
        url = self.analyzed()
        digest = state_repo.get_state(url, self.db_path)["input_hash"]
        for state in (state_repo.STATE_DEEP_DEFERRED_INPUT, state_repo.STATE_ESCALATION_CANDIDATE,
                      state_repo.STATE_DEEP_ERROR, state_repo.STATE_DEEP_DEFERRED_BUDGET,
                      state_repo.STATE_SKIPPED_EXPIRED, state_repo.STATE_TRIAGE_COMPLETED):
            with self.subTest(state=state):
                state_repo.save_state(url, state, input_hash=digest, db_path=self.db_path)
                client = FakeTelegramClient()
                self.deliver(client)
                self.assertEqual(client.texts, [])

    def test_expired_current_tender_not_sent(self):
        self.analyzed(deadline=FUTURE)
        client = FakeTelegramClient()
        summary = self.deliver(client, clock=lambda: NOW + timedelta(days=30))
        self.assertEqual((client.texts, summary["skipped_expired"]), ([], 1))

    def test_unknown_deadline_is_still_sent(self):
        self.analyzed(deadline=None)
        client = FakeTelegramClient()
        self.deliver(client)
        self.assertEqual(len(client.texts), 1)
        self.assertNotIn("Дедлайн", client.texts[0])


class ConfigAndFailureTests(DeliveryCase):
    def test_enabled_without_token_or_chat_is_config_error(self):
        url = self.analyzed()
        factory = mock.Mock()
        for env in ({"TELEGRAM_NOTIFICATIONS_ENABLED": "true"},
                    {"TELEGRAM_NOTIFICATIONS_ENABLED": "true", "TELEGRAM_BOT_TOKEN": TOKEN},
                    {"TELEGRAM_NOTIFICATIONS_ENABLED": "true", "TELEGRAM_CHAT_ID": "1"}):
            result = delivery.run_delivery_from_env(self.db_path, env, factory)
            self.assertEqual(result["status"], "config_error")
        factory.assert_not_called()
        self.assertIsNone(repo.get_delivery(url, state_repo.get_state(url, self.db_path)["input_hash"], self.db_path))

    def test_config_repr_hides_token(self):
        config = delivery.load_config({"TELEGRAM_BOT_TOKEN": TOKEN, "TELEGRAM_CHAT_ID": "1"})
        self.assertNotIn(TOKEN, repr(config))
        self.assertNotIn(TOKEN, repr(TelegramClient(TOKEN, "1")))

    def test_failed_send_is_stored_and_retried_next_run_once_per_run(self):
        url = self.analyzed()
        client = FakeTelegramClient([FAIL])
        summary = self.deliver(client)
        self.assertEqual((summary["failed"], summary["sent"], len(client.texts)), (1, 0, 1))  # одна попытка за запуск
        row = self.delivery_row(url)
        self.assertEqual((row["status"], row["attempt_count"], row["last_error"], row["telegram_message_id"]),
                         ("failed", 1, "http_500: boom", None))
        self.assertIsNone(row["sent_at"])
        summary = self.deliver(client)  # следующий запуск: retry
        self.assertEqual((summary["sent"], len(client.texts)), (1, 2))
        row = self.delivery_row(url)
        self.assertEqual((row["status"], row["attempt_count"], row["last_error"]), ("sent", 2, None))

    def test_retry_never_calls_ai(self):
        url = self.analyzed()
        before = self.ai_calls()
        self.deliver(FakeTelegramClient([FAIL]))
        self.deliver(FakeTelegramClient())
        self.assertEqual(self.ai_calls(), before)
        self.openai_client.assert_not_called()
        self.assertEqual(self.delivery_row(url)["status"], "sent")

    def test_failure_keeps_ai_results_and_other_tenders_continue(self):
        first, second = self.analyzed(1), self.analyzed(2)
        deep_before = analysis_repository.get_deep_analysis(first, self.db_path)
        client = FakeTelegramClient([FAIL])
        summary = self.deliver(client)
        self.assertEqual((summary["failed"], summary["sent"]), (1, 1))
        self.assertEqual(analysis_repository.get_deep_analysis(first, self.db_path), deep_before)
        self.assertEqual(state_repo.get_state(first, self.db_path)["state"], state_repo.STATE_DEEP_COMPLETED)
        self.assertEqual(state_repo.get_state(second, self.db_path)["state"], state_repo.STATE_DEEP_COMPLETED)

    def test_client_exception_is_contained(self):
        url = self.analyzed()
        client = mock.Mock()
        client.send_message.side_effect = RuntimeError("boom")
        with self.assertLogs(delivery.logger, "ERROR"):
            summary = self.deliver(client)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(self.delivery_row(url)["status"], "failed")

    def test_token_never_in_logs_or_stored_error(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        root = logging.getLogger()
        root.addHandler(handler)
        old_level = root.level
        root.setLevel(logging.DEBUG)
        self.addCleanup(lambda: (root.removeHandler(handler), root.setLevel(old_level)))
        url = self.analyzed()
        post = mock.Mock(side_effect=requests.ConnectionError(f"/bot{TOKEN}/sendMessage"))
        summary = self.deliver(TelegramClient(TOKEN, "42", post=post))
        self.assertEqual(summary["failed"], 1)
        self.assertNotIn(TOKEN, stream.getvalue())
        self.assertNotIn(TOKEN, self.delivery_row(url)["last_error"])

    def test_monitor_stage_survives_delivery_crash(self):
        with mock.patch.object(delivery, "run_delivery_from_env", side_effect=RuntimeError(TOKEN)):
            result = monitor._run_telegram_if_enabled()
        self.assertEqual(result, {"status": "error", "error_type": "RuntimeError"})


class MonitorIntegrationTests(unittest.TestCase):
    def setUp(self):
        names = ("configure_tls", "init_db", "fetch_all_sections", "save_announcements", "count_announcements",
                 "process_announcements", "process_enriched_announcements")
        for name in names:
            patcher = mock.patch.object(monitor, name)
            setattr(self, name, patcher.start())
            self.addCleanup(patcher.stop)
        for target, name in ((monitor.enrichment_repository, "init_db"), (monitor.document_repository, "init_db"),
                             (monitor.enrichment_repository, "count_enrichment_processing_candidates"),
                             (monitor.enrichment_repository, "get_enrichment_processing_candidates"),
                             (monitor.document_repository, "count_document_processing_candidates"),
                             (monitor.document_repository, "get_document_processing_candidates")):
            patcher = mock.patch.object(target, name, return_value=[] if "get_" in name else 0)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.fetch_all_sections.return_value = []
        self.save_announcements.return_value = make_save_result()
        self.count_announcements.return_value = 0
        self.process_announcements.return_value = make_enrichment_result()
        self.process_enriched_announcements.return_value = make_document_result()

    def run_with_env(self, env):
        with mock.patch.dict("os.environ", env, clear=True), \
                mock.patch.object(delivery, "run_delivery_from_env", wraps=delivery.run_delivery_from_env) as run:
            return monitor.run_monitor(), run

    def test_disabled_telegram_is_none_and_runs_after_ai(self):
        result, _ = self.run_with_env({})
        self.assertIsNone(result["telegram_delivery"])

    def test_enabled_without_config_does_not_abort_monitor(self):
        result, _ = self.run_with_env({"TELEGRAM_NOTIFICATIONS_ENABLED": "true"})
        self.assertEqual(result["telegram_delivery"]["status"], "config_error")
        self.assertEqual(result["fetched_count"], 0)

    def test_runs_after_ai_stage(self):
        order = []
        with mock.patch.object(monitor, "_run_ai_analysis_if_enabled", side_effect=lambda: order.append("ai")), \
                mock.patch.object(monitor, "_run_telegram_if_enabled", side_effect=lambda: order.append("telegram")):
            monitor.run_monitor()
        self.assertEqual(order, ["ai", "telegram"])


FULL_RESULT = {
    "summary": "Закупка ноутбуков", "opportunity_type": "procurement", "category": "IT оборудование",
    "contracting_authority": 'ООО "Ромашка" <Филиал> & Ко', "procedure_code": "ԷԱՃ-123",
    "confidence": "medium", "manual_review_required": False, "evidence": [{"text": "EVIDENCE-QUOTE"}],
    "participation_barriers": [
        {"type": "license", "description": "Нужна лицензия", "severity": "low", "evidence": [{"text": "EVIDENCE-QUOTE"}]},
        {"type": "bid_security", "description": "Обеспечение заявки 1%", "severity": "high", "evidence": []},
    ],
    "missing_information": ["Срок поставки не указан"],
    "procurement": {
        "subject": "Ноутбуки <15\"> & аксессуары", "quantity_summary": "30 шт.", "total_lots": 2,
        "items": [{"item_name": "Ноутбук", "quantity": "30", "unit": "шт.", "evidence": [{"text": "EVIDENCE-QUOTE"}]}],
        "lots": [], "technical_requirements": ["RAM 16 GB"], "country_of_origin_requirements": [], "certifications": [],
    },
    "logistics": None,
}


class MessageTests(unittest.TestCase):
    URL = "https://example.test/resource/1?a=1&b=2"

    def test_card_content_and_no_evidence(self):
        text = message.build_card(FULL_RESULT, self.URL, "10.10.2026 11:00", 980000)
        for expected in ("🟢 Новый подходящий тендер", "Ноутбук — 30 шт.", "Лотов: 2", "980 000 AMD",
                         "<b>Уверенность анализа:</b>\nсредняя", "10.10.2026 11:00", "RAM 16 GB", "Срок поставки не указан"):
            self.assertIn(expected, text)
        self.assertNotIn("EVIDENCE-QUOTE", text)
        self.assertLess(text.index("Обеспечение заявки"), text.index("Нужна лицензия"))  # severity high первым
        self.assertLess(len(text), len(str(FULL_RESULT)) + 600)

    def test_html_escaping(self):
        text = message.build_card(FULL_RESULT, self.URL, None, None)
        self.assertIn("Ноутбуки &lt;15\"&gt; &amp; аксессуары", text)
        self.assertIn("ООО \"Ромашка\" &lt;Филиал&gt; &amp; Ко", text)
        self.assertIn('<a href="https://example.test/resource/1?a=1&amp;b=2">Открыть тендер</a>', text)
        self.assertNotIn("<Филиал>", text)

    def test_missing_optional_fields(self):
        text = message.build_card({"summary": "Краткое описание", "confidence": "low"}, None)
        self.assertNotIn("None", text)
        self.assertNotIn("<a ", text)  # нет URL — нет ссылки
        self.assertIn("<b>Заказчик:</b>\nНе указано", text)
        for title in ("Процедура", "Категория", "Дедлайн", "Что закупают", "Количество", "Оценочная стоимость",
                      "Ключевые требования", "Барьеры", "Нужно уточнить"):
            self.assertNotIn(title, text)
        for section in text.split("\n\n"):
            self.assertTrue(section.strip())

    def test_none_like_values_are_dropped(self):
        result = {**FULL_RESULT, "category": "None", "procedure_code": " ", "missing_information": ["null", "", None]}
        text = message.build_card(result, self.URL)
        for title in ("Категория", "Процедура", "Нужно уточнить"):
            self.assertNotIn(title, text)

    def test_value_only_when_provided(self):
        result = {**FULL_RESULT, "procurement": {**FULL_RESULT["procurement"], "estimated_value_amd": "999"}}
        self.assertNotIn("Оценочная стоимость", message.build_card(result, self.URL, None, None))

    def test_truncation_with_remainder(self):
        many = {**FULL_RESULT, "missing_information": [f"Вопрос {n}" for n in range(12)],
                "participation_barriers": [{"description": f"Барьер {n}", "severity": "low"} for n in range(9)],
                "procurement": {**FULL_RESULT["procurement"],
                                "items": [{"item_name": f"Товар {n}"} for n in range(30)]}}
        text = message.build_card(many, self.URL)
        self.assertIn("... и ещё 25", text)  # items: 30 - 5
        self.assertIn("... и ещё 8", text)  # missing: 12 - 4
        self.assertIn("... и ещё 6", text)  # barriers: 9 - 3
        self.assertNotIn("Товар 5", text)
        self.assertLessEqual(len(text), message.TELEGRAM_MESSAGE_LIMIT)

    def test_worst_case_fits_telegram_limit(self):
        long = "<&>" * 200
        many = {"summary": long, "confidence": "high", "contracting_authority": long,
                "missing_information": [long] * 20, "participation_barriers": [{"description": long}] * 20,
                "procurement": {"subject": long, "items": [{"item_name": long}] * 50,
                                "technical_requirements": [long] * 20}}
        self.assertLessEqual(len(message.build_card(many, self.URL)), message.TELEGRAM_MESSAGE_LIMIT)


def _bullet_count(text: str, title: str) -> int:
    section = next(p for p in text.split("\n\n") if p.startswith(f"<b>{title}"))
    return sum(line.startswith("• ") for line in section.split("\n"))


class CompactCardTests(unittest.TestCase):
    URL = "https://example.test/r/1"

    def card(self, **changes):
        procurement = {**FULL_RESULT["procurement"], **changes.pop("procurement", {})}
        return message.build_card({**FULL_RESULT, **changes, "procurement": procurement}, self.URL)

    def test_category_human_label(self):
        text = self.card(category="computer_equipment")
        self.assertIn("<b>Категория:</b>\nКомпьютерное оборудование", text)
        self.assertNotIn("computer_equipment", text)
        for code in message.CATEGORY_RU:
            self.assertNotIn("_", message.category_label(code))

    def test_unknown_category_degrades_safely(self):
        self.assertEqual(message.category_label("office_chairs"), "Office chairs")
        self.assertEqual(message.category_label("IT оборудование"), "IT оборудование")
        self.assertIsNone(message.category_label(None))

    def test_placeholder_lots_do_not_create_what_is_procured(self):
        for items in ([{"item_name": "Лот 1"}, {"item_name": "Лот 2"}], [{"item_name": "Lot 1"}]):
            text = self.card(procurement={"items": items, "lots": [{"lot_number": "1"}, {"lot_number": "2"}]})
            self.assertNotIn("Что закупают", text)
            self.assertIn("Лотов: 2", text)

    def test_lots_not_duplicated_when_summary_mentions_them(self):
        text = self.card(procurement={"quantity_summary": "2 лота. Позиции не указаны.", "total_lots": 2})
        self.assertNotIn("Лотов: 2", text)

    def test_real_item_names_still_shown(self):
        text = self.card(procurement={"items": [{"item_name": "Лот 1"}, {"item_name": "Ноутбук"}]})
        self.assertIn("<b>Что закупают:</b>\n• Ноутбук", text)
        self.assertNotIn("Лот 1", text)

    def test_list_limits(self):
        text = self.card(
            missing_information=[f"Вопрос {n}" for n in range(9)],
            participation_barriers=[{"description": f"Барьер {n}"} for n in range(9)],
            procurement={"technical_requirements": [f"Требование {n}" for n in range(9)],
                         "items": [{"item_name": f"Товар {n}"} for n in range(9)]})
        self.assertEqual(_bullet_count(text, "Ключевые требования"), 3)
        self.assertEqual(_bullet_count(text, "⚠️ Барьеры"), 3)
        self.assertEqual(_bullet_count(text, "❓ Нужно уточнить"), 4)
        self.assertEqual(_bullet_count(text, "Что закупают"), 5)
        for expected in ("... и ещё 6", "... и ещё 5", "... и ещё 4"):
            self.assertIn(expected, text)

    def test_why_interesting_limited_and_optional(self):
        why = "Первое. Второе! Третье? Четвёртое."
        text = self.card(why_interesting=why)
        self.assertEqual(_bullet_count(text, "Почему может быть интересно"), 2)
        self.assertIn("• Второе!", text)
        self.assertNotIn("Третье", text)
        for empty in ("", None, "  "):
            self.assertNotIn("Почему может быть интересно", self.card(why_interesting=empty))
        self.assertNotIn("Почему может быть интересно", message.build_card({"summary": "x"}, None))

    def test_truncation_never_cuts_words(self):
        text = "Поставщик должен подтвердить наличие сертификата соответствия на всю партию товара"
        for limit in range(15, len(text)):
            clipped = message._clip(text, limit)
            self.assertLessEqual(len(clipped), limit)
            body = clipped.removesuffix("...")
            self.assertTrue(body == text or text.startswith(body), clipped)
            self.assertTrue(len(body) == len(text) or text[len(body)] == " ", clipped)
        self.assertEqual(message._clip("Коротко", 50), "Коротко")  # не сокращён — без "..."
        self.assertEqual(message._clip("Первое предложение. Второе предложение длинное", 35), "Первое предложение.")

    def test_total_length_with_all_sections(self):
        long = "Очень длинное требование к поставке " * 20
        text = self.card(why_interesting=long, missing_information=[long] * 9,
                         participation_barriers=[{"description": long}] * 9,
                         procurement={"technical_requirements": [long] * 9, "items": [{"item_name": long}] * 9})
        self.assertLessEqual(len(text), message.TELEGRAM_MESSAGE_LIMIT)


class RealClientTests(unittest.TestCase):
    """Настоящий TelegramClient, но requests.post подменён: сети нет."""

    def response(self, status, payload):
        resp = mock.Mock(status_code=status)
        resp.json.return_value = payload
        return resp

    def client(self, post):
        return TelegramClient(TOKEN, "42", timeout=5, post=post)

    def test_success_payload_and_timeout(self):
        post = mock.Mock(return_value=self.response(200, {"ok": True, "result": {"message_id": 77}}))
        result = self.client(post).send_message("hi")
        self.assertEqual(result, SendResult(True, 77))
        url = post.call_args.args[0]
        self.assertEqual(url, f"https://api.telegram.org/bot{TOKEN}/sendMessage")
        self.assertEqual(post.call_args.kwargs["timeout"], 5)
        self.assertEqual(post.call_args.kwargs["json"]["parse_mode"], "HTML")

    def test_failures_are_controlled_and_never_leak_token(self):
        cases = [
            mock.Mock(side_effect=requests.ConnectionError(f"HTTPSConnectionPool: /bot{TOKEN}/sendMessage")),
            mock.Mock(side_effect=requests.Timeout(f"timeout /bot{TOKEN}/sendMessage")),
            mock.Mock(return_value=self.response(401, {"ok": False, "description": "Unauthorized"})),
            mock.Mock(return_value=self.response(200, {"ok": False, "description": f"bad {TOKEN}"})),
        ]
        for post in cases:
            result = self.client(post).send_message("hi")
            self.assertFalse(result.ok)
            self.assertNotIn(TOKEN, result.error)


class EnvExampleTests(unittest.TestCase):
    def test_env_example_documents_disabled_telegram(self):
        text = "\n" + (Path(__file__).resolve().parents[1] / ".env.example").read_text(encoding="utf-8")
        for line in ("TELEGRAM_NOTIFICATIONS_ENABLED=false", "TELEGRAM_BOT_TOKEN=", "TELEGRAM_CHAT_ID="):
            self.assertIn("\n" + line + "\n", text.replace("\r\n", "\n"))


if __name__ == "__main__":
    unittest.main()
