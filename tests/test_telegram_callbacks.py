"""
Тесты интерактивного слоя Telegram («📋 Полный анализ»): fake client, временная SQLite БД,
ZERO сети, ZERO OpenAI.

    python -m unittest tests.test_telegram_callbacks -v
"""

import ast
import io
import logging
import re
import unittest
from pathlib import Path
from unittest import mock

import requests

from src.database import analysis_repository, enrichment_repository
from src.database import telegram_delivery_repository as repo
from src.telegram import callback_worker, callbacks, full_analysis, message
from src.telegram.client import AnswerResult, SendResult, TelegramClient, UpdatesResult
from tests.test_telegram_delivery import FULL_RESULT, TOKEN, DeliveryCase, FakeTelegramClient

CHAT = "42"
LIMIT = message.TELEGRAM_MESSAGE_LIMIT
SRC = Path(__file__).resolve().parents[1] / "src" / "telegram"


def big_result(items=30, **changes) -> dict:
    result = {
        **FULL_RESULT,
        "why_interesting": "Первая причина. Вторая причина. Третья причина.",
        "participation_barriers": [
            {"type": "bid_security", "description": "Обеспечение квалификации — 15% от цены закупки",
             "severity": "high", "evidence": [{"text": "EVIDENCE-QUOTE"}]},
            {"type": "contract_security", "description": "10% от цены закупки", "severity": "medium", "evidence": []},
            {"type": "license", "description": "Нужна лицензия", "severity": "low", "evidence": []},
        ],
        "missing_information": [f"Вопрос {n}" for n in range(12)],
        "source_conflicts": [],
        "procurement": {
            **FULL_RESULT["procurement"],
            "items": [{"item_name": f"Товар номер {n}", "quantity": "5", "unit": "шт.", "lot_number": str(n),
                       "key_specifications": [f"характеристика {n}"], "evidence": [{"text": "EVIDENCE-QUOTE"}]}
                      for n in range(1, items + 1)],
            "lots": [{"lot_number": "1", "description": "Первый лот", "item_count": 3, "evidence": []}],
            "technical_requirements": [f"Требование {n}" for n in range(1, 16)],
        },
    }
    result.update(changes)
    return result


class FakeClient(FakeTelegramClient):
    def __init__(self, results=None):
        super().__init__(results)
        self.answers = []

    def answer_callback_query(self, callback_query_id, text=None):
        self.answers.append((callback_query_id, text))
        return AnswerResult(True)


def callback_update(data, chat=CHAT, update_id=1):
    return {"update_id": update_id, "callback_query": {"id": f"cb{update_id}", "data": data,
                                                       "message": {"chat": {"id": int(chat)}}}}


class ReplyMarkupTests(DeliveryCase):
    def test_short_card_gets_callback_and_url_buttons(self):
        self.deep.default_outcome = big_result()
        url = self.analyzed()
        client = FakeClient()
        self.deliver(client)
        row = self.delivery_row(url)
        buttons = client.markups[0]["inline_keyboard"][0]
        self.assertEqual(buttons[0], {"text": "📋 Полный анализ", "callback_data": f"full:{row['id']}"})
        self.assertEqual(buttons[1], {"text": "🔗 Открыть тендер", "url": url})

    def test_no_url_button_without_url_and_callback_data_short(self):
        for url in (None, "", "ftp://x"):
            self.assertEqual(len(message.build_reply_markup(1, url)["inline_keyboard"][0]), 1)
        data = message.build_reply_markup(10**15, "https://e.test/" + "a" * 500)["inline_keyboard"][0][0]["callback_data"]
        self.assertLess(len(data.encode()), 64)
        self.assertNotIn("http", data)

    def test_parse_callback(self):
        self.assertEqual(message.parse_full_callback("full:42"), 42)
        for bad in ("full:", "full:x", "full:1:2", "other:1", None, "full:-1", 5):
            self.assertIsNone(message.parse_full_callback(bad))


class CallbackFlowTests(DeliveryCase):
    def setUp(self):
        super().setUp()
        self.deep.default_outcome = big_result()
        self.url = self.analyzed()
        self.client = FakeClient()
        self.deliver(self.client)
        self.row = self.delivery_row(self.url)
        self.calls_before = self.ai_calls()

    def press(self, data=None, chat=CHAT, client=None):
        client = client or FakeClient()
        outcome = callbacks.handle_callback_query(
            client, callback_update(data or f"full:{self.row['id']}", chat)["callback_query"], CHAT,
            db_path=self.db_path, clock=self.clock)
        return outcome, client

    def test_correct_id_sends_exact_version_without_ai(self):
        outcome, client = self.press()
        self.assertEqual(outcome, "sent")
        self.assertEqual(client.answers, [("cb1", callbacks.ANSWER_OPENING)])
        text = "\n".join(client.texts)
        self.assertIn("Товар номер 30", text)
        self.assertIn(self.url, text)
        self.assertIn("Дедлайн", text)
        self.assertEqual(self.ai_calls(), self.calls_before)
        self.openai_client.assert_not_called()

    def test_repeat_press_keeps_delivery_state_and_sends_again(self):
        before = (self.row, self.count("telegram_deliveries"))
        first = self.press()[1].texts
        second = self.press()[1].texts
        self.assertEqual(first, second)
        self.assertEqual((self.delivery_row(self.url), self.count("telegram_deliveries")), before)
        self.assertEqual(self.ai_calls(), self.calls_before)
        self.assertEqual(len(self.client.texts), 1)  # короткая карточка не дублируется

    def test_old_button_not_replaced_by_new_analysis(self):
        enrichment_repository.save_enrichment(
            self.url, {"enrichment_status": "success", "description": "Изменённое", "documents": [],
                       "deadline_at_detail": None}, self.db_path)
        self.deep.default_outcome = big_result(items=2, summary="НОВЫЙ АНАЛИЗ")
        self.make().process_announcement(self.url)
        new_hash = analysis_repository.get_deep_analysis(self.url, self.db_path)["input_hash"]
        self.assertNotEqual(new_hash, self.row["analysis_input_hash"])
        outcome, client = self.press()
        self.assertEqual(outcome, "version_unavailable")
        self.assertEqual(client.texts, [callbacks.VERSION_UNAVAILABLE])
        self.assertNotIn("НОВЫЙ АНАЛИЗ", "".join(client.texts))
        self.assertEqual(client.answers[0][1], callbacks.VERSION_UNAVAILABLE)

    def test_missing_version_and_missing_delivery_do_not_crash(self):
        with mock.patch.object(analysis_repository, "get_deep_analysis", return_value=None):
            self.assertEqual(self.press()[0], "version_unavailable")
        self.assertEqual(self.press("full:99999")[0], "delivery_not_found")

    def test_not_sent_delivery_is_not_shown(self):
        with mock.patch.object(repo, "get_delivery_by_id", return_value={**self.row, "status": "failed"}):
            outcome, client = self.press()
        self.assertEqual((outcome, client.texts), ("delivery_not_found", []))

    def test_unauthorized_chat_gets_no_data(self):
        outcome, client = self.press(chat="777")
        self.assertEqual(outcome, "forbidden_chat")
        self.assertEqual(client.texts, [])
        self.assertEqual(client.answers, [("cb1", callbacks.ANSWER_FORBIDDEN)])
        self.assertNotIn(self.url, str(client.answers))

    def test_message_without_chat_is_unauthorized(self):
        client = FakeClient()
        outcome = callbacks.handle_callback_query(
            client, {"id": "x", "data": f"full:{self.row['id']}"}, CHAT, self.db_path)
        self.assertEqual((outcome, client.texts), ("forbidden_chat", []))

    def test_unknown_prefix_is_acknowledged_safely(self):
        for data in ("other:1", "x", "FULL:1"):
            outcome, client = self.press(data)
            self.assertEqual((outcome, client.texts), ("unknown_prefix", []))
            self.assertEqual(client.answers, [("cb1", None)])
        self.assertEqual(self.press("full:abc")[0], "bad_callback_data")

    def test_send_failure_stops_and_is_logged(self):
        client = FakeClient([SendResult(False, error="http_500")])
        with self.assertLogs(callbacks.logger, "ERROR"):
            outcome, _ = self.press(client=client)
        self.assertEqual(outcome, "send_failed")
        self.assertEqual(len(client.texts), 1)

    def test_modules_do_not_import_ai_pipeline_or_monitor(self):
        for name in ("callbacks.py", "callback_worker.py", "full_analysis.py"):
            tree = ast.parse((SRC / name).read_text(encoding="utf-8"))
            imported = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    imported += [node.module or "", *(a.name for a in node.names)]
                elif isinstance(node, ast.Import):
                    imported += [a.name for a in node.names]
            for forbidden in ("openai", "analysis_pipeline", "monitor"):
                self.assertFalse(any(forbidden in name for name in imported), (name, forbidden))


class FullAnalysisFormatTests(unittest.TestCase):
    URL = "https://example.test/resource/1?a=1&b=2"

    def build(self, **changes):
        return full_analysis.build_full_analysis_messages(big_result(**changes), self.URL, "10.10.2026 11:00", 980000)

    def test_all_30_items_present_and_split_within_limit(self):
        pages = self.build()
        text = "\n".join(pages)
        for n in range(1, 31):
            self.assertIn(f"Товар номер {n} — 5 шт. (лот {n})", text)
            self.assertIn(f"характеристика {n}", text)
        for page in pages:
            self.assertLessEqual(len(page), LIMIT)
        self.assertNotIn("Показано", text)
        self.assertNotIn("и ещё", text)
        many = self.build(items=150)
        self.assertGreater(len(many), 2)
        self.assertTrue(many[0].startswith(f"📋 Полный анализ — 1/{len(many)}"))
        self.assertTrue(many[-1].startswith(f"📋 Полный анализ — {len(many)}/{len(many)}"))
        self.assertTrue(all(len(p) <= LIMIT for p in many))

    def test_split_never_cuts_bullets_or_words(self):
        pages = self.build(items=150)
        text = "\n".join(pages)
        for n in range(1, 151):
            self.assertIn(f"• Товар номер {n} — 5 шт. (лот {n})\n  Характеристики: характеристика {n}", text)
        for page in pages:
            self.assertNotRegex(page, r"(?m)^•\s*$")
            self.assertFalse(page.rstrip().endswith("</b>"))  # заголовок без пункта в конце страницы
        self.assertIn("(продолжение)", pages[1])

    def test_very_long_single_bullet_splits_on_word_boundaries(self):
        words = ["слово%d" % n for n in range(3000)]
        pages = full_analysis.build_full_analysis_messages(
            {"summary": "s", "missing_information": [" ".join(words)]}, None)
        self.assertGreater(len(pages), 1)
        for page in pages:
            self.assertLessEqual(len(page), LIMIT)
        joined = " ".join(re.findall(r"слово\d+", "\n".join(pages)))
        self.assertEqual(joined, " ".join(words))

    def test_all_requirements_barriers_missing_present(self):
        text = "\n".join(self.build())
        for n in range(1, 16):
            self.assertIn(f"Требование {n}", text)
        for n in range(12):
            self.assertIn(f"Вопрос {n}", text)
        for expected in ("Обеспечение квалификации — 15% от цены закупки",
                         "Обеспечение исполнения договора — 10% от цены закупки", "Нужна лицензия"):
            self.assertIn(expected, text)
        self.assertNotIn("bid_security", text)
        self.assertNotIn("contract_security", text)

    def test_financial_barrier_human_readable(self):
        text = "\n".join(full_analysis.build_full_analysis_messages(big_result(participation_barriers=[
            {"type": "bid_security", "description": "15% от цены закупки", "severity": "high", "evidence": []},
            {"type": "financial_requirement", "description": None, "severity": "low", "evidence": []}]), None))
        self.assertIn("• Обеспечение заявки — 15% от цены закупки.", text)
        self.assertIn("• Финансовое требование.", text)

    def test_evidence_quotes_not_shown(self):
        self.assertNotIn("EVIDENCE-QUOTE", "\n".join(self.build()))

    def test_html_escaping(self):
        text = "\n".join(self.build())
        self.assertIn("Ноутбуки &lt;15\"&gt; &amp; аксессуары", text)
        self.assertIn("ООО \"Ромашка\" &lt;Филиал&gt; &amp; Ко", text)
        self.assertIn('<a href="https://example.test/resource/1?a=1&amp;b=2">Открыть тендер</a>', text)
        self.assertNotIn("<Филиал>", text)

    def test_sections_and_conditionals(self):
        text = "\n".join(self.build())
        for expected in ("Краткое резюме:", "• Первая причина.", "• Третья причина.",
                         "• Лот 1: Первый лот (позиций: 3)", "980 000 AMD", "10.10.2026 11:00",
                         "Уверенность анализа:", "Официальная ссылка:"):
            self.assertIn(expected, text)
        self.assertNotIn("Конфликты источников", text)
        conflict = {"sources": ["announcement", "document"], "conflict_description": "Разные сроки", "impact": "Риск"}
        with_conflict = "\n".join(self.build(source_conflicts=[conflict]))
        self.assertIn("Разные сроки (источники: объявление, документы) Влияние: Риск", with_conflict)

    def test_minimal_result_single_message(self):
        pages = full_analysis.build_full_analysis_messages({"summary": "Кратко", "confidence": "low"}, None)
        self.assertEqual(len(pages), 1)
        self.assertTrue(pages[0].startswith("📋 Полный анализ тендера"))
        self.assertNotIn("None", pages[0])

    def test_worst_case_every_page_within_limit(self):
        long = "<&>" * 300
        pages = full_analysis.build_full_analysis_messages(
            {"summary": long, "why_interesting": long, "missing_information": [long] * 30,
             "participation_barriers": [{"description": long, "type": "license"}] * 10,
             "procurement": {"subject": long, "items": [{"item_name": long, "key_specifications": [long] * 3}] * 20}},
            self.URL)
        self.assertTrue(all(len(p) <= LIMIT for p in pages))


class ClientTests(unittest.TestCase):
    def response(self, status, payload):
        resp = mock.Mock(status_code=status)
        resp.json.return_value = payload
        return resp

    def client(self, post):
        return TelegramClient(TOKEN, CHAT, timeout=5, post=post)

    def test_send_message_with_reply_markup(self):
        post = mock.Mock(return_value=self.response(200, {"ok": True, "result": {"message_id": 5}}))
        markup = message.build_reply_markup(3, "https://e.test/1")
        self.assertEqual(self.client(post).send_message("hi", reply_markup=markup), SendResult(True, 5))
        self.assertEqual(post.call_args.kwargs["json"]["reply_markup"], markup)
        self.assertEqual(post.call_args.kwargs["json"]["parse_mode"], "HTML")
        post.reset_mock()
        self.client(post).send_message("hi")
        self.assertNotIn("reply_markup", post.call_args.kwargs["json"])

    def test_get_updates_and_answer_payloads(self):
        post = mock.Mock(return_value=self.response(200, {"ok": True, "result": [{"update_id": 7}]}))
        result = self.client(post).get_updates(offset=8, timeout=20)
        self.assertEqual(result, UpdatesResult(True, [{"update_id": 7}]))
        self.assertTrue(post.call_args.args[0].endswith("/getUpdates"))
        self.assertEqual(post.call_args.kwargs["json"],
                         {"timeout": 20, "offset": 8, "allowed_updates": ["callback_query"]})
        self.assertGreater(post.call_args.kwargs["timeout"], 20)
        post = mock.Mock(return_value=self.response(200, {"ok": True, "result": True}))
        self.assertTrue(self.client(post).answer_callback_query("cb", "ok").ok)
        self.assertTrue(post.call_args.args[0].endswith("/answerCallbackQuery"))
        self.assertEqual(post.call_args.kwargs["json"], {"callback_query_id": "cb", "text": "ok"})

    def test_http_failures_handled_and_token_redacted(self):
        posts = [
            mock.Mock(side_effect=requests.ConnectionError(f"pool /bot{TOKEN}/getUpdates")),
            mock.Mock(side_effect=requests.Timeout(f"timeout /bot{TOKEN}/getUpdates")),
            mock.Mock(side_effect=ValueError(TOKEN)),
            mock.Mock(return_value=self.response(409, {"ok": False, "description": f"Conflict {TOKEN}"})),
            mock.Mock(return_value=self.response(200, {"ok": False, "description": "bad"})),
            mock.Mock(return_value=self.response(200, None)),
        ]
        for post in posts:
            for call in (lambda c: c.get_updates(timeout=1), lambda c: c.answer_callback_query("x"),
                         lambda c: c.send_message("x")):
                result = call(self.client(post))
                self.assertFalse(result.ok)
                self.assertNotIn(TOKEN, result.error)
        self.assertNotIn(TOKEN, repr(self.client(mock.Mock())))


class WorkerTests(DeliveryCase):
    def run_worker(self, batches):
        client = FakeClient()
        polls = []
        results = iter(batches)

        def get_updates(offset=None, timeout=None):
            polls.append(offset)
            return next(results)

        client.get_updates = get_updates
        sleeps = []
        callback_worker.run_worker(client, CHAT, self.db_path, sleep=sleeps.append,
                                   should_continue=lambda: len(polls) < len(batches))
        return client, polls, sleeps

    def sent_delivery(self):
        self.deep.default_outcome = big_result(items=3)
        url = self.analyzed()
        self.deliver(FakeClient())
        return self.delivery_row(url)

    def test_offset_moves_to_update_id_plus_one(self):
        row = self.sent_delivery()
        batches = [UpdatesResult(True, [callback_update(f"full:{row['id']}", update_id=10),
                                        callback_update("x:1", update_id=11)]),
                   UpdatesResult(True, [{"update_id": 12, "message": {}}]),  # не callback_query: пропуск
                   UpdatesResult(True, [])]
        client, polls, _ = self.run_worker(batches)
        self.assertEqual(polls, [None, 12, 13])
        self.assertEqual(len(client.texts), 1)
        self.assertEqual([a[0] for a in client.answers], ["cb10", "cb11"])

    def test_handler_exception_does_not_stop_worker_and_offset_advances(self):
        row = self.sent_delivery()
        with mock.patch.object(callbacks, "handle_callback_query", side_effect=RuntimeError("boom")):
            with self.assertLogs(callback_worker.logger, "ERROR"):
                _, polls, _ = self.run_worker([UpdatesResult(True, [callback_update(f"full:{row['id']}", update_id=5)]),
                                               UpdatesResult(True, [])])
        self.assertEqual(polls, [None, 6])

    def test_network_failure_backs_off_and_retries_with_same_offset(self):
        batches = [UpdatesResult(False, error="network_error"), UpdatesResult(False, error="http_502"),
                   UpdatesResult(True, [{"update_id": 3}]), UpdatesResult(False, error="x"), UpdatesResult(True, [])]
        with self.assertLogs(callback_worker.logger, "ERROR"):
            _, polls, sleeps = self.run_worker(batches)
        self.assertEqual(polls, [None, None, None, 4, 4])
        self.assertEqual(sleeps, [2, 4, 2])  # растёт, после успеха сбрасывается

    def test_main_fails_fast_without_config_and_without_network(self):
        factory = mock.Mock()
        for env in ({}, {"TELEGRAM_BOT_TOKEN": TOKEN}, {"TELEGRAM_CHAT_ID": CHAT}):
            self.assertEqual(callback_worker.main(env, factory, self.db_path), 2)
        factory.assert_not_called()

    def test_main_keyboard_interrupt_is_clean(self):
        client = mock.Mock()
        client.get_updates.side_effect = KeyboardInterrupt
        env = {"TELEGRAM_BOT_TOKEN": TOKEN, "TELEGRAM_CHAT_ID": CHAT}
        self.assertEqual(callback_worker.main(env, lambda token, chat: client, self.db_path), 0)

    def test_token_absent_from_logs(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        root = logging.getLogger()
        root.addHandler(handler)
        old_level = root.level
        root.setLevel(logging.DEBUG)
        self.addCleanup(lambda: (root.removeHandler(handler), root.setLevel(old_level)))
        post = mock.Mock(side_effect=requests.ConnectionError(f"pool /bot{TOKEN}/getUpdates"))
        client = TelegramClient(TOKEN, CHAT, post=post)
        remaining = iter([True, True, False])
        callback_worker.run_worker(client, CHAT, self.db_path, sleep=lambda s: None,
                                   should_continue=lambda: next(remaining))
        self.assertIn("network_error", stream.getvalue())
        self.assertNotIn(TOKEN, stream.getvalue())

    def test_monitor_does_not_listen_for_callbacks(self):
        text = (SRC.parent / "monitor.py").read_text(encoding="utf-8")
        for forbidden in ("get_updates", "callback_worker", "callbacks"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
