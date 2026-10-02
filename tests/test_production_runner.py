"""
Тесты production runner: fake monitor / fake worker / fake clock, без сети, без OpenAI, без реального ожидания.

    python -m unittest tests.test_production_runner -v
"""

import ast
import logging
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from src import monitor, production_runner as pr
from src.database import tender_repository

TOKEN = "123456:SECRET-TOKEN"
DEFAULT = tender_repository.DEFAULT_DB_PATH


def env(**changes) -> dict:
    base = {pr.INTERVAL_ENV: "30", "TELEGRAM_BOT_TOKEN": TOKEN, "TELEGRAM_CHAT_ID": "42"}
    base.update(changes)
    return {key: value for key, value in base.items() if value is not None}


def config(minutes=30, db_path=DEFAULT) -> pr.RunnerConfig:
    return pr.RunnerConfig(minutes, Path(db_path), TOKEN, "42")


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class ConfigTests(unittest.TestCase):
    def test_valid_config(self):
        cfg = pr.load_config(env())
        self.assertEqual(cfg.interval_minutes, 30)
        self.assertEqual(cfg.chat_id, "42")
        self.assertNotIn(TOKEN, repr(cfg))

    def test_interval_missing_rejected(self):
        with self.assertRaises(pr.ConfigError):
            pr.load_config(env(**{pr.INTERVAL_ENV: None}))

    def test_interval_zero_negative_non_integer_rejected(self):
        for bad in ("0", "-5", "1.5", "abc", "", "  "):
            with self.subTest(bad=bad), self.assertRaises(pr.ConfigError):
                pr.load_config(env(**{pr.INTERVAL_ENV: bad}))

    def test_missing_telegram_token_and_chat_id_rejected(self):
        for name in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
            with self.subTest(name=name), self.assertRaises(pr.ConfigError) as ctx:
                pr.load_config(env(**{name: None}))
            self.assertIn(name, str(ctx.exception))

    def test_main_config_error_exits_before_network(self):
        worker, factory = mock.Mock(), mock.Mock()
        for name in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", pr.INTERVAL_ENV):
            with self.subTest(name=name):
                code = pr.main(env(**{name: None}), worker_fn=worker, client_factory=factory, monitor_fn=mock.Mock())
                self.assertEqual(code, pr.EXIT_CONFIG_ERROR)
        worker.assert_not_called()
        factory.assert_not_called()

    def test_database_path_resolves_and_must_match_monitor_db(self):
        self.assertEqual(pr.load_config(env()).db_path, DEFAULT)  # не задан -> тот же default, что у monitor
        with tempfile.TemporaryDirectory() as tmp:
            other = Path(tmp) / "x.db"
            # monitor_db_path == DATABASE_PATH -> принято, путь берётся из resolver
            cfg = pr.load_config(env(DATABASE_PATH=str(other)), monitor_db_path=other)
            self.assertEqual(cfg.db_path, other)
            # расхождение monitor и callback БД -> fail-fast
            with self.assertRaises(pr.ConfigError):
                pr.load_config(env(DATABASE_PATH=str(other)))


class SchedulerTests(unittest.TestCase):
    def test_immediate_run_then_interval_run(self):
        stop = threading.Event()
        clock = FakeClock()
        calls, waits = [], []

        def monitor_fn():
            calls.append(clock.now)

        def wait(timeout):
            waits.append(timeout)
            clock.now += timeout
            if len(calls) >= 2:
                stop.set()
            return stop.is_set()

        pr.MonitorScheduler(monitor_fn, 1800, stop, clock=clock, wait=wait).run_forever()
        self.assertEqual(calls, [0.0, 1800.0])
        self.assertEqual(waits, [1800, 1800])

    def test_monitor_exception_does_not_kill_scheduler(self):
        stop = threading.Event()
        clock = FakeClock()
        attempts = []

        def monitor_fn():
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("boom")

        def wait(timeout):
            clock.now += timeout
            if len(attempts) >= 2:
                stop.set()
            return stop.is_set()

        with self.assertLogs("src.production_runner", level="ERROR"):
            pr.MonitorScheduler(monitor_fn, 60, stop, clock=clock, wait=wait).run_forever()
        self.assertEqual(len(attempts), 2)

    def test_long_monitor_skips_overlapping_ticks_without_backlog(self):
        stop = threading.Event()
        clock = FakeClock()
        starts, waits = [], []

        def monitor_fn():
            starts.append(clock.now)
            if len(starts) == 1:
                clock.now += 250  # run длиннее 4 интервалов по 60с -> тики 60,120,180,240 пропущены

        def wait(timeout):
            waits.append(timeout)
            clock.now += timeout
            if len(starts) >= 2:
                stop.set()
            return stop.is_set()

        with self.assertLogs("src.production_runner", level="WARNING") as logs:
            pr.MonitorScheduler(monitor_fn, 60, stop, clock=clock, wait=wait).run_forever()
        self.assertEqual(sum(pr.OVERLAP_WARNING in line for line in logs.output), 4)
        self.assertEqual(starts, [0.0, 300.0])  # ровно один следующий run на ближайшем тике, без догоняющих
        self.assertEqual(waits, [50.0, 60.0])  # сетка сохранена: 300, затем 360

    def test_overlapping_run_once_is_rejected_by_lock(self):
        stop = threading.Event()
        inner = []
        scheduler = None

        def monitor_fn():
            inner.append(scheduler.run_once())  # повторный вход во время run'а

        scheduler = pr.MonitorScheduler(monitor_fn, 60, stop)
        with self.assertLogs("src.production_runner", level="WARNING") as logs:
            self.assertTrue(scheduler.run_once())
        self.assertEqual(inner, [False])
        self.assertTrue(any(pr.OVERLAP_WARNING in line for line in logs.output))

    def test_stop_event_interrupts_wait(self):
        stop = threading.Event()
        calls = []
        scheduler = pr.MonitorScheduler(lambda: calls.append(1), 3600, stop)  # реальный Event.wait
        thread = threading.Thread(target=scheduler.run_forever)
        thread.start()
        stop.set()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(calls), 1)


class RunTests(unittest.TestCase):
    def run_runner(self, cfg=None, monitor_fn=None, worker_fn=None, **kwargs):
        factory = mock.Mock(return_value="client")
        init = mock.Mock()
        code = pr.run(
            cfg or config(), monitor_fn=monitor_fn or (lambda: {"new_count": 1}),
            worker_fn=worker_fn or (lambda *a, **k: None), client_factory=factory, init_fn=init, **kwargs,
        )
        return code, factory, init

    def test_startup_starts_worker_and_runs_monitor_once_immediately(self):
        stop = threading.Event()
        monitor_calls, worker_calls = [], []

        def monitor_fn():
            monitor_calls.append(1)
            stop.set()
            return {"new_count": 1}

        def worker_fn(client, chat_id, db_path=None, should_continue=None):
            worker_calls.append((client, chat_id, db_path))
            while should_continue():
                stop.wait(0.01)

        code, factory, init = self.run_runner(monitor_fn=monitor_fn, worker_fn=worker_fn, stop_event=stop)
        self.assertEqual(code, pr.EXIT_OK)
        self.assertEqual(len(monitor_calls), 1)
        self.assertEqual(worker_calls, [("client", "42", DEFAULT)])  # test 14: тот же DATABASE_PATH
        init.assert_called_once_with(DEFAULT)  # схемы инициализируются тем же путём
        factory.assert_called_once_with(TOKEN, "42")

    def test_callback_fatal_exception_returns_nonzero_and_stops_scheduler(self):
        def worker_fn(*args, **kwargs):
            raise RuntimeError("fatal")

        with self.assertLogs("src.production_runner", level="ERROR"):
            code, _, _ = self.run_runner(worker_fn=worker_fn, monitor_fn=lambda: None)
        self.assertEqual(code, pr.EXIT_FATAL)
        self.assertEqual([t for t in threading.enumerate() if t.name == "monitor-scheduler"], [])

    def test_keyboard_interrupt_is_clean_shutdown(self):
        def worker_fn(*args, **kwargs):
            raise KeyboardInterrupt

        code, _, _ = self.run_runner(worker_fn=worker_fn, monitor_fn=lambda: None)
        self.assertEqual(code, pr.EXIT_OK)
        self.assertEqual([t for t in threading.enumerate() if t.name == "monitor-scheduler"], [])

    def test_stop_event_ends_worker_loop_cleanly(self):
        stop = threading.Event()

        def worker_fn(client, chat_id, db_path=None, should_continue=None):
            stop.set()  # как SIGTERM
            self.assertFalse(should_continue())

        code, _, _ = self.run_runner(worker_fn=worker_fn, stop_event=stop)
        self.assertEqual(code, pr.EXIT_OK)

    def test_monitor_uses_same_database_path_as_runner(self):
        self.assertEqual(pr.load_config(env()).db_path, tender_repository.DEFAULT_DB_PATH)

    def test_logs_do_not_contain_secrets(self):
        stop = threading.Event()
        with self.assertLogs(level="INFO") as logs:
            self.run_runner(worker_fn=lambda *a, **k: stop.set(), monitor_fn=lambda: {"new_count": 2}, stop_event=stop)
            logging.getLogger("src.production_runner").info("sentinel")
        text = "\n".join(logs.output)
        self.assertIn("Production runner started", text)
        self.assertIn("Monitor run started", text)
        self.assertIn("Monitor run finished", text)
        self.assertNotIn(TOKEN, text)
        self.assertNotIn("SECRET", text)


class MonitorCliTests(unittest.TestCase):
    def test_python_m_src_monitor_entrypoint_unchanged(self):
        self.assertTrue(callable(monitor.main))
        self.assertTrue(callable(monitor.run_monitor))
        source = Path(monitor.__file__).read_text(encoding="utf-8")
        self.assertIn('if __name__ == "__main__":\n    main()', source)

    def test_runner_has_no_openai_or_network_imports(self):
        tree = ast.parse(Path(pr.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        self.assertFalse([name for name in imported if "openai" in name or name.startswith("requests")])


if __name__ == "__main__":
    unittest.main()
