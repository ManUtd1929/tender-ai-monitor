"""
Тесты самой test-only защиты (tests/safety_guards.py).

Ни один тест не обращается к реальному внешнему адресу: адреса — из диапазона
документации 192.0.2.0/24 (TEST-NET-1), а production-база при срабатывании защиты
не открывается вовсе.

Запуск из корня проекта:
    python -m unittest tests.test_safety_guards -v
"""

import os
import socket
import sqlite3
import tempfile
import unittest
import urllib.request
from contextlib import closing
from pathlib import Path
from unittest import mock

import requests

from src.database import announcement_repository, enrichment_repository
from tests import safety_guards
from tests.safety_guards import (
    DATABASE_MESSAGE,
    NETWORK_MESSAGE,
    PRODUCTION_DB_PATH,
    expect_violation,
)

EXTERNAL_ADDRESS = ("192.0.2.1", 80)  # TEST-NET-1: не маршрутизируется даже без защиты


class NetworkGuardTest(unittest.TestCase):
    def test_external_socket_connect_is_blocked(self):
        with socket.socket() as sock, expect_violation(self, NETWORK_MESSAGE):
            sock.connect(EXTERNAL_ADDRESS)

    def test_external_socket_connect_ex_is_blocked(self):
        with socket.socket() as sock, expect_violation(self, NETWORK_MESSAGE):
            sock.connect_ex(EXTERNAL_ADDRESS)

    def test_create_connection_is_blocked(self):
        with expect_violation(self, NETWORK_MESSAGE):
            socket.create_connection(EXTERNAL_ADDRESS, timeout=1)

    def test_dns_lookup_of_external_name_is_blocked(self):
        with expect_violation(self, NETWORK_MESSAGE):
            socket.getaddrinfo("gnumner.minfin.am", 443)
        with expect_violation(self, NETWORK_MESSAGE):
            socket.gethostbyname("gnumner.minfin.am")

    def test_unmocked_requests_call_is_blocked(self):
        with expect_violation(self, NETWORK_MESSAGE):
            requests.get("http://192.0.2.1/", timeout=1)

    def test_unmocked_urllib_call_is_blocked(self):
        with expect_violation(self, NETWORK_MESSAGE):
            urllib.request.urlopen("http://192.0.2.1/", timeout=1)

    def test_loopback_connection_is_allowed(self):
        for host, family in (("127.0.0.1", socket.AF_INET), ("localhost", socket.AF_INET)):
            with self.subTest(host=host):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                    server.bind(("127.0.0.1", 0))
                    server.listen(1)
                    port = server.getsockname()[1]
                    with socket.create_connection((host, port), timeout=5) as client:
                        connection, _ = server.accept()
                        with connection:
                            client.sendall(b"ping")
                            self.assertEqual(connection.recv(4), b"ping")

    def test_mocked_http_still_works(self):
        response = mock.Mock(status_code=200, text="<html>ok</html>")
        with mock.patch("requests.get", return_value=response) as fake_get:
            result = requests.get("https://gnumner.minfin.am/", timeout=1)

        self.assertIs(result, response)
        fake_get.assert_called_once()

    def test_mocked_session_still_works(self):
        with mock.patch.object(requests.Session, "get", return_value=mock.Mock(status_code=200)):
            self.assertEqual(requests.Session().get("https://armeps.am/").status_code, 200)


class DatabaseGuardTest(unittest.TestCase):
    def test_production_db_path_is_project_data_tenders_db(self):
        self.assertEqual(PRODUCTION_DB_PATH.name, "tenders.db")
        self.assertEqual(PRODUCTION_DB_PATH.parent.name, "data")
        self.assertTrue((PRODUCTION_DB_PATH.parents[1] / "src").is_dir())

    def test_production_path_as_path_and_str_is_blocked(self):
        for database in (PRODUCTION_DB_PATH, str(PRODUCTION_DB_PATH)):
            with self.subTest(type=type(database).__name__):
                with expect_violation(self, DATABASE_MESSAGE):
                    sqlite3.connect(database)

    def test_production_path_in_other_spelling_is_blocked(self):
        data_dir = PRODUCTION_DB_PATH.parent
        spellings = (
            os.path.join(data_dir, "..", "data", "tenders.db"),
            os.path.join(data_dir, ".", "tenders.db"),
            str(PRODUCTION_DB_PATH).upper(),  # на Windows регистр не важен; иначе путь другой
        )
        for database in spellings:
            with self.subTest(database=database):
                if database == str(PRODUCTION_DB_PATH).upper() and os.name != "nt":
                    self.skipTest("регистр пути значим не на Windows")
                with expect_violation(self, DATABASE_MESSAGE):
                    sqlite3.connect(database)

    def test_production_path_as_uri_is_blocked(self):
        with expect_violation(self, DATABASE_MESSAGE):
            sqlite3.connect(f"{PRODUCTION_DB_PATH.as_uri()}?mode=ro", uri=True)

    def test_production_repository_call_without_db_path_is_blocked(self):
        # db_path=None -> DEFAULT_DB_PATH -> production-база.
        with expect_violation(self, DATABASE_MESSAGE):
            enrichment_repository.count_enrichments()
        with expect_violation(self, DATABASE_MESSAGE):
            announcement_repository.count_announcements()

    def test_blocked_connect_does_not_create_or_touch_files(self):
        exists_before = PRODUCTION_DB_PATH.exists()
        mtime_before = PRODUCTION_DB_PATH.stat().st_mtime_ns if exists_before else None

        with expect_violation(self, DATABASE_MESSAGE):
            sqlite3.connect(PRODUCTION_DB_PATH)

        self.assertEqual(PRODUCTION_DB_PATH.exists(), exists_before)
        if exists_before:
            self.assertEqual(PRODUCTION_DB_PATH.stat().st_mtime_ns, mtime_before)

    def test_memory_db_is_allowed(self):
        with closing(sqlite3.connect(":memory:")) as conn:
            self.assertEqual(conn.execute("SELECT 1").fetchone()[0], 1)

    def test_tempfile_db_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tenders.db"  # то же имя файла, но не production-каталог
            with closing(sqlite3.connect(path)) as conn, conn:
                conn.execute("CREATE TABLE t (x INTEGER)")
                conn.execute("INSERT INTO t VALUES (1)")
            with closing(sqlite3.connect(str(path))) as conn:
                self.assertEqual(conn.execute("SELECT x FROM t").fetchone()[0], 1)

    def test_repository_works_on_tempfile_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            enrichment_repository.init_db(db_path)
            self.assertEqual(enrichment_repository.count_enrichments(db_path), 0)


class ViolationTrackingTest(unittest.TestCase):
    """Нарушение, проглоченное production-кодом (except Exception), всё равно валит тест."""

    def run_inner(self, test_method) -> unittest.TestResult:
        class Inner(unittest.TestCase):
            def runTest(self):
                test_method()

        result = unittest.TestResult()
        Inner().run(result)
        return result

    def test_swallowed_network_violation_fails_the_test(self):
        def swallowing():
            try:
                socket.create_connection(EXTERNAL_ADDRESS, timeout=1)
            except Exception:
                pass  # как process_announcements: ошибка «обработана» и тест был бы зелёным

        result = self.run_inner(swallowing)

        self.assertEqual(len(result.failures), 1)
        self.assertIn(NETWORK_MESSAGE, result.failures[0][1])

    def test_swallowed_database_violation_fails_the_test(self):
        def swallowing():
            try:
                sqlite3.connect(PRODUCTION_DB_PATH)
            except Exception:
                pass

        result = self.run_inner(swallowing)

        self.assertEqual(len(result.failures), 1)
        self.assertIn(DATABASE_MESSAGE, result.failures[0][1])

    def test_unswallowed_violation_is_reported_once(self):
        def not_swallowing():
            socket.create_connection(EXTERNAL_ADDRESS, timeout=1)

        result = self.run_inner(not_swallowing)

        self.assertEqual(len(result.failures) + len(result.errors), 1)

    def test_clean_test_is_not_affected(self):
        result = self.run_inner(lambda: None)

        self.assertTrue(result.wasSuccessful())

    def test_expected_violation_is_not_left_in_the_list(self):
        before = safety_guards.violations()

        with expect_violation(self, NETWORK_MESSAGE):
            socket.getaddrinfo("gnumner.minfin.am", 443)

        self.assertEqual(safety_guards.violations(), before)

    def test_install_is_idempotent(self):
        connect = sqlite3.connect
        run = unittest.TestCase.run

        safety_guards.install()

        self.assertIs(sqlite3.connect, connect)
        self.assertIs(unittest.TestCase.run, run)


if __name__ == "__main__":
    unittest.main()
