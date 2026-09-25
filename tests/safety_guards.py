"""
Test-only защита: в процессе тестов нельзя ходить во внешнюю сеть и открывать
production-базу data/tenders.db.

Защита ставится один раз при импорте пакета tests (см. tests/__init__.py) и действует
на весь test process. Production-код не изменяется: подменяются только функции
socket и sqlite3.connect.

Нарушение (AssertionError) фиксируется ещё и в списке нарушений: production-код
местами перехватывает Exception (например, process_announcements), и без этого
случайный сетевой вызов мог бы тихо превратиться в «ошибку одного объявления» и тест
остался бы зелёным. Обёртка над TestCase.run превращает такое нарушение в failure теста.
"""

import ipaddress
import os
import socket
import sqlite3
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import unquote, urlparse

NETWORK_MESSAGE = "Real network access is forbidden during tests"
DATABASE_MESSAGE = "Production database access is forbidden during tests"

# Независимо от src: если DEFAULT_DB_PATH в production-коде изменится, защита не ослабнет.
PRODUCTION_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "tenders.db"

LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}

_violations: list[str] = []
_installed = False


def violations() -> list[str]:
    return list(_violations)


# --- сеть ---

def _is_loopback_host(host) -> bool:
    if isinstance(host, bytes):
        host = host.decode("ascii", errors="replace")
    if not isinstance(host, str):
        return False
    host = host.strip().strip("[]").split("%", 1)[0].lower()
    if host in LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _forbid_network(target) -> None:
    _violations.append(f"{NETWORK_MESSAGE}: {target!r}")
    raise AssertionError(f"{NETWORK_MESSAGE}: {target!r}")


def _check_address(address) -> None:
    # Не tuple — путь AF_UNIX или другой не-IP адрес: это не внешняя сеть.
    if isinstance(address, tuple) and address and not _is_loopback_host(address[0]):
        _forbid_network(address)


def _install_network_guard() -> None:
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo
    real_gethostbyname = socket.gethostbyname
    real_gethostbyname_ex = socket.gethostbyname_ex
    real_create_connection = socket.create_connection

    def guarded_connect(self, address):
        _check_address(address)
        return real_connect(self, address)

    def guarded_connect_ex(self, address):
        _check_address(address)
        return real_connect_ex(self, address)

    def guarded_getaddrinfo(host, *args, **kwargs):
        # DNS-запрос к внешнему имени — тоже сетевая операция.
        if host is not None and not _is_loopback_host(host):
            _forbid_network(host)
        return real_getaddrinfo(host, *args, **kwargs)

    def guarded_gethostbyname(host):
        if not _is_loopback_host(host):
            _forbid_network(host)
        return real_gethostbyname(host)

    def guarded_gethostbyname_ex(host):
        if not _is_loopback_host(host):
            _forbid_network(host)
        return real_gethostbyname_ex(host)

    def guarded_create_connection(address, *args, **kwargs):
        _check_address(address)
        return real_create_connection(address, *args, **kwargs)

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    socket.getaddrinfo = guarded_getaddrinfo
    socket.gethostbyname = guarded_gethostbyname
    socket.gethostbyname_ex = guarded_gethostbyname_ex
    socket.create_connection = guarded_create_connection


# --- production база ---

def _normalize(path) -> str:
    return os.path.normcase(os.path.abspath(os.path.realpath(path)))


def _database_path(database, uri: bool):
    """Путь к файлу из аргумента sqlite3.connect или None (:memory:, временная БД, не файл)."""
    if isinstance(database, bytes):
        database = os.fsdecode(database)
    if isinstance(database, os.PathLike):
        database = os.fspath(database)
    if not isinstance(database, str) or database in ("", ":memory:"):
        return None
    if uri and database.startswith("file:"):
        parsed = urlparse(database)
        database = unquote(parsed.path)
        # file:///C:/x -> "/C:/x"; file://host/x не бывает у локальных файлов.
        if os.name == "nt" and len(database) > 2 and database[0] == "/" and database[2] == ":":
            database = database[1:]
    return database


def is_production_db(database, uri: bool = False) -> bool:
    path = _database_path(database, uri)
    if path is None:
        return False
    if _normalize(path) == _normalize(PRODUCTION_DB_PATH):
        return True
    # Совпадение файла независимо от написания пути (8.3-имена, символические ссылки).
    try:
        return os.path.exists(path) and os.path.samefile(path, PRODUCTION_DB_PATH)
    except OSError:
        return False


def _install_database_guard() -> None:
    real_connect = sqlite3.connect

    def guarded_connect(database, *args, **kwargs):
        if is_production_db(database, uri=bool(kwargs.get("uri", False))):
            _violations.append(f"{DATABASE_MESSAGE}: {database!r}")
            raise AssertionError(f"{DATABASE_MESSAGE}: {database!r}")
        return real_connect(database, *args, **kwargs)

    sqlite3.connect = guarded_connect


# --- нарушения, проглоченные production-кодом ---

def _already_reported(result, test, new_violations: list[str]) -> bool:
    """Нарушение дошло до теста как обычный AssertionError — второй failure не нужен."""
    entries = [*getattr(result, "failures", []), *getattr(result, "errors", [])]
    for reported_test, traceback_text in entries:
        if getattr(reported_test, "test_case", reported_test) is test:
            if any(violation in traceback_text for violation in new_violations):
                return True
    return False


def _install_violation_check() -> None:
    real_run = unittest.TestCase.run

    def run(self, result=None):
        before = len(_violations)
        outcome = real_run(self, result)
        new = _violations[before:]
        del _violations[before:]
        if new and result is not None and not _already_reported(result, self, new):
            try:
                raise AssertionError(
                    "Guard violation was raised during the test (possibly swallowed by "
                    "production error handling): " + "; ".join(new)
                )
            except AssertionError:
                result.addFailure(self, sys.exc_info())
        return outcome

    unittest.TestCase.run = run


@contextmanager
def expect_violation(test: unittest.TestCase, message: str):
    """Для тестов самой защиты: ждёт AssertionError с message и не считает его нарушением теста."""
    before = len(_violations)
    with test.assertRaisesRegex(AssertionError, message):
        yield
    del _violations[before:]


def install() -> None:
    """Идемпотентно: повторный import/вызов не оборачивает функции второй раз."""
    global _installed
    if _installed:
        return
    _install_network_guard()
    _install_database_guard()
    _install_violation_check()
    _installed = True
