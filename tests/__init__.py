"""
Пакет тестов. При импорте включает test-only защиту (tests/safety_guards.py):
запрет реальной сети и доступа к production data/tenders.db на время test process.
"""

from tests import safety_guards

safety_guards.install()
