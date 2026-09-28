"""
Тесты src.ai.golden_set на временной SQLite БД и временных JSON-файлах: формат и
валидация golden set, build/hydrate (hash match/stale), read-only гарантии и CLI.
Production data/tenders.db и сеть блокируются общей test-only защитой (tests/safety_guards.py).

Запуск из корня проекта:
    python -m unittest tests.test_golden_set -v
"""

import copy
import hashlib
import io
import json
import socket
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from src.ai import golden_set
from src.ai import tender_context as tender_context_module
from src.database import announcement_repository, document_repository, enrichment_repository
from tests import safety_guards
from tests.test_evaluation_dataset import make_announcement, make_enrichment

URL_1 = "https://example.test/resource/1"
URL_2 = "https://example.test/resource/2"


def make_expected(**overrides) -> dict:
    expected = {
        "relevance_status": "relevant",
        "opportunity_type": "procurement",
        "category": "computers",
        "expected_reason": "Поставка компьютеров: физический товар.",
        "notes": None,
    }
    expected.update(overrides)
    return expected


def make_case(number=1, **expected_overrides) -> dict:
    return {
        "case_id": f"case-{number}",
        "resource_url": f"https://example.test/resource/{number}",
        "input_hash": hashlib.sha256(str(number).encode()).hexdigest(),
        "title": f"Тендер {number}",
        "expected": make_expected(**expected_overrides),
    }


class ValidateGoldenSetTests(unittest.TestCase):
    def assertValid(self, cases):
        self.assertEqual(golden_set.validate_golden_set(cases), [])

    def assertInvalid(self, cases, fragment=None):
        errors = golden_set.validate_golden_set(cases)
        self.assertTrue(errors, "ожидались ошибки валидации")
        if fragment is not None:
            self.assertIn(fragment, "\n".join(errors))

    def test_empty_golden_set_is_valid(self):
        self.assertValid([])

    def test_not_a_list_rejected(self):
        self.assertInvalid({"case_id": "x"}, "списком")

    def test_relevant_procurement_accepted(self):
        self.assertValid([make_case(1, relevance_status="relevant", opportunity_type="procurement")])

    def test_not_relevant_other_service_accepted(self):
        self.assertValid([make_case(
            1, relevance_status="not_relevant", opportunity_type="other_service", category="construction_works",
        )])

    def test_maybe_unclear_accepted_with_null_category(self):
        self.assertValid([make_case(1, relevance_status="maybe", opportunity_type="unclear", category=None)])

    def test_not_relevant_unrelated_accepted_with_null_category(self):
        self.assertValid([make_case(1, relevance_status="not_relevant", opportunity_type="unrelated", category=None)])

    def test_relevant_unclear_rejected_by_schema_rule(self):
        self.assertInvalid(
            [make_case(1, relevance_status="relevant", opportunity_type="unclear", category=None)],
            "relevance schema",
        )

    def test_duplicate_case_id_rejected(self):
        first, second = make_case(1), make_case(2)
        second["case_id"] = first["case_id"]
        self.assertInvalid([first, second], "дубликат case_id")

    def test_duplicate_resource_url_rejected(self):
        first, second = make_case(1), make_case(2)
        second["resource_url"] = first["resource_url"]
        self.assertInvalid([first, second], "дубликат resource_url")

    def test_invalid_relevance_status_rejected(self):
        self.assertInvalid([make_case(1, relevance_status="definitely")], "relevance_status")

    def test_invalid_opportunity_type_rejected(self):
        self.assertInvalid([make_case(1, opportunity_type="magic")], "opportunity_type")

    def test_logistics_rejected_in_procurement_golden_set(self):
        for opportunity_type in ("logistics", "logistics_and_procurement"):
            with self.subTest(opportunity_type=opportunity_type):
                self.assertInvalid(
                    [make_case(1, opportunity_type=opportunity_type)], "procurement-only",
                )

    def test_global_relevance_schema_still_knows_logistics(self):
        # Ограничение действует только в golden set: глобальная схема не менялась.
        from src.ai import relevance_schema
        self.assertIn("logistics", relevance_schema.OPPORTUNITY_TYPES)
        self.assertIn("logistics_and_procurement", relevance_schema.OPPORTUNITY_TYPES)

    def test_empty_expected_reason_rejected(self):
        for reason in ("", "   ", None):
            with self.subTest(reason=reason):
                self.assertInvalid([make_case(1, expected_reason=reason)], "expected_reason")

    def test_null_category_rejected_for_procurement(self):
        self.assertInvalid([make_case(1, opportunity_type="procurement", category=None)], "category")

    def test_null_category_rejected_for_other_service(self):
        self.assertInvalid(
            [make_case(1, relevance_status="not_relevant", opportunity_type="other_service", category=None)],
            "category",
        )

    def test_blank_category_rejected(self):
        self.assertInvalid([make_case(1, category="  ")], "category")

    def test_missing_case_field_rejected(self):
        for field in golden_set.REQUIRED_CASE_FIELDS:
            with self.subTest(field=field):
                case = make_case(1)
                del case[field]
                self.assertInvalid([case], field)

    def test_missing_expected_field_rejected(self):
        case = make_case(1)
        del case["expected"]["notes"]
        self.assertInvalid([case], "notes")

    def test_unknown_expected_field_rejected(self):
        case = make_case(1)
        case["expected"]["relevance_statuss"] = "relevant"
        self.assertInvalid([case], "неизвестные поля")

    def test_bad_input_hash_rejected(self):
        for value in ("abc", "", None, "Z" * 64):
            with self.subTest(value=value):
                case = make_case(1)
                case["input_hash"] = value
                self.assertInvalid([case], "input_hash")

    def test_non_dict_case_rejected(self):
        self.assertInvalid(["not a case"], "dict")

    def test_all_errors_are_reported(self):
        errors = golden_set.validate_golden_set([
            make_case(1, relevance_status="bad"), make_case(2, expected_reason=""),
        ])
        self.assertGreaterEqual(len(errors), 2)

    def test_validation_does_not_mutate_input(self):
        cases = [make_case(1), make_case(2)]
        original = copy.deepcopy(cases)
        golden_set.validate_golden_set(cases)
        self.assertEqual(cases, original)


class FileIoTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "evaluation" / "golden.json"

    def test_round_trip(self):
        cases = [make_case(1), make_case(2, relevance_status="maybe", opportunity_type="unclear", category=None)]

        saved_path = golden_set.save_golden_set(cases, self.path)
        loaded = golden_set.load_golden_set(self.path)

        self.assertEqual(saved_path, self.path)
        self.assertEqual(loaded, cases)

    def test_round_trip_preserves_non_ascii(self):
        cases = [make_case(1)]
        cases[0]["title"] = "Համակարգիչների մատակարարում / Поставка"
        golden_set.save_golden_set(cases, self.path)

        self.assertIn("Համակարգիչների", self.path.read_text(encoding="utf-8"))
        self.assertEqual(golden_set.load_golden_set(self.path), cases)

    def test_save_rejects_invalid_set_and_writes_nothing(self):
        with self.assertRaises(ValueError):
            golden_set.save_golden_set([make_case(1, expected_reason="")], self.path)
        self.assertFalse(self.path.exists())

    def test_invalid_save_keeps_existing_file(self):
        golden_set.save_golden_set([make_case(1)], self.path)
        before = self.path.read_bytes()

        with self.assertRaises(ValueError):
            golden_set.save_golden_set([make_case(1), make_case(1)], self.path)

        self.assertEqual(self.path.read_bytes(), before)

    def test_no_temp_files_left_behind(self):
        golden_set.save_golden_set([make_case(1)], self.path)
        self.assertEqual([p.name for p in self.path.parent.iterdir()], ["golden.json"])

    def test_load_missing_file_raises_and_does_not_create(self):
        with self.assertRaises(FileNotFoundError):
            golden_set.load_golden_set(self.path)
        self.assertFalse(self.path.exists())
        self.assertFalse(self.path.parent.exists())

    def test_load_invalid_json_raises_value_error(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(ValueError):
            golden_set.load_golden_set(self.path)

    def test_load_non_list_raises_value_error(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text('{"a": 1}', encoding="utf-8")
        with self.assertRaises(ValueError):
            golden_set.load_golden_set(self.path)

    def test_default_path_is_evaluation_golden_set_procurement(self):
        self.assertEqual(
            golden_set.DEFAULT_GOLDEN_SET_PATH.parts[-2:], ("evaluation", "golden_set_procurement.json"),
        )


class GoldenSetDbTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_dir = Path(self._tmp.name)
        self.db_path = self.tmp_dir / "test.db"
        self.golden_path = self.tmp_dir / "golden.json"
        document_repository.init_db(self.db_path)
        enrichment_repository.init_db(self.db_path)

    def add(self, number, **enrichment_overrides) -> str:
        announcement = make_announcement(number)
        announcement_repository.save_announcement(announcement, self.db_path)
        enrichment_repository.save_enrichment(
            announcement["resource_url"], make_enrichment(**enrichment_overrides), self.db_path,
        )
        return announcement["resource_url"]

    def current_hash(self, resource_url) -> str:
        context = tender_context_module.build_tender_context(resource_url, db_path=self.db_path)
        return tender_context_module.compute_input_hash(context)

    def db_file_hash(self) -> str:
        return hashlib.sha256(self.db_path.read_bytes()).hexdigest()

    def write_golden(self, cases) -> bytes:
        golden_set.save_golden_set(cases, self.golden_path)
        return self.golden_path.read_bytes()

    def run_cli(self, *argv) -> tuple:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = golden_set.main(["--path", str(self.golden_path), "--db-path", str(self.db_path), *argv])
        return code, buffer.getvalue()


class BuildGoldenCaseTests(GoldenSetDbTestCase):
    def test_uses_current_input_hash(self):
        url = self.add(1)

        case = golden_set.build_golden_case(url, make_expected(), db_path=self.db_path)

        self.assertEqual(case["input_hash"], self.current_hash(url))

    def test_compact_case_shape(self):
        url = self.add(1)
        case = golden_set.build_golden_case(url, make_expected(), db_path=self.db_path)

        self.assertEqual(list(case), ["case_id", "resource_url", "input_hash", "title", "expected"])
        self.assertEqual(case["resource_url"], url)
        self.assertEqual(case["title"], "Тендер 1")
        self.assertTrue(case["case_id"].startswith("case-"))
        self.assertEqual(golden_set.validate_golden_set([case]), [])

    def test_does_not_copy_triage_context(self):
        url = self.add(1)
        case = golden_set.build_golden_case(url, make_expected(), db_path=self.db_path)

        self.assertNotIn("triage_context", case)
        self.assertNotIn("document_previews", json.dumps(case))

    def test_case_id_matches_evaluation_dataset(self):
        from src.ai import evaluation_dataset
        url = self.add(1)

        case = golden_set.build_golden_case(url, make_expected(), db_path=self.db_path)

        self.assertEqual(case["case_id"], evaluation_dataset._case_id(url))

    def test_category_and_notes_default_to_null(self):
        url = self.add(1)
        expected = {
            "relevance_status": "maybe", "opportunity_type": "unclear",
            "expected_reason": "Контекста недостаточно.",
        }

        case = golden_set.build_golden_case(url, expected, db_path=self.db_path)

        self.assertIsNone(case["expected"]["category"])
        self.assertIsNone(case["expected"]["notes"])

    def test_invalid_expected_rejected(self):
        url = self.add(1)
        for expected in (
            make_expected(opportunity_type="logistics"),
            make_expected(expected_reason=""),
            make_expected(relevance_status="nope"),
            {**make_expected(), "extra": 1},
            "relevant",
        ):
            with self.subTest(expected=expected):
                with self.assertRaises(ValueError):
                    golden_set.build_golden_case(url, expected, db_path=self.db_path)

    def test_unknown_resource_url_raises(self):
        with self.assertRaises(ValueError):
            golden_set.build_golden_case("https://example.test/none", make_expected(), db_path=self.db_path)

    def test_does_not_mutate_expected_argument(self):
        url = self.add(1)
        expected = {"relevance_status": "not_relevant", "opportunity_type": "unrelated",
                    "expected_reason": "Не товар."}
        original = dict(expected)

        golden_set.build_golden_case(url, expected, db_path=self.db_path)

        self.assertEqual(expected, original)


class HydrateGoldenCaseTests(GoldenSetDbTestCase):
    def test_detects_matching_hash(self):
        url = self.add(1)
        case = golden_set.build_golden_case(url, make_expected(), db_path=self.db_path)

        result = golden_set.hydrate_golden_case(case, db_path=self.db_path)

        self.assertIs(result["hash_match"], True)
        self.assertEqual(result["stored_input_hash"], result["current_input_hash"])

    def test_detects_stale_hash(self):
        url = self.add(1)
        case = golden_set.build_golden_case(url, make_expected(), db_path=self.db_path)

        enrichment_repository.save_enrichment(url, make_enrichment(description="Изменённое описание"), self.db_path)
        result = golden_set.hydrate_golden_case(case, db_path=self.db_path)

        self.assertIs(result["hash_match"], False)
        self.assertEqual(result["stored_input_hash"], case["input_hash"])
        self.assertEqual(result["current_input_hash"], self.current_hash(url))
        self.assertNotEqual(result["stored_input_hash"], result["current_input_hash"])

    def test_stale_hash_is_not_silently_replaced(self):
        url = self.add(1)
        case = golden_set.build_golden_case(url, make_expected(), db_path=self.db_path)
        original = copy.deepcopy(case)

        enrichment_repository.save_enrichment(url, make_enrichment(description="Изменённое описание"), self.db_path)
        result = golden_set.hydrate_golden_case(case, db_path=self.db_path)

        self.assertEqual(case, original)
        self.assertEqual(result["golden_case"]["input_hash"], original["input_hash"])

    def test_returns_current_evaluation_case_with_triage_context(self):
        url = self.add(1)
        case = golden_set.build_golden_case(url, make_expected(), db_path=self.db_path)

        result = golden_set.hydrate_golden_case(case, db_path=self.db_path)

        current = result["current_case"]
        self.assertEqual(current["case_id"], case["case_id"])
        self.assertEqual(current["input_hash"], result["current_input_hash"])
        self.assertIn("triage_context", current)

    def test_invalid_case_rejected(self):
        url = self.add(1)
        case = golden_set.build_golden_case(url, make_expected(), db_path=self.db_path)
        case["expected"]["expected_reason"] = ""
        with self.assertRaises(ValueError):
            golden_set.hydrate_golden_case(case, db_path=self.db_path)

    def test_tender_missing_from_db_raises(self):
        with self.assertRaises(ValueError):
            golden_set.hydrate_golden_case(make_case(9), db_path=self.db_path)


class CheckGoldenSetTests(GoldenSetDbTestCase):
    def test_statuses_and_one_failure_does_not_stop_others(self):
        ok_url = self.add(1)
        stale_url = self.add(2)
        cases = [
            golden_set.build_golden_case(ok_url, make_expected(), db_path=self.db_path),
            golden_set.build_golden_case(stale_url, make_expected(), db_path=self.db_path),
            make_case(9),  # в БД нет
        ]
        enrichment_repository.save_enrichment(stale_url, make_enrichment(description="Другое"), self.db_path)

        results = golden_set.check_golden_set(cases, db_path=self.db_path)

        self.assertEqual(
            [result["status"] for result in results],
            [golden_set.HASH_OK, golden_set.HASH_STALE, golden_set.HASH_MISSING],
        )

    def test_missing_db_file_is_error_status_and_not_created(self):
        missing_db = self.tmp_dir / "nope.db"

        results = golden_set.check_golden_set([make_case(1)], db_path=missing_db)

        self.assertEqual(results[0]["status"], golden_set.HASH_ERROR)
        self.assertFalse(missing_db.exists())


class NoMutationTests(GoldenSetDbTestCase):
    def test_build_and_hydrate_do_not_change_db_file(self):
        url = self.add(1)
        before = self.db_file_hash()

        case = golden_set.build_golden_case(url, make_expected(), db_path=self.db_path)
        golden_set.hydrate_golden_case(case, db_path=self.db_path)
        golden_set.check_golden_set([case], db_path=self.db_path)

        self.assertEqual(self.db_file_hash(), before)

    def test_missing_db_is_not_created(self):
        missing_db = self.tmp_dir / "nope.db"
        with self.assertRaises(Exception):
            golden_set.build_golden_case(URL_1, make_expected(), db_path=missing_db)
        self.assertFalse(missing_db.exists())

    def test_cli_does_not_change_db_file(self):
        url = self.add(1)
        case = golden_set.build_golden_case(url, make_expected(), db_path=self.db_path)
        self.write_golden([case])
        before = self.db_file_hash()

        self.run_cli("--show")
        self.run_cli("--validate")
        self.run_cli("--hydrate")
        self.run_cli("--resource-url", url)

        self.assertEqual(self.db_file_hash(), before)

    def test_no_network_access(self):
        url = self.add(1)
        with mock.patch.object(socket.socket, "connect", side_effect=AssertionError("network")) as connect, \
                mock.patch.object(socket, "getaddrinfo", side_effect=AssertionError("network")) as resolve:
            case = golden_set.build_golden_case(url, make_expected(), db_path=self.db_path)
            golden_set.hydrate_golden_case(case, db_path=self.db_path)
            self.write_golden([case])
            self.run_cli("--show")
            self.run_cli("--hydrate")

        connect.assert_not_called()
        resolve.assert_not_called()
        self.assertEqual(safety_guards.violations(), [])


class ProductionDbBlockedTests(unittest.TestCase):
    """Без db_path модуль читает production data/tenders.db — test-only защита это блокирует."""

    def test_build_against_production_db_is_blocked(self):
        with safety_guards.expect_violation(self, safety_guards.DATABASE_MESSAGE):
            golden_set.build_golden_case(URL_1, make_expected())

    def test_hydrate_against_production_db_is_blocked(self):
        with safety_guards.expect_violation(self, safety_guards.DATABASE_MESSAGE):
            golden_set.hydrate_golden_case(make_case(1))

    def test_cli_resource_url_against_production_db_is_blocked(self):
        with redirect_stdout(io.StringIO()):
            with safety_guards.expect_violation(self, safety_guards.DATABASE_MESSAGE):
                golden_set.main(["--resource-url", URL_1])


class CliTests(GoldenSetDbTestCase):
    def test_validate_success(self):
        self.write_golden([make_case(1), make_case(2)])

        code, output = self.run_cli("--validate")

        self.assertEqual(code, 0)
        self.assertIn("валиден: 2", output)

    def test_validate_failure_returns_nonzero_and_lists_errors(self):
        self.golden_path.write_text(
            json.dumps([make_case(1, expected_reason=""), make_case(1)], ensure_ascii=False), encoding="utf-8",
        )

        code, output = self.run_cli("--validate")

        self.assertNotEqual(code, 0)
        self.assertIn("expected_reason", output)
        self.assertIn("дубликат", output)

    def test_validate_invalid_json_returns_nonzero(self):
        self.golden_path.write_text("{oops", encoding="utf-8")
        code, _ = self.run_cli("--validate")
        self.assertNotEqual(code, 0)

    def test_validate_missing_file_returns_nonzero_and_does_not_create_it(self):
        for mode in ("--validate", "--show", "--hydrate"):
            with self.subTest(mode=mode):
                code, output = self.run_cli(mode)
                self.assertNotEqual(code, 0)
                self.assertIn("не найден", output)
                self.assertFalse(self.golden_path.exists())

    def test_show_prints_compact_table(self):
        url = self.add(1)
        case = golden_set.build_golden_case(url, make_expected(category="computers"), db_path=self.db_path)
        self.write_golden([case])

        code, output = self.run_cli("--show")

        self.assertEqual(code, 0)
        for text in (case["case_id"], "Тендер 1", "relevant", "procurement", "computers", "ok"):
            self.assertIn(text, output)

    def test_show_marks_stale_case(self):
        url = self.add(1)
        case = golden_set.build_golden_case(url, make_expected(), db_path=self.db_path)
        self.write_golden([case])
        enrichment_repository.save_enrichment(url, make_enrichment(description="Другое"), self.db_path)

        code, output = self.run_cli("--show")

        self.assertEqual(code, 0)
        self.assertIn("stale", output)

    def test_hydrate_all_current_returns_zero(self):
        url = self.add(1)
        self.write_golden([golden_set.build_golden_case(url, make_expected(), db_path=self.db_path)])

        code, output = self.run_cli("--hydrate")

        self.assertEqual(code, 0)
        self.assertIn("актуальных: 1", output)

    def test_hydrate_reports_stale_and_missing_but_does_not_rewrite_json(self):
        url = self.add(1)
        cases = [golden_set.build_golden_case(url, make_expected(), db_path=self.db_path), make_case(9)]
        before = self.write_golden(cases)
        mtime_before = self.golden_path.stat().st_mtime_ns
        enrichment_repository.save_enrichment(url, make_enrichment(description="Другое"), self.db_path)

        code, output = self.run_cli("--hydrate")

        self.assertEqual(code, 1)
        self.assertIn("STALE", output)
        self.assertIn("MISSING", output)
        self.assertEqual(self.golden_path.read_bytes(), before)
        self.assertEqual(self.golden_path.stat().st_mtime_ns, mtime_before)

    def test_hydrate_refuses_invalid_golden_set(self):
        self.golden_path.write_text(json.dumps([make_case(1, expected_reason="")]), encoding="utf-8")
        code, output = self.run_cli("--hydrate")
        self.assertNotEqual(code, 0)
        self.assertIn("невалиден", output)

    def test_resource_url_prints_current_evaluation_case(self):
        url = self.add(1)

        code, output = self.run_cli("--resource-url", url)

        self.assertEqual(code, 0)
        current = json.loads(output)
        self.assertEqual(current["resource_url"], url)
        self.assertEqual(current["input_hash"], self.current_hash(url))
        self.assertIn("triage_context", current)

    def test_resource_url_does_not_require_or_create_golden_file(self):
        url = self.add(1)
        code, _ = self.run_cli("--resource-url", url)
        self.assertEqual(code, 0)
        self.assertFalse(self.golden_path.exists())

    def test_resource_url_unknown_returns_nonzero(self):
        code, output = self.run_cli("--resource-url", "https://example.test/none")
        self.assertNotEqual(code, 0)
        self.assertIn("Ошибка", output)

    def test_mode_is_required(self):
        with redirect_stdout(io.StringIO()), mock.patch("sys.stderr", new=io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                golden_set.main([])
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
