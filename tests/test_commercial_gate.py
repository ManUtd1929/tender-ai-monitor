"""
Тесты src.ai.value_facts (детерминированные факты стоимости/количества, provenance, защита
от дублей) и src.ai.commercial_gate (pre-Deep решение без LLM). Сеть/production DB закрыты
safety_guards; фикстуры — фрагменты реальных документов, сохранённые inline.

Запуск: python -m unittest tests.test_commercial_gate -v
"""

import copy
import unittest

from src.ai import commercial_gate as gate
from src.ai import value_facts
from tests.test_deep_prompt import make_tender_context, make_triage_result

# Фрагмент реального объявления (тендер «Поставка шин», 5 лотов): 2 570 000 AMD в сумме.
TIRES_TEXT = (
    "ПРЕДМЕТОМ ЗАКУПКИ ЯВЛЯЕТСЯ ПРИОБРЕТЕНИЕ ПОСТАВОК ШИН, КОТОРЫЕ СГРУППИРОВАНЫ В 5 ЛОТ\n"
    "Лотов\tНаименование лота\n"
    "Номера\tЦена закупки\t\n"
    "1\t980 000\tШина для мусоровоза МАЗ-КО 9.00 R20\n"
    "2\t180 000\tШина 12.5/80-18 для экскаватора CASE 570 SV\n"
    "3\t280 000\tШина задняя для автобуса ПАЗ 245/70 R19,5\n"
    "4\t282 000\tШина 385/65 R 22.5 для грузового автомобиля SHACMAN F2000\n"
    "5\t848 000\tШина 315/80 R 22.5 для грузовика SHACMAN F2000\n"
    "Технические характеристики товара составляют неотъемлемую часть договора.\n"
    "Если предлагаемые цены представлены в двух валютах, они сопоставляются с драмом Республики Армения.\n"
)

# Фрагмент реального XLSX технической спецификации (армянский шаблон): колонка «Քանակը» — 8-я.
XLSX_TEXT = (
    "[Sheet: Лист1]\n"
    "\tԳնման առարկայի\t\tԳնման ձևը\tՉափի միավորը\tՄիավորի գինը\tԸնդամենը Գումարը\tՔանակը\tՏեխնիկական բնութագիր\n"
    "\tմիջանցիկ ծածկագիրը` ըստ ԳՄԱ\tանվանումը\n"
    "1\t38311100\tԲժշկական կշեռք\tԷԱՃ\tհատ\t40000\t160000\t4\tՄեխանիկական\n"
    "2\t31521570\tԼապտերիկ\tԷԱՃ\tհատ\t4000\t4000\t1\tԳրչաձև\n"
)


def doc(download_id, member_name, text, file_type="docx", status="success"):
    return {
        "download_id": download_id, "member_name": member_name, "file_type": file_type,
        "extraction_status": status, "text": text, "metrics": {},
    }


def context(documents=None, estimated_value=None, number_of_lots=None, successful=None):
    ctx = make_tender_context()
    ctx["enrichment"]["estimated_value_amd"] = estimated_value
    ctx["enrichment"]["number_of_lots"] = number_of_lots
    ctx["documents"] = documents if documents is not None else []
    ctx["document_coverage"]["successful_extractions"] = (
        len([d for d in ctx["documents"] if d["extraction_status"] == "success"]) if successful is None else successful
    )
    return ctx


class ParseAmountTests(unittest.TestCase):
    def test_accepts_plain_and_grouped_integers(self):
        self.assertEqual(value_facts.parse_amount("980000"), 980000)
        self.assertEqual(value_facts.parse_amount("980 000"), 980000)
        self.assertEqual(value_facts.parse_amount("1 980 000"), 1980000)
        self.assertEqual(value_facts.parse_amount("980 000"), 980000)
        self.assertEqual(value_facts.parse_amount(5000000), 5000000)

    def test_rejects_ambiguous_formats(self):
        for raw in ("", None, "abc", "1,5", "980 00", "12.50", "-5", "10 000 драмов"):
            with self.subTest(raw=raw):
                self.assertIsNone(value_facts.parse_amount(raw))


class ValueFactsTests(unittest.TestCase):
    def test_official_lot_values_are_summed_deterministically(self):
        facts = value_facts.extract_value_facts(context([doc(1, "hraver.docx", TIRES_TEXT)]))
        self.assertEqual(facts["estimated_value_amd"], 980000 + 180000 + 280000 + 282000 + 848000)
        self.assertEqual(facts["estimated_value_amd"], 2_570_000)
        self.assertEqual(facts["value_source"], value_facts.VALUE_SOURCE_LOT_SUM)
        self.assertEqual(facts["total_lots"], 5)
        self.assertEqual(facts["source_refs"], [{"download_id": 1, "member_name": "hraver.docx", "file_type": "docx"}])

    def test_official_tender_value_from_enrichment_has_priority_and_provenance(self):
        facts = value_facts.extract_value_facts(context([doc(1, "a.docx", "нет таблицы")], estimated_value="2 570 000"))
        self.assertEqual(facts["estimated_value_amd"], 2_570_000)
        self.assertEqual(facts["value_source"], value_facts.VALUE_SOURCE_OFFICIAL)
        self.assertEqual(facts["source_refs"], [{"field": "enrichment.estimated_value_amd"}])

    def test_official_value_equal_to_lot_sum_is_not_a_conflict(self):
        facts = value_facts.extract_value_facts(context([doc(1, "a.docx", TIRES_TEXT)], estimated_value=2570000))
        self.assertEqual(facts["value_source"], value_facts.VALUE_SOURCE_OFFICIAL)
        self.assertEqual(facts["notes"], [])

    def test_conflict_between_official_value_and_lot_sum_yields_unknown(self):
        facts = value_facts.extract_value_facts(context([doc(1, "a.docx", TIRES_TEXT)], estimated_value=1000))
        self.assertIsNone(facts["estimated_value_amd"])
        self.assertEqual(facts["value_source"], value_facts.VALUE_SOURCE_UNKNOWN)
        self.assertTrue(any(note.startswith("конфликт") for note in facts["notes"]))

    def test_arbitrary_numbers_in_text_are_not_summed(self):
        text = "Гарантия 24 месяца. Стоимость 1 000 000 драмов. Штраф 5000 драмов.\n"
        facts = value_facts.extract_value_facts(context([doc(1, "a.docx", text)]))
        self.assertIsNone(facts["estimated_value_amd"])
        self.assertEqual(facts["value_source"], value_facts.VALUE_SOURCE_UNKNOWN)

    def test_empty_price_makes_table_ambiguous(self):
        text = TIRES_TEXT.replace("3\t280 000\t", "3\t\t")
        facts = value_facts.extract_value_facts(context([doc(1, "a.docx", text)]))
        self.assertIsNone(facts["estimated_value_amd"])
        self.assertTrue(any("неоднозначна" in note for note in facts["notes"]))

    def test_duplicate_or_missing_lot_numbers_are_ambiguous(self):
        for old, new in (("2\t180 000", "1\t180 000"), ("4\t282 000", "6\t282 000")):
            with self.subTest(new=new):
                facts = value_facts.extract_value_facts(context([doc(1, "a.docx", TIRES_TEXT.replace(old, new))]))
                self.assertIsNone(facts["estimated_value_amd"])

    def test_foreign_currency_marker_or_missing_amd_marker_is_ambiguous(self):
        with_euro = TIRES_TEXT + "Оплата возможна в евро.\n"
        without_amd = TIRES_TEXT.replace("драмом Республики Армения", "местной валютой")
        for text in (with_euro, without_amd):
            facts = value_facts.extract_value_facts(context([doc(1, "a.docx", text)]))
            self.assertIsNone(facts["estimated_value_amd"])

    def test_two_language_versions_with_same_table_are_not_double_counted(self):
        hy = TIRES_TEXT + "Այլ լեզու\n"  # другой текст -> другой content hash, та же таблица лотов
        facts = value_facts.extract_value_facts(context([doc(1, "hraver.docx", TIRES_TEXT), doc(1, "hraver_hy.docx", hy)]))
        self.assertEqual(facts["estimated_value_amd"], 2_570_000)
        self.assertEqual(len(facts["source_refs"]), 2)

    def test_documents_with_different_lot_tables_are_ambiguous(self):
        other = TIRES_TEXT.replace("980 000", "990 000")
        facts = value_facts.extract_value_facts(context([doc(1, "a.docx", TIRES_TEXT), doc(2, "b.docx", other)]))
        self.assertIsNone(facts["estimated_value_amd"])

    def test_failed_documents_are_ignored(self):
        facts = value_facts.extract_value_facts(context([doc(1, "a.docx", TIRES_TEXT, status="failed")]))
        self.assertIsNone(facts["estimated_value_amd"])

    def test_context_is_not_mutated(self):
        ctx = context([doc(1, "a.docx", TIRES_TEXT)])
        before = copy.deepcopy(ctx)
        value_facts.extract_value_facts(ctx)
        self.assertEqual(ctx, before)


class DuplicateProtectionTests(unittest.TestCase):
    def test_thirty_identical_xlsx_do_not_multiply_quantity(self):
        one = value_facts.extract_value_facts(context([doc(1, "lot_1.xlsx", XLSX_TEXT, "xlsx")]))
        thirty_docs = [doc(1, f"lot_{index}.xlsx", XLSX_TEXT, "xlsx") for index in range(30)]
        thirty = value_facts.extract_value_facts(context(thirty_docs))
        self.assertEqual(one["total_quantity"], 5)
        self.assertEqual(thirty["total_quantity"], 5)
        self.assertEqual(len(thirty["quantity_refs"]), 1)

    def test_canonical_source_does_not_depend_on_document_order(self):
        docs = [doc(1, f"lot_{index}.xlsx", XLSX_TEXT, "xlsx") for index in range(5)]
        forward = value_facts.extract_value_facts(context(docs))
        backward = value_facts.extract_value_facts(context(list(reversed(docs))))
        self.assertEqual(forward["quantity_refs"], backward["quantity_refs"])

    def test_identical_lot_table_docs_are_counted_once(self):
        docs = [doc(index, f"copy_{index}.docx", TIRES_TEXT) for index in range(30)]
        facts = value_facts.extract_value_facts(context(docs))
        self.assertEqual(facts["estimated_value_amd"], 2_570_000)
        self.assertEqual(len(facts["source_refs"]), 1)

    def test_distinct_xlsx_contents_are_summed_once_each(self):
        second = XLSX_TEXT.replace("\t4\tՄեխ", "\t7\tՄեխ")
        facts = value_facts.extract_value_facts(context([
            doc(1, "a.xlsx", XLSX_TEXT, "xlsx"), doc(1, "b.xlsx", second, "xlsx"), doc(1, "c.xlsx", XLSX_TEXT, "xlsx"),
        ]))
        self.assertEqual(facts["total_quantity"], 5 + 8)

    def test_non_numeric_quantity_gives_none_not_a_guess(self):
        text = XLSX_TEXT.replace("\t4\tՄեխ", "\tչորս\tՄեխ")
        self.assertIsNone(value_facts.extract_value_facts(context([doc(1, "a.xlsx", text, "xlsx")]))["total_quantity"])


class CommercialGateTests(unittest.TestCase):
    MIN = 1_000_000

    def evaluate(self, ctx, minimum=None, **triage_overrides):
        return gate.evaluate(make_triage_result(**triage_overrides), ctx, minimum)

    def test_official_value_below_threshold_is_skipped(self):
        result = self.evaluate(context([doc(1, "a.docx", "x")], estimated_value=500_000), self.MIN)
        self.assertEqual(result["gate_decision"], gate.SKIP_LOW_VALUE)
        self.assertEqual(result["estimated_value_amd"], 500_000)
        self.assertEqual(result["facts_used"]["value_source"], value_facts.VALUE_SOURCE_OFFICIAL)
        self.assertEqual(result["confidence"], gate.CONFIDENCE_HIGH)

    def test_value_at_or_above_threshold_is_deep_candidate(self):
        for value in (1_000_000, 5_000_000):
            with self.subTest(value=value):
                result = self.evaluate(context([doc(1, "a.docx", "x")], estimated_value=value), self.MIN)
                self.assertEqual(result["gate_decision"], gate.DEEP_CANDIDATE)

    def test_lot_sum_is_used_with_medium_confidence(self):
        below = self.evaluate(context([doc(1, "a.docx", TIRES_TEXT)]), 3_000_000)
        above = self.evaluate(context([doc(1, "a.docx", TIRES_TEXT)]), 2_000_000)
        self.assertEqual(below["gate_decision"], gate.SKIP_LOW_VALUE)
        self.assertEqual(above["gate_decision"], gate.DEEP_CANDIDATE)
        self.assertEqual(above["confidence"], gate.CONFIDENCE_MEDIUM)
        self.assertEqual(above["total_lots"], 5)

    def test_threshold_absent_disables_monetary_skip(self):
        result = self.evaluate(context([doc(1, "a.docx", "x")], estimated_value=1))
        self.assertEqual(result["gate_decision"], gate.DEEP_CANDIDATE)
        self.assertIn("MIN_DEEP_VALUE_AMD не задан", result["gate_reason"])

    def test_relevance_status_is_not_modified(self):
        triage = make_triage_result()
        before = copy.deepcopy(triage)
        result = gate.evaluate(triage, context([doc(1, "a.docx", "x")], estimated_value=1), self.MIN)
        self.assertEqual(result["gate_decision"], gate.SKIP_LOW_VALUE)
        self.assertEqual(triage, before)
        self.assertEqual(triage["relevance_status"], "relevant")
        self.assertNotIn("relevance_status", result)

    def test_quantity_alone_never_causes_skip(self):
        # 1 единица, стоимость неизвестна, порог задан: не skip
        one_item = XLSX_TEXT.split("\n1\t")[0] + "\n1\t38311100\tАппарат\tԷԱՃ\tհատ\t9000000\t9000000\t1\tX\n"
        ctx = context([doc(1, "spec.xlsx", one_item, "xlsx")])
        result = self.evaluate(ctx, self.MIN)
        self.assertEqual(result["total_quantity"], 1)
        self.assertEqual(result["gate_decision"], gate.DEEP_CANDIDATE)

    def test_one_expensive_item_passes_if_official_value_supports_it(self):
        result = self.evaluate(context([doc(1, "a.docx", "x")], estimated_value=90_000_000), self.MIN)
        self.assertEqual(result["gate_decision"], gate.DEEP_CANDIDATE)

    def test_many_low_value_items_can_skip(self):
        many = XLSX_TEXT.replace("\t4\tՄեխ", "\t500\tՄեխ")
        ctx = context([doc(1, "a.xlsx", many, "xlsx")], estimated_value=250_000)
        result = self.evaluate(ctx, self.MIN)
        self.assertEqual(result["total_quantity"], 501)
        self.assertEqual(result["gate_decision"], gate.SKIP_LOW_VALUE)

    def test_unknown_value_is_explicit_and_not_skipped(self):
        result = self.evaluate(context([doc(1, "a.docx", "нет цен")]), self.MIN)
        self.assertIsNone(result["estimated_value_amd"])
        self.assertEqual(result["facts_used"]["value_source"], value_facts.VALUE_SOURCE_UNKNOWN)
        self.assertEqual(result["gate_decision"], gate.DEEP_CANDIDATE)
        self.assertEqual(result["confidence"], gate.CONFIDENCE_LOW)

    def test_unknown_value_with_documents_is_not_skipped_when_threshold_is_none(self):
        # MIN_DEEP_VALUE_AMD не задан: отсутствие стоимости само по себе не даёт skip
        result = self.evaluate(context([doc(1, "a.docx", "Поставка товаров, технические требования")]), None)
        self.assertIsNone(result["estimated_value_amd"])
        self.assertEqual(result["gate_decision"], gate.DEEP_CANDIDATE)
        self.assertEqual(result["confidence"], gate.CONFIDENCE_LOW)

    def test_unknown_value_with_quantity_only_is_not_skipped_when_threshold_is_none(self):
        # количество в спецификации без цены: ни skip, ни «too small»
        ctx = context([doc(1, "spec.xlsx", XLSX_TEXT, file_type="xlsx")])
        result = self.evaluate(ctx, None)
        self.assertEqual(result["gate_decision"], gate.DEEP_CANDIDATE)

    def test_unknown_value_without_documents_is_insufficient_when_threshold_is_none(self):
        result = self.evaluate(context([]), None)
        self.assertEqual(result["gate_decision"], gate.INSUFFICIENT_INFORMATION)

    def test_unknown_value_without_any_extracted_document_is_insufficient_information(self):
        result = self.evaluate(context([]), self.MIN)
        self.assertEqual(result["gate_decision"], gate.INSUFFICIENT_INFORMATION)

    def test_conflicting_facts_go_to_manual_review(self):
        ctx = context([doc(1, "a.docx", TIRES_TEXT)], estimated_value=1000)
        self.assertEqual(self.evaluate(ctx, self.MIN)["gate_decision"], gate.MANUAL_REVIEW)

    def test_category_is_a_fact_only_and_creates_no_value(self):
        for category in ("medical_equipment", "vehicles", "stationery"):
            with self.subTest(category=category):
                result = self.evaluate(context([doc(1, "a.docx", "x")]), self.MIN, category=category)
                self.assertEqual(result["category"], category)
                self.assertIsNone(result["estimated_value_amd"])
                self.assertEqual(result["gate_decision"], gate.DEEP_CANDIDATE)

    def test_skip_too_small_is_never_produced(self):
        contexts = [
            context([doc(1, "a.xlsx", XLSX_TEXT, "xlsx")]), context([]),
            context([doc(1, "a.docx", "x")], estimated_value=1),
        ]
        for ctx in contexts:
            self.assertNotEqual(self.evaluate(ctx, self.MIN)["gate_decision"], gate.SKIP_TOO_SMALL)

    def test_not_relevant_is_rejected(self):
        with self.assertRaises(ValueError):
            gate.evaluate(make_triage_result(relevance_status="not_relevant"), context([]), self.MIN)

    def test_maybe_is_evaluated_like_relevant(self):
        result = self.evaluate(context([doc(1, "a.docx", "x")], estimated_value=1), self.MIN, relevance_status="maybe")
        self.assertEqual(result["gate_decision"], gate.SKIP_LOW_VALUE)

    def test_invalid_threshold_rejected(self):
        for bad in (0, -1, True, 1.5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.evaluate(context([]), bad)

    def test_result_shape(self):
        result = self.evaluate(context([doc(1, "a.docx", TIRES_TEXT)]), self.MIN)
        self.assertEqual(
            set(result),
            {"gate_decision", "gate_reason", "facts_used", "estimated_value_amd", "total_quantity",
             "total_lots", "category", "confidence"},
        )
        self.assertIn(result["gate_decision"], gate.GATE_DECISIONS)


if __name__ == "__main__":
    unittest.main()
