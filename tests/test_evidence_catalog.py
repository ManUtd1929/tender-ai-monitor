"""
Тесты src.ai.evidence_catalog (procurement-deep-v4): детерминированные evidence ID, состав
каталога, строгая материализация evidence_ids -> evidence с текстом из каталога, dedup.
Сети и OpenAI нет.

Запуск из корня проекта:
    python -m unittest tests.test_evidence_catalog -v
"""

import copy
import unittest

from src.ai import deep_prompt, evidence_catalog, relevance_schema, tender_context as ctx
from tests.test_deep_dedup import TABLE, lot_documents, make_document, make_tender_context
from tests.test_deep_prompt import make_triage_result

TRIAGE = make_triage_result()


def build(documents, chunk_size=4000, enrichment=None, announcement=None):
    tender_context = make_tender_context(documents)
    if enrichment:
        tender_context["enrichment"].update(enrichment)
    if announcement:
        tender_context["announcement"].update(announcement)
    deep = ctx.build_deep_analysis_context(tender_context, chunk_size=chunk_size)
    return deep_prompt.build_deep_context(tender_context, deep, TRIAGE)


def by_text(context, text):
    (unit,) = [u for u in context["evidence_catalog"] if u["exact_text"] == text]
    return unit


def raw_output(**overrides) -> dict:
    """Минимальный model output v4 (evidence_ids везде)."""
    output = {
        "summary": "s", "opportunity_type": "procurement", "category": "medical_equipment",
        "why_interesting": "w", "contracting_authority": None, "procedure_code": None,
        "confidence": "high", "participation_barriers": [], "missing_information": [],
        "source_conflicts": [], "manual_review_required": False, "evidence_ids": [],
        "procurement": None, "logistics": None,
    }
    output.update(overrides)
    return output


class IdTests(unittest.TestCase):
    def test_ids_are_deterministic_across_repeated_builds(self):
        first = build(lot_documents(3), enrichment={"description": "line one\nline two"})
        second = build(lot_documents(3), enrichment={"description": "line one\nline two"})
        self.assertEqual(first["evidence_catalog"], second["evidence_catalog"])
        self.assertEqual(deep_prompt.build_user_input(first), deep_prompt.build_user_input(second))

    def test_ids_do_not_depend_on_document_order_for_duplicates(self):
        docs = lot_documents(5)
        forward = build(docs)["evidence_catalog"]
        backward = build(list(reversed(docs)))["evidence_catalog"]
        self.assertEqual(forward, backward)

    def test_ids_are_unique(self):
        catalog = build(lot_documents(4), enrichment={"description": "a\nb\nc"})["evidence_catalog"]
        ids = [unit["evidence_id"] for unit in catalog]
        self.assertEqual(len(ids), len(set(ids)))

    def test_announcement_id(self):
        context = build([], announcement={"title": "Medical equipment"})
        unit = by_text(context, "Medical equipment")
        self.assertEqual(unit["evidence_id"], "ev_ann_title_0")
        self.assertEqual(
            (unit["source_type"], unit["field"], unit["download_id"], unit["member_name"]),
            ("announcement", "title", None, None),
        )

    def test_enrichment_ids_for_text_dict_and_list_fields(self):
        context = build([], enrichment={
            "description": "first paragraph\nsecond paragraph",
            "dates": {"published_at_detail": "2026-01-10", "deadline_at_detail": "2026-02-01"},
            "cpv_codes": [{"code": "33100000", "name": "Medical equipments"}],
            "estimated_value_amd": 5000000,
        })
        self.assertEqual(by_text(context, "first paragraph")["evidence_id"], "ev_enr_description_0")
        self.assertEqual(by_text(context, "second paragraph")["evidence_id"], "ev_enr_description_1")
        deadline = by_text(context, "2026-02-01")
        self.assertEqual((deadline["field"], deadline["key"]), ("dates", "deadline_at_detail"))
        self.assertEqual(deadline["source_type"], "enrichment")
        cpv = by_text(context, '{"code":"33100000","name":"Medical equipments"}')
        self.assertEqual(cpv["evidence_id"], "ev_enr_cpv_codes_0")
        self.assertEqual(by_text(context, "5000000")["field"], "estimated_value_amd")

    def test_empty_fields_produce_no_units(self):
        context = build([], enrichment={"description": None, "dates": {}, "cpv_codes": []})
        fields = {unit["field"] for unit in context["evidence_catalog"]}
        self.assertNotIn("description", fields)
        self.assertNotIn("dates", fields)
        self.assertNotIn("cpv_codes", fields)

    def test_document_id_and_metadata(self):
        context = build([make_document("spec.docx", "Первый абзац\nВторой абзац", download_id=22, file_type="docx")])
        first = by_text(context, "Первый абзац")
        second = by_text(context, "Второй абзац")
        self.assertRegex(first["evidence_id"], r"^ev_doc_22_[0-9a-f]{6}_0000$")
        self.assertEqual(second["evidence_id"], first["evidence_id"][:-4] + "0001")
        self.assertEqual(
            (first["source_type"], first["field"], first["download_id"], first["member_name"]),
            ("document", "text", 22, "spec.docx"),
        )
        self.assertTrue(first["content_group_id"].startswith("content-"))

    def test_same_download_different_members_get_different_ids(self):
        context = build([
            make_document("a.xlsx", "same row A", download_id=5),
            make_document("b.xlsx", "same row B", download_id=5),
        ])
        self.assertNotEqual(by_text(context, "same row A")["evidence_id"], by_text(context, "same row B")["evidence_id"])

    def test_changed_document_text_changes_document_ids(self):
        old = build([make_document("a.xlsx", "row")])
        new = build([make_document("a.xlsx", "row changed")])
        self.assertNotEqual(old["evidence_catalog"][-1]["evidence_id"], new["evidence_catalog"][-1]["evidence_id"])


class UnitTests(unittest.TestCase):
    def test_xlsx_table_rows_are_units(self):
        context = build([make_document("lot.xlsx", TABLE)])
        rows = [u["exact_text"] for u in context["evidence_catalog"] if u["source_type"] == "document"]
        self.assertEqual(rows, TABLE.split("\n"))
        self.assertIn("Item 17\t17 pcs", rows)

    def test_document_units_are_exact_source_slices_and_not_normalized(self):
        text = "  leading spaces\r\nTab\tseparated\t\r\n\r\n   \nlast line without newline"
        context = build([make_document("a.docx", text, file_type="docx")])
        units = [u for u in context["evidence_catalog"] if u["source_type"] == "document"]
        for unit in units:
            self.assertEqual(text[unit["start"]:unit["end"]], unit["exact_text"])
        self.assertEqual(
            [u["exact_text"] for u in units],
            ["  leading spaces\r", "Tab\tseparated\t\r", "last line without newline"],
        )

    def test_long_line_is_split_into_bounded_contiguous_units_without_loss(self):
        line = " ".join(f"word{n}" for n in range(400))  # ~2.7k символов, одна строка
        context = build([make_document("a.docx", line, file_type="docx")], chunk_size=700)
        units = [u for u in context["evidence_catalog"] if u["source_type"] == "document"]
        self.assertGreater(len(units), 1)
        for unit in units:
            self.assertLessEqual(len(unit["exact_text"]), evidence_catalog.MAX_UNIT_CHARS)
        self.assertEqual("".join(u["exact_text"] for u in units), line)
        for previous, following in zip(units, units[1:]):
            self.assertEqual(previous["end"], following["start"])

    def test_unbroken_long_line_is_hard_split(self):
        line = "x" * 1300
        context = build([make_document("a.docx", line, file_type="docx")])
        units = [u for u in context["evidence_catalog"] if u["source_type"] == "document"]
        self.assertEqual([len(u["exact_text"]) for u in units], [500, 500, 300])

    def test_units_do_not_depend_on_chunk_size(self):
        small = build([make_document("a.docx", TABLE, file_type="docx")], chunk_size=50)["evidence_catalog"]
        large = build([make_document("a.docx", TABLE, file_type="docx")], chunk_size=4000)["evidence_catalog"]
        self.assertEqual(small, large)

    def test_every_unit_fits_materialized_evidence_validation(self):
        context = build([make_document("a.docx", "x" * 1300, file_type="docx")])
        for unit in context["evidence_catalog"]:
            evidence = evidence_catalog.materialize_evidence(
                [unit["evidence_id"]], evidence_catalog.index_catalog(context["evidence_catalog"]), "t",
            )[0]
            relevance_schema.validate_deep_evidence_item(evidence)

    def test_unicode_text_is_kept_exactly(self):
        text = "Ճ 10 հատ (nbsp)\nРусский текст — тире «ёлочки»"
        context = build([make_document("a.docx", text, file_type="docx")])
        texts = [u["exact_text"] for u in context["evidence_catalog"] if u["source_type"] == "document"]
        self.assertEqual(texts, text.split("\n"))

    def test_builder_does_not_mutate_inputs(self):
        tender_context = make_tender_context(lot_documents(3))
        deep = ctx.build_deep_analysis_context(tender_context)
        before = copy.deepcopy((tender_context, deep))
        deep_prompt.build_deep_context(tender_context, deep, TRIAGE)
        self.assertEqual((tender_context, deep), before)


class DedupTests(unittest.TestCase):
    def test_30_duplicate_xlsx_are_compact(self):
        single = build(lot_documents(1))
        many = build(lot_documents(30))
        self.assertEqual(single["evidence_catalog"], many["evidence_catalog"])
        user_input = deep_prompt.build_user_input(many)
        self.assertEqual(user_input.count("Item 17\t17 pcs"), 1)
        self.assertEqual(user_input.count("Item 1\t1 pcs"), 1)

    def test_catalog_uses_canonical_source_and_group_keeps_represented_sources(self):
        context = build(lot_documents(30))
        documents = [u for u in context["evidence_catalog"] if u["source_type"] == "document"]
        self.assertEqual({(u["download_id"], u["member_name"]) for u in documents}, {(5, "lot_1.xlsx")})
        (group,) = context["content_groups"]
        self.assertEqual({u["content_group_id"] for u in documents}, {group["content_group_id"]})
        self.assertEqual(len(group["represented_sources"]), 30)
        self.assertEqual(group["canonical_source"]["member_name"], "lot_1.xlsx")

    def test_distinct_contents_keep_separate_units(self):
        context = build([make_document("a.xlsx", "aaa"), make_document("b.xlsx", "bbb")])
        self.assertEqual(
            [(u["member_name"], u["exact_text"]) for u in context["evidence_catalog"] if u["source_type"] == "document"],
            [("a.xlsx", "aaa"), ("b.xlsx", "bbb")],
        )


class MaterializeTests(unittest.TestCase):
    def setUp(self):
        self.context = build(
            lot_documents(3),
            enrichment={"description": "Требуется сертификат ISO 13485"},
        )
        self.catalog = self.context["evidence_catalog"]
        self.row = by_text(self.context, "Item 3\t3 pcs")["evidence_id"]
        self.cert = by_text(self.context, "Требуется сертификат ISO 13485")["evidence_id"]
        self.title = "ev_ann_title_0"

    def materialize(self, **overrides):
        return evidence_catalog.materialize_deep_model_output(raw_output(**overrides), self.catalog)

    def test_returns_exact_original_text_from_catalog(self):
        result = self.materialize(evidence_ids=[self.row, self.cert, self.title])
        self.assertEqual(result["evidence"], [
            {"evidence_id": self.row, "source_type": "document", "field": "text", "download_id": 5,
             "member_name": "lot_1.xlsx", "text": "Item 3\t3 pcs"},
            {"evidence_id": self.cert, "source_type": "enrichment", "field": "description",
             "download_id": None, "member_name": None, "text": "Требуется сертификат ISO 13485"},
            {"evidence_id": self.title, "source_type": "announcement", "field": "title",
             "download_id": None, "member_name": None, "text": "Medical equipment"},
        ])
        self.assertNotIn("evidence_ids", result)

    def test_materialized_result_passes_relevance_validation(self):
        result = self.materialize(evidence_ids=[self.row], opportunity_type="unclear", category=None)
        validated = relevance_schema.validate_deep_analysis_result(result)
        self.assertEqual(validated["evidence"][0]["evidence_id"], self.row)
        self.assertEqual(validated["evidence"][0]["text"], "Item 3\t3 pcs")

    def test_top_level_barrier_item_and_lot_ids_are_materialized(self):
        barrier = {"type": "certification", "description": "d", "severity": "high", "evidence_ids": [self.cert]}
        item = {
            "item_name": "Monitor", "lot_number": "1", "quantity": "1", "unit": "pcs",
            "key_specifications": [], "brand_or_equivalent": None, "evidence_ids": [self.row, self.title],
        }
        lot = {"lot_number": "1", "description": None, "item_count": 1, "evidence_ids": [self.row]}
        procurement = {name: None for name in relevance_schema.PROCUREMENT_SCALAR_FIELDS}
        procurement.update({name: [] for name in relevance_schema.PROCUREMENT_LIST_FIELDS})
        procurement.update(brand_or_equivalent=None, items=[item], lots=[lot])
        result = self.materialize(
            evidence_ids=[self.title], participation_barriers=[barrier], procurement=procurement,
        )
        validated = relevance_schema.validate_deep_analysis_result(result)
        self.assertEqual([e["text"] for e in validated["evidence"]], ["Medical equipment"])
        self.assertEqual(validated["participation_barriers"][0]["evidence"][0]["text"], "Требуется сертификат ISO 13485")
        self.assertEqual([e["evidence_id"] for e in validated["procurement"]["items"][0]["evidence"]], [self.row, self.title])
        self.assertEqual(validated["procurement"]["lots"][0]["evidence"][0]["text"], "Item 3\t3 pcs")

    def test_inputs_are_not_mutated(self):
        raw = raw_output(evidence_ids=[self.row])
        before = copy.deepcopy((raw, self.catalog))
        evidence_catalog.materialize_deep_model_output(raw, self.catalog)
        self.assertEqual((raw, self.catalog), before)

    def test_unknown_id_rejected(self):
        with self.assertRaisesRegex(ValueError, "неизвестный evidence_id"):
            self.materialize(evidence_ids=["ev_doc_5_000000_9999"])

    def test_id_from_another_tender_rejected(self):
        other = build([make_document("other.xlsx", "Completely different tender row", download_id=99)])
        foreign = by_text(other, "Completely different tender row")["evidence_id"]
        self.assertNotIn(foreign, {u["evidence_id"] for u in self.catalog})
        with self.assertRaisesRegex(ValueError, "неизвестный evidence_id"):
            self.materialize(evidence_ids=[foreign])

    def test_no_fuzzy_matching_or_repair(self):
        for bad in (self.row.upper(), " " + self.row, self.row + " ", self.row[:-1], self.row + "0", ""):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.materialize(evidence_ids=[bad])

    def test_duplicate_id_in_one_list_rejected(self):
        with self.assertRaisesRegex(ValueError, "повторяется"):
            self.materialize(evidence_ids=[self.row, self.row])

    def test_non_string_and_non_list_ids_rejected(self):
        for bad in ([{"text": "Item 3"}], [1], [None], "ev_ann_title_0", None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.materialize(evidence_ids=bad)

    def test_model_written_evidence_text_is_rejected(self):
        evidence = [{"source_type": "announcement", "field": "title", "download_id": None,
                     "member_name": None, "text": "Medical equipment"}]
        with self.assertRaisesRegex(ValueError, "только evidence_ids"):
            evidence_catalog.materialize_deep_model_output({**raw_output(), "evidence": evidence}, self.catalog)
        item = {"item_name": "x", "lot_number": None, "quantity": None, "unit": None, "key_specifications": [],
                "brand_or_equivalent": None, "evidence_ids": [self.row], "evidence": evidence}
        procurement = {name: None for name in relevance_schema.PROCUREMENT_SCALAR_FIELDS}
        procurement.update({name: [] for name in relevance_schema.PROCUREMENT_LIST_FIELDS})
        procurement.update(brand_or_equivalent=None, items=[item])
        with self.assertRaisesRegex(ValueError, "только evidence_ids"):
            self.materialize(procurement=procurement)

    def test_missing_evidence_ids_rejected_at_every_level(self):
        raw = raw_output()
        del raw["evidence_ids"]
        with self.assertRaisesRegex(ValueError, "evidence_ids"):
            evidence_catalog.materialize_deep_model_output(raw, self.catalog)
        barrier = {"type": "certification", "description": "d", "severity": "high"}
        with self.assertRaisesRegex(ValueError, "evidence_ids"):
            self.materialize(participation_barriers=[barrier])

    def test_business_rule_empty_item_and_barrier_evidence_still_rejected(self):
        barrier = {"type": "certification", "description": "d", "severity": "high", "evidence_ids": []}
        with self.assertRaises(ValueError):
            relevance_schema.validate_deep_analysis_result(self.materialize(participation_barriers=[barrier]))

    def test_empty_catalog_rejects_any_id(self):
        with self.assertRaises(ValueError):
            evidence_catalog.materialize_deep_model_output(raw_output(evidence_ids=["ev_ann_title_0"]), [])

    def test_duplicate_catalog_ids_are_rejected(self):
        unit = self.catalog[0]
        with self.assertRaisesRegex(ValueError, "повторяющийся"):
            evidence_catalog.index_catalog([unit, dict(unit)])


class RenderTests(unittest.TestCase):
    def test_render_groups_units_under_source_headers(self):
        context = build([make_document("a.xlsx", "r1\nr2", download_id=8)], enrichment={"dates": {"deadline_at_detail": "2026-02-01"}})
        text = evidence_catalog.render_catalog(context["evidence_catalog"])
        self.assertIn("[SOURCE type=announcement field=title]", text)
        self.assertIn('[SOURCE type=document field=text download_id=8 member_name="a.xlsx"', text)
        self.assertIn("[ev_enr_dates_0 key=deadline_at_detail] 2026-02-01", text)
        self.assertEqual(text.count("[SOURCE type=document"), 1)

    def test_render_empty_catalog(self):
        self.assertIn("empty", evidence_catalog.render_catalog([]))


class TriageUnchangedTests(unittest.TestCase):
    def test_triage_evidence_item_still_rejects_evidence_id(self):
        item = {"source_type": "announcement", "field": "title", "download_id": None, "member_name": None,
                "text": "t", "evidence_id": "ev_ann_title_0"}
        with self.assertRaises(ValueError):
            relevance_schema.validate_evidence_item(item)

    def test_triage_result_still_rejects_evidence_id(self):
        triage = {
            "relevance_status": "relevant", "opportunity_type": "procurement", "category": "c", "confidence": "high",
            "reason": "r", "requires_deep_analysis": True,
            "evidence": [{"source_type": "announcement", "field": "title", "download_id": None,
                          "member_name": None, "text": "t", "evidence_id": "ev_ann_title_0"}],
        }
        with self.assertRaises(ValueError):
            relevance_schema.validate_triage_result(triage)

    def test_legacy_deep_evidence_without_evidence_id_still_valid(self):
        legacy = {"source_type": "announcement", "field": "title", "download_id": None, "member_name": None, "text": "t"}
        self.assertEqual(relevance_schema.validate_deep_evidence_item(legacy), legacy)

    def test_deep_evidence_id_must_be_nonblank_string(self):
        item = {"evidence_id": " ", "source_type": "announcement", "field": "title", "download_id": None,
                "member_name": None, "text": "t"}
        with self.assertRaises(ValueError):
            relevance_schema.validate_deep_evidence_item(item)

    def test_triage_schema_and_prompt_files_are_independent_of_catalog(self):
        from src.ai import triage_prompt
        self.assertNotIn("evidence_id", str(triage_prompt.build_evidence_item_schema()))
        self.assertNotIn("evidence_ids", triage_prompt.SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
