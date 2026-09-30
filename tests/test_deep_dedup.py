"""
Deterministic deep-context deduplication: одинаковый extracted text отправляется модели один раз.
Без БД и без OpenAI: tender_context собирается вручную.
"""

import copy
import unittest

from src.ai import deep_prompt, evidence_grounding, relevance_schema, tender_context as ctx

RESOURCE_URL = "https://example.test/resource/1"
TABLE = "[Sheet: Lot]\n" + "\n".join(f"Item {n}\t{n} pcs" for n in range(1, 31))


def make_document(member_name, text, download_id=5, file_type="xlsx", status="success") -> dict:
    return {
        "download_id": download_id, "member_name": member_name, "file_type": file_type,
        "extraction_status": status, "text": text, "metrics": {},
    }


def make_tender_context(documents) -> dict:
    return {
        "resource_url": RESOURCE_URL,
        "announcement": {"title": "Medical equipment", "section": "s", "resource_type": "t",
                         "published_at": None, "deadline_at": None, "resource_url": RESOURCE_URL},
        "enrichment": {"detail_titles": {}, "description": None, "procurement_type": None,
                       "procedure_type": None, "contracting_authority": None,
                       "estimated_value_amd": None, "dates": {}, "cpv_codes": [], "number_of_lots": 30},
        "document_coverage": {"successful_extractions": len(documents), "successful_xlsx": len(documents)},
        "documents": documents,
    }


TRIAGE = {"relevance_status": "relevant", "opportunity_type": "procurement", "category": "medical",
          "confidence": "high", "reason": "r"}


def build(documents, chunk_size=4000):
    tender_context = make_tender_context(documents)
    deep = ctx.build_deep_analysis_context(tender_context, chunk_size=chunk_size)
    return tender_context, deep, deep_prompt.build_deep_context(tender_context, deep, TRIAGE)


def evidence(text, download_id=5, member_name="lot_1.xlsx", field="text") -> dict:
    return {"source_type": "document", "field": field, "download_id": download_id,
            "member_name": member_name, "text": text}


def lot_documents(count=30):
    return [make_document(f"lot_{n}.xlsx", TABLE) for n in range(1, count + 1)]


class DedupTests(unittest.TestCase):
    def test_exact_duplicates_deduplicated(self):
        _, deep, _ = build(lot_documents())
        self.assertEqual("".join(c["text"] for c in deep["chunks"]), TABLE)
        self.assertEqual({c["member_name"] for c in deep["chunks"]}, {"lot_1.xlsx"})

    def test_different_filenames_do_not_prevent_dedupe(self):
        _, deep, _ = build([make_document("alpha.xlsx", "same"), make_document("zzz_other.xlsx", "same")])
        self.assertEqual(len(deep["content_groups"]), 1)
        self.assertEqual(len(deep["content_groups"][0]["represented_sources"]), 2)

    def test_similar_names_but_different_content_not_deduplicated(self):
        _, deep, _ = build([make_document("lot_1.xlsx", "aaa"), make_document("lot_2.xlsx", "aab")])
        self.assertEqual(len(deep["content_groups"]), 2)
        self.assertEqual(deep["document_content_stats"]["duplicate_documents"], 0)
        self.assertEqual([c["text"] for c in deep["chunks"]], ["aaa", "aab"])

    def test_canonical_source_is_independent_of_document_order(self):
        docs = lot_documents(5)
        _, forward, _ = build(docs)
        _, backward, _ = build(list(reversed(docs)))
        self.assertEqual(forward["content_groups"], backward["content_groups"])
        self.assertEqual(forward["chunks"], backward["chunks"])
        self.assertEqual(forward["content_groups"][0]["canonical_source"]["member_name"], "lot_1.xlsx")

    def test_represented_sources_preserved_and_sorted(self):
        _, deep, _ = build(list(reversed(lot_documents(30))))
        names = [s["member_name"] for s in deep["content_groups"][0]["represented_sources"]]
        self.assertEqual(len(names), 30)
        self.assertEqual(names, sorted(names))
        self.assertIn("lot_30.xlsx", names)

    def test_non_duplicate_document_order_preserved(self):
        _, deep, _ = build([make_document("b.xlsx", "bbb"), make_document("a.xlsx", "aaa")])
        self.assertEqual([c["member_name"] for c in deep["chunks"]], ["b.xlsx", "a.xlsx"])

    def test_duplicate_stats(self):
        docs = lot_documents(30) + [make_document("other.xlsx", "unique"),
                                    make_document("bad.xlsx", "", status="failed")]
        _, deep, _ = build(docs)
        self.assertEqual(deep["document_content_stats"], {
            "total_documents_with_text": 31, "unique_content_groups": 2, "duplicate_documents": 29,
        })

    def test_coverage_counts_original_documents(self):
        tender_context, _, deep_context = build(lot_documents(30))
        self.assertEqual(deep_context["document_coverage"]["successful_xlsx"], 30)
        self.assertEqual(len(deep_context["documents"]), 30)
        self.assertEqual(deep_context["document_content_stats"]["unique_content_groups"], 1)

    def test_context_does_not_mutate_input(self):
        tender_context = make_tender_context(lot_documents(3))
        snapshot = copy.deepcopy(tender_context)
        ctx.build_deep_analysis_context(tender_context)
        self.assertEqual(tender_context, snapshot)

    def test_medical_like_30_duplicates_text_once_and_deterministic(self):
        docs = lot_documents(30)
        _, _, first = build(docs)
        _, _, second = build(docs)
        user_input = deep_prompt.build_user_input(first)
        self.assertEqual(user_input, deep_prompt.build_user_input(second))
        self.assertEqual(user_input.count("Item 17\t17 pcs"), 1)
        self.assertEqual(first["evidence_catalog"], second["evidence_catalog"])
        documents = [u for u in first["evidence_catalog"] if u["source_type"] == "document"]
        self.assertEqual(len(documents), 31)  # заголовок листа + 30 строк, один раз (не 30 x 31)
        self.assertEqual(len(first["content_groups"][0]["represented_sources"]), 30)
        self.assertEqual(
            {unit["content_group_id"] for unit in documents},
            {first["content_groups"][0]["content_group_id"]},
        )

    def test_multichunk_content_dedup(self):
        _, deep, _ = build(lot_documents(4), chunk_size=50)
        self.assertGreater(len(deep["chunks"]), 1)
        self.assertEqual("".join(c["text"] for c in deep["chunks"]), TABLE)


class DedupEvidenceTests(unittest.TestCase):
    """Legacy (v3) grounding по chunks остаётся рабочим для чтения старых артефактов."""

    def setUp(self):
        _, deep, self.deep_context = build(lot_documents(30), chunk_size=50)
        self.deep_context["chunks"] = deep["chunks"]

    def validate(self, *items):
        evidence_grounding.validate_deep_evidence_grounding(list(items), self.deep_context)

    def test_evidence_against_canonical_source_passes(self):
        self.validate(evidence("Item 3\t3 pcs\nItem 4\t4 pcs"))

    def test_evidence_against_represented_source_grounded_via_canonical(self):
        self.validate(evidence("Item 3\t3 pcs", member_name="lot_17.xlsx"))

    def test_fabricated_text_fails_for_canonical_and_represented(self):
        for name in ("lot_1.xlsx", "lot_17.xlsx"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.validate(evidence("Item 999", member_name=name))

    def test_nonexistent_source_fails(self):
        with self.assertRaises(ValueError):
            self.validate(evidence("Item 3\t3 pcs", member_name="lot_31.xlsx"))
        with self.assertRaises(ValueError):
            self.validate(evidence("Item 3\t3 pcs", download_id=99))

    def test_source_missing_from_content_groups_fails(self):
        self.deep_context["documents"].append(
            {"download_id": 5, "member_name": "ghost.xlsx", "file_type": "xlsx", "extraction_status": "success"})
        with self.assertRaises(ValueError):
            self.validate(evidence("Item 3\t3 pcs", member_name="ghost.xlsx"))

    def test_triage_grounding_unchanged(self):
        triage_context = {
            "title": "t", "documents": [{"download_id": 5, "member_name": "lot_2.xlsx"}],
            "document_previews": [{"download_id": 5, "member_name": "lot_1.xlsx", "preview_text": "abc"}],
        }
        # у представленного, но не имеющего собственного preview документа triage по-прежнему отклоняет цитату
        with self.assertRaises(ValueError):
            evidence_grounding.validate_evidence_grounding(
                [evidence("abc", member_name="lot_2.xlsx", field="preview_text")], triage_context)
        triage_context["documents"].append({"download_id": 5, "member_name": "lot_1.xlsx"})
        evidence_grounding.validate_evidence_grounding(
            [evidence("abc", member_name="lot_1.xlsx", field="preview_text")], triage_context)


def make_item(**overrides) -> dict:
    item = {
        "item_name": "Gas detector", "lot_number": "1", "quantity": "1", "unit": "pcs",
        "key_specifications": [],
        "evidence": [{"source_type": "announcement", "field": "title", "download_id": None,
                      "member_name": None, "text": "t"}],
    }
    item.update(overrides)
    return item


class BrandEquivalentTests(unittest.TestCase):
    def test_item_specific_brand_accepted(self):
        brand = {"specified_brand": "FNIRSI GC-01", "equivalent_allowed": True}
        validated = relevance_schema.validate_procurement_item(make_item(brand_or_equivalent=brand))
        self.assertEqual(validated["brand_or_equivalent"], brand)

    def test_item_without_brand_key_is_backward_compatible(self):
        self.assertIsNone(relevance_schema.validate_procurement_item(make_item())["brand_or_equivalent"])

    def test_item_brand_null_accepted_and_bad_shape_rejected(self):
        self.assertIsNone(
            relevance_schema.validate_procurement_item(make_item(brand_or_equivalent=None))["brand_or_equivalent"])
        with self.assertRaises(ValueError):
            relevance_schema.validate_procurement_item(make_item(brand_or_equivalent="FNIRSI"))

    def test_tender_level_brand_still_accepted_and_item_brand_does_not_imply_it(self):
        from tests.test_relevance_schema import make_procurement_block
        brand = {"specified_brand": "FNIRSI GC-01", "equivalent_allowed": None}
        block = make_procurement_block(
            brand_or_equivalent=None, items=[make_item(brand_or_equivalent=brand)])
        validated = relevance_schema.validate_procurement_block(block)
        self.assertIsNone(validated["brand_or_equivalent"])
        self.assertEqual(validated["items"][0]["brand_or_equivalent"], brand)
        tender_wide = relevance_schema.validate_procurement_block(
            make_procurement_block(brand_or_equivalent=brand))
        self.assertEqual(tender_wide["brand_or_equivalent"], brand)
        self.assertIsNone(tender_wide["items"][0]["brand_or_equivalent"] if tender_wide["items"] else None)

    def test_output_schema_has_item_brand_required(self):
        schema = deep_prompt.build_deep_output_schema()
        item_schema = schema["properties"]["procurement"]["properties"]["items"]["items"]
        self.assertIn("brand_or_equivalent", item_schema["required"])
        self.assertEqual(
            set(item_schema["properties"]),
            {"evidence_ids" if name == "evidence" else name for name in relevance_schema.PROCUREMENT_ITEM_FIELDS},
        )


if __name__ == "__main__":
    unittest.main()
