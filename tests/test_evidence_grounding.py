"""
Тесты src.ai.evidence_grounding: evidence должен быть дословным фрагментом источника из
triage_context (announcement / enrichment / document preview). Сети и OpenAI здесь нет.

Запуск из корня проекта:
    python -m unittest tests.test_evidence_grounding -v
"""

import unittest

from src.ai import evidence_grounding, relevance_schema, triage_prompt

CONTEXT = {
    "resource_url": "https://example.test/resource/1",
    "title": "Անվտանգության դռների ձեռքբերում",
    "section": "Հայտարարություններ",
    "resource_type": "armeps_documents_page",
    "published_at": "2025-01-10",
    "deadline_at": None,
    "detail_titles": {"default": "Անվտանգության դռներ", "ru": "Поставка дверей", "en": None},
    "description": "Требуется поставка\nстальных дверей\n  для  здания.",
    "procurement_type": "goods",
    "procedure_type": None,
    "contracting_authority": 'Министерство "Пример"',
    "estimated_value_amd": 1500000,
    "dates": {"published_at_detail": "2025-01-10", "deadline_at_detail": "2025-02-01"},
    "cpv_codes": [{"code": "44221200", "name": "Двери"}],
    "documents": [
        {"download_id": 7, "member_name": "spec.docx", "file_type": "docx", "extraction_status": "success"},
        {"download_id": 8, "member_name": "notes.docx", "file_type": "docx", "extraction_status": "success"},
    ],
    "document_previews": [
        {
            "download_id": 7, "member_name": "spec.docx", "file_type": "docx",
            "preview_text": "Дверь стальная, 2 шт.\nГарантия 24 месяца.\tДоставка в Ереван.",
            "truncated": False,
        },
    ],
}


def item(source_type="announcement", field="title", text=None, download_id=None, member_name=None) -> dict:
    return {
        "source_type": source_type, "field": field, "download_id": download_id,
        "member_name": member_name, "text": text if text is not None else CONTEXT.get(field) or "",
    }


def document_item(text, download_id=7, member_name="spec.docx", field="preview_text") -> dict:
    return item("document", field, text, download_id, member_name)


class Grounded(unittest.TestCase):
    def assertGrounded(self, *evidence):
        evidence_grounding.validate_evidence_grounding(list(evidence), CONTEXT)

    def assertRejected(self, *evidence, contains=None):
        with self.assertRaises(ValueError) as context:
            evidence_grounding.validate_evidence_grounding(list(evidence), CONTEXT)
        if contains:
            self.assertIn(contains, str(context.exception))
        return context.exception


class AnnouncementEvidenceTests(Grounded):
    def test_full_value_and_fragment_pass(self):
        self.assertGrounded(item(text=CONTEXT["title"]), item(text="դռների ձեռքբերում"))

    def test_every_announcement_field_can_be_quoted(self):
        for field in ("title", "section", "resource_type", "published_at"):
            with self.subTest(field=field):
                self.assertGrounded(item(field=field, text=CONTEXT[field]))

    def test_empty_context_field_is_rejected(self):
        self.assertRejected(item(field="deadline_at", text="None"), contains="пусто")

    def test_translated_title_is_rejected(self):
        self.assertRejected(item(text="Purchase of security doors"), contains="дословным")

    def test_explanation_appended_to_quote_is_rejected(self):
        self.assertRejected(item(text=CONTEXT["title"] + " (Приобретение защитных дверей)"))

    def test_ellipsis_with_translated_continuation_is_rejected(self):
        self.assertRejected(item(text="Անվտանգության դռների... (покупка дверей)"))

    def test_enrichment_field_under_announcement_source_is_rejected(self):
        self.assertRejected(item(field="description", text="поставка"), contains="не существует")

    def test_nonexistent_field_is_rejected(self):
        for field in ("no_such_field", None, "resource_url"):
            with self.subTest(field=field):
                self.assertRejected(item(field=field, text="x"), contains="не существует")

    def test_announcement_cannot_have_document_reference(self):
        self.assertRejected(item(download_id=7), contains="download_id/member_name")
        self.assertRejected(item(member_name="spec.docx"), contains="download_id/member_name")


class EnrichmentEvidenceTests(Grounded):
    def test_string_fields_pass(self):
        self.assertGrounded(
            item("enrichment", "procurement_type", "goods"),
            item("enrichment", "contracting_authority", 'Министерство "Пример"'),
            item("enrichment", "description", "стальных дверей"),
        )

    def test_number_field_passes_only_as_source_digits(self):
        self.assertGrounded(item("enrichment", "estimated_value_amd", "1500000"))
        self.assertRejected(item("enrichment", "estimated_value_amd", "1 500 000"))
        self.assertRejected(item("enrichment", "estimated_value_amd", "1.5 million"))

    def test_structured_field_leaf_value_passes(self):
        self.assertGrounded(
            item("enrichment", "detail_titles", "Поставка дверей"),
            item("enrichment", "dates", "2025-02-01"),
            item("enrichment", "cpv_codes", "Двери"),
            item("enrichment", "cpv_codes", "44221200"),
        )

    def test_structured_field_serialized_form_passes(self):
        self.assertGrounded(item("enrichment", "detail_titles", '"ru":"Поставка дверей"'))
        self.assertGrounded(item("enrichment", "cpv_codes", '{"code":"44221200","name":"Двери"}'))

    def test_structured_field_translation_is_rejected(self):
        self.assertRejected(item("enrichment", "detail_titles", "Supply of doors"))
        self.assertRejected(item("enrichment", "cpv_codes", "Doors"))

    def test_json_escaped_quote_as_seen_by_model_passes(self):
        self.assertGrounded(item("enrichment", "contracting_authority", 'Министерство \\"Пример\\"'))

    def test_field_not_in_source_is_rejected(self):
        self.assertRejected(item("enrichment", "title", "x"), contains="не существует")
        self.assertRejected(item("enrichment", "document_coverage", "x"), contains="не существует")

    def test_empty_field_is_rejected(self):
        self.assertRejected(item("enrichment", "procedure_type", "x"), contains="пусто")

    def test_enrichment_cannot_have_document_reference(self):
        self.assertRejected(item("enrichment", "description", "поставка", download_id=7))


class DocumentEvidenceTests(Grounded):
    def test_verbatim_fragment_passes(self):
        self.assertGrounded(document_item("Дверь стальная, 2 шт."), document_item("Гарантия 24 месяца."))

    def test_translated_or_paraphrased_fragment_is_rejected(self):
        self.assertRejected(document_item("Steel door, 2 pcs."), contains="дословным")
        self.assertRejected(document_item("Стальная дверь, две штуки"), contains="дословным")

    def test_wrong_download_id_is_rejected(self):
        self.assertRejected(document_item("Дверь стальная", download_id=999), contains="download_id")
        self.assertRejected(document_item("Дверь стальная", download_id=None), contains="download_id")

    def test_wrong_member_name_is_rejected(self):
        self.assertRejected(document_item("Дверь стальная", member_name="other.docx"), contains="member_name")
        self.assertRejected(document_item("Дверь стальная", member_name=None), contains="member_name")

    def test_member_of_another_download_is_rejected(self):
        self.assertRejected(document_item("Дверь стальная", download_id=8, member_name="spec.docx"))

    def test_document_without_preview_cannot_be_quoted(self):
        self.assertRejected(document_item("Дверь", download_id=8, member_name="notes.docx"), contains="нет preview")

    def test_wrong_field_is_rejected(self):
        self.assertRejected(document_item("Дверь стальная", field="text"), contains="preview_text")
        self.assertRejected(document_item("Дверь стальная", field=None), contains="preview_text")

    def test_quote_from_other_source_is_rejected(self):
        self.assertRejected(document_item(CONTEXT["title"]))

    def test_text_beyond_truncated_preview_is_rejected(self):
        context = {**CONTEXT, "document_previews": [{**CONTEXT["document_previews"][0], "preview_text": "Дверь стальная"}]}
        with self.assertRaises(ValueError):
            evidence_grounding.validate_evidence_grounding([document_item("Гарантия 24 месяца.")], context)


class NormalizationTests(Grounded):
    def test_normalize_collapses_whitespace_and_strips(self):
        self.assertEqual(evidence_grounding.normalize_evidence_text("  a \n\t b\r\n\r\nc  "), "a b c")

    def test_normalize_is_idempotent_and_keeps_case_and_text(self):
        once = evidence_grounding.normalize_evidence_text("Дверь\n Стальная")
        self.assertEqual(once, "Дверь Стальная")
        self.assertEqual(evidence_grounding.normalize_evidence_text(once), once)

    def test_newline_in_source_matches_space_in_quote(self):
        self.assertGrounded(item("enrichment", "description", "поставка стальных дверей для здания."))
        self.assertGrounded(document_item("Гарантия 24 месяца. Доставка в Ереван."))

    def test_newline_in_quote_matches_space_in_source(self):
        self.assertGrounded(item("enrichment", "description", "стальных\nдверей"))

    def test_quote_with_extra_whitespace_around_and_inside_passes(self):
        self.assertGrounded(document_item("  Дверь   стальная,\n2 шт.  "))

    def test_literal_escaped_newline_as_seen_in_input_json_passes(self):
        self.assertGrounded(document_item("2 шт.\\nГарантия"))

    def test_case_is_not_normalized(self):
        self.assertRejected(document_item("дверь стальная, 2 шт."))

    def test_removing_whitespace_is_not_normalization(self):
        self.assertRejected(document_item("Дверьстальная"))

    def test_only_whitespace_text_does_not_ground(self):
        self.assertRejected(item("enrichment", "description", "   "))


class ConstantsTests(unittest.TestCase):
    def test_announcement_fields_are_keys_of_triage_context_shape(self):
        for field in evidence_grounding.ANNOUNCEMENT_EVIDENCE_FIELDS + evidence_grounding.ENRICHMENT_EVIDENCE_FIELDS:
            self.assertIn(field, CONTEXT)

    def test_prompt_lists_the_same_fields(self):
        prompt = triage_prompt.SYSTEM_PROMPT
        for field in evidence_grounding.ANNOUNCEMENT_EVIDENCE_FIELDS + evidence_grounding.ENRICHMENT_EVIDENCE_FIELDS:
            self.assertIn(field, prompt)

    def test_evidence_max_length_still_enforced_by_schema(self):
        with self.assertRaises(ValueError):
            relevance_schema.validate_evidence_item(item(text="x" * (relevance_schema.EVIDENCE_TEXT_MAX_CHARS + 1)))


# --------------------------------------------------------------------------
# deep analysis (chunk-aware) grounding
# --------------------------------------------------------------------------

DEEP_CONTEXT = {
    "resource_url": "https://example.test/resource/1",
    "title": CONTEXT["title"],
    "section": CONTEXT["section"],
    "resource_type": CONTEXT["resource_type"],
    "published_at": CONTEXT["published_at"],
    "deadline_at": CONTEXT["deadline_at"],
    "detail_titles": CONTEXT["detail_titles"],
    "description": CONTEXT["description"],
    "procurement_type": CONTEXT["procurement_type"],
    "procedure_type": CONTEXT["procedure_type"],
    "contracting_authority": CONTEXT["contracting_authority"],
    "estimated_value_amd": CONTEXT["estimated_value_amd"],
    "dates": CONTEXT["dates"],
    "cpv_codes": CONTEXT["cpv_codes"],
    "documents": [
        {"download_id": 7, "member_name": "spec.docx", "file_type": "docx", "extraction_status": "success"},
        {"download_id": 8, "member_name": "notes.docx", "file_type": "docx", "extraction_status": "success"},
    ],
    "chunks": [
        {"chunk_id": "7:spec.docx:0", "download_id": 7, "member_name": "spec.docx", "file_type": "docx",
         "text": "Дверь стальная, 2 шт.\nГаран"},
        {"chunk_id": "7:spec.docx:1", "download_id": 7, "member_name": "spec.docx", "file_type": "docx",
         "text": "тия 24 месяца.\tДоставка в Ереван."},
    ],
}


def deep_document_item(text, download_id=7, member_name="spec.docx", field=None) -> dict:
    if field is None:
        field = evidence_grounding.DEEP_DOCUMENT_EVIDENCE_FIELD
    return item("document", field, text, download_id, member_name)


class DeepDocumentEvidenceTests(unittest.TestCase):
    def assertGrounded(self, *evidence):
        evidence_grounding.validate_deep_evidence_grounding(list(evidence), DEEP_CONTEXT)

    def assertRejected(self, *evidence, contains=None):
        with self.assertRaises(ValueError) as context:
            evidence_grounding.validate_deep_evidence_grounding(list(evidence), DEEP_CONTEXT)
        if contains:
            self.assertIn(contains, str(context.exception))
        return context.exception

    def test_quote_within_a_single_chunk_passes(self):
        self.assertGrounded(deep_document_item("Дверь стальная, 2 шт."))

    def test_quote_spanning_a_chunk_boundary_passes(self):
        # "Гаран" — конец chunk 0, "тия 24 месяца." — начало chunk 1: реконструированный
        # полный текст документа склеивает их без потери/дублирования.
        self.assertGrounded(deep_document_item("Гарантия 24 месяца."))

    def test_translated_or_paraphrased_fragment_is_rejected(self):
        self.assertRejected(deep_document_item("Steel door, 2 pcs."), contains="дословным")

    def test_field_must_be_deep_document_field_not_triage_preview_text(self):
        self.assertRejected(deep_document_item("Дверь стальная", field="preview_text"), contains="text")

    def test_wrong_download_id_is_rejected(self):
        self.assertRejected(deep_document_item("Дверь стальная", download_id=999), contains="download_id")

    def test_wrong_member_name_is_rejected(self):
        self.assertRejected(deep_document_item("Дверь стальная", member_name="other.docx"), contains="member_name")

    def test_document_without_chunks_cannot_be_quoted(self):
        self.assertRejected(deep_document_item("x", download_id=8, member_name="notes.docx"), contains="chunks")

    # --- shortest-sufficient-evidence regression (Luna tires case: model glued a continuation) ---

    SECURITY_CLAUSE = (
        "10.3 Размер обеспечения договора составляет 10 процентов от цены закупки. "
        "Если цена закупки товара меньше цены заключаемого договора, то размер обеспечения "
        "договора исчисляется в отношении цены договора."
    )
    SHORT_SECURITY_QUOTE = "Размер обеспечения договора составляет 10 процентов от цены закупки."

    def _security_context(self, text):
        context = dict(DEEP_CONTEXT)
        context["chunks"] = [
            {"chunk_id": "7:spec.docx:0", "download_id": 7, "member_name": "spec.docx",
             "file_type": "docx", "text": text},
        ]
        return context

    def assertSecurityRejected(self, quote, source):
        with self.assertRaises(ValueError) as context:
            evidence_grounding.validate_deep_evidence_grounding(
                [deep_document_item(quote)], self._security_context(source),
            )
        self.assertIn("дословным", str(context.exception))

    def test_short_exact_evidence_passes(self):
        source = self.SECURITY_CLAUSE
        evidence_grounding.validate_deep_evidence_grounding(
            [deep_document_item(self.SHORT_SECURITY_QUOTE)], self._security_context(source),
        )

    def test_model_added_continuation_fails(self):
        # Источник кончается на первом предложении; вторую половину модель дописала сама.
        source = "10.3 Размер обеспечения договора составляет 10 процентов от цены закупки. Иное не установлено."
        self.assertSecurityRejected(self.SECURITY_CLAUSE, source)

    def test_concatenation_of_non_contiguous_fragments_fails(self):
        source = (
            "10.3 Размер обеспечения договора составляет 10 процентов от цены закупки. "
            "10.4 Обеспечение возвращается в течение 10 дней. "
            "Если цена закупки товара меньше цены заключаемого договора, то размер обеспечения "
            "договора исчисляется в отношении цены договора."
        )
        self.assertSecurityRejected(self.SECURITY_CLAUSE, source)

    def test_paraphrase_of_security_clause_fails(self):
        self.assertSecurityRejected(
            "Обеспечение договора равно 10% от цены закупки.", self.SECURITY_CLAUSE,
        )

    def test_translation_of_security_clause_fails(self):
        self.assertSecurityRejected(
            "The contract security amount is 10 percent of the procurement price.", self.SECURITY_CLAUSE,
        )

    def test_announcement_and_enrichment_evidence_reused_unchanged(self):
        self.assertGrounded(
            item(text=DEEP_CONTEXT["title"]),
            item("enrichment", "description", "поставка стальных дверей для здания."),
        )


if __name__ == "__main__":
    unittest.main()
