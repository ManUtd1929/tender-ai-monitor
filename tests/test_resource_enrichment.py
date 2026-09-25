"""
Тесты resource_enrichment без сети: fixture-строки — сокращённые фрагменты
реального HTML eAuction и ARMEPS, сетевые вызовы замоканы.

Запуск из корня проекта:
    python -m unittest tests.test_resource_enrichment -v
"""

import unittest
from unittest import mock

import requests

from src.scraper import resource_enrichment as enrichment

EAUCTION_HTML = """
<div class="w_40 de_g">
    <div class="de_t">Ծածկագիր</div>
    <div class="de_v">ԵՔ-ԷԱՃԾՁԲ-26/148</div>
</div>
<div class="w_40 de_g">
    <div class="de_t">Վերնագիր</div>
    <div class="de_v">արխիվացման</div>
</div>
<div class="w_40 fe_g">
    <div class="fe_t">Հրապարակման ժամանակ</div>
    <div class="fe_v">
        2026-09-25 16:53:27
    </div>
</div>
<div class="w_40 fe_g">
    <div class="fe_t">Հայտերի ընդունման վերջնաժամկետ</div>
    <div class="fe_v">
        2026-10-06 09:00:00
    </div>
</div>
<div class="w_40 de_g">
    <div class="de_t">Կարգավիճակ</div>
    <div class="de_v">
        Հրապարակված        </div>
</div>
<div class="w_40 fe_g">
    <div class="fe_t">Հրավեր</div>
    <div class="fe_v">
        <a  href="https://eauction.armeps.am/application/documents/public_invitation/tender_47686.zip" download>Բեռնել</a>        </div>
</div>
"""

ARMEPS_HTML = """
<dl class="Grid">
<dt>Name of Contracting Authority:</dt>
<dd>
    <a href="/epps/prepareViewCAOrganisation.do?id=31018">
        Արաքսի համայնքապետարան
    </a>
</dd>
<dt>Title:</dt>
<dd>ՀՀ Արմավիրի մարզի Արաքս համայնք</dd>
<dt>Title (EN):</dt>
<dd>Procurement of goods for the Araks community</dd>
<dt>Title (RU):</dt>
<dd>Закупка товаров для нужд общины Аракс</dd>
<dt>CfT Id:</dt>
<dd>
    ԱՄԱՀ-ԱՊ-ԳՀԱՊՁԲ-26/107
</dd>
<dt>Evaluation Mechanism :</dt>
<dd> Lowest Price </dd>
<dt>Procurement Type:</dt>
<dd>Supplies</dd>
<dt>Procedure:</dt>
<dd>Request for Quotations</dd>
<dt>CPV Codes:</dt>
<dd>
    44131300-թուջե կափարիչներ<br/>
    44423690-դիտահոր երկաթբետոնե<br/>
</dd>
<dt>Estimated value (AMD):</dt>
<dd>
</dd>
<dt>Time-limit for receipt of tenders or requests to participate:</dt>
<dd>
    05/10/2026 14:30
</dd>
<dt>Number Of Lots:</dt>
<dd>2 </dd>
<dt>Date of Publication/Invitation :</dt>
<dd>
    25/09/2026 18:49
</dd>
</dl>
"""

ARMEPS_DOCUMENTS_HTML = """
<table id="T02">
<thead>
<tr>
    <th width=100><a href="javascript:sortBy('addendaId','sort')"><div class="THSort"></div></a>Addendum ID</th>
    <th><a href="javascript:sortBy('title','sort')"><div class="THSort"></div></a>Title</th>
    <th><a href="javascript:sortBy('fileName','sort')"><div class="THSort"></div></a>File</th>
    <th><a href="javascript:sortBy('description','sort')"><div class="THSort"></div></a>Description</th>
    <th abbr="Language"><a href="javascript:sortBy('language','sort')"><div class="THSort"></div></a>Lang.</th>
</tr>
</thead>
<tbody>
<tr>
    <td>N/A</td>
    <td align="left">Procurement of goods for the Araks community</td>
    <td align="left"><a href="#" onclick="downloadDocForAnonymous(12486850)">ԱՄԱՀ-ԱՊ-ԳՀԱՊՁԲ-26-117 անգլ..docx</a></td>
    <td align="left">Description of the procurement</td>
    <td align="left">EN</td>
</tr>
<tr>
    <td>N/A</td>
    <td align="left">Tender Structure XML - Cycle 1</td>
    <td align="left"><a href="#" onclick="downloadDocForAnonymous(12486845)">c4t_12486627_1.xml</a></td>
    <td align="left">N/A</td>
    <td align="left">HY</td>
</tr>
</tbody>
</table>
"""

ARMEPS_PAGE_HTML = ARMEPS_HTML + ARMEPS_DOCUMENTS_HTML

EAUCTION_URL = "https://eauction.armeps.am/hy/public/tender_details/tmid/d903e27c-3159-45ae-be43-6f33a82068ba"
ARMEPS_URL = "https://armeps.am/epps/cft/listContractDocuments.do?resourceId=12486627"


def make_announcement(resource_type: str, resource_url: str, **overrides) -> dict:
    announcement = {
        "title": "Тестовое объявление",
        "resource_type": resource_type,
        "resource_url": resource_url,
        "published_at": "2026-09-25 18:59:34",
        "deadline_at": "2026-10-02 11:10:00",
    }
    announcement.update(overrides)
    return announcement


def make_response(html: str) -> mock.Mock:
    response = mock.Mock()
    response.text = html
    return response


class NormalizeDetailDatetimeTest(unittest.TestCase):
    def test_iso_datetime_is_unchanged(self):
        self.assertEqual(
            enrichment.normalize_detail_datetime("2026-10-06 09:00:00"), "2026-10-06 09:00:00",
        )

    def test_day_first_without_seconds(self):
        self.assertEqual(
            enrichment.normalize_detail_datetime("05/10/2026 14:30"), "2026-10-05 14:30:00",
        )

    def test_day_first_with_seconds(self):
        self.assertEqual(
            enrichment.normalize_detail_datetime("25/09/2026 18:49:07"), "2026-09-25 18:49:07",
        )

    def test_surrounding_whitespace_is_ignored(self):
        self.assertEqual(
            enrichment.normalize_detail_datetime("\n  05/10/2026 14:30 \n"), "2026-10-05 14:30:00",
        )

    def test_none_and_empty_give_none(self):
        self.assertIsNone(enrichment.normalize_detail_datetime(None))
        self.assertIsNone(enrichment.normalize_detail_datetime(""))
        self.assertIsNone(enrichment.normalize_detail_datetime("   "))

    def test_unknown_format_gives_none_and_warning(self):
        with self.assertLogs(enrichment.logger, level="WARNING"):
            result = enrichment.normalize_detail_datetime("October 5, 2026 14:30")

        self.assertIsNone(result)

    def test_impossible_date_gives_none(self):
        with self.assertLogs(enrichment.logger, level="WARNING"):
            result = enrichment.normalize_detail_datetime("31/02/2026 10:00")

        self.assertIsNone(result)


class CompareAnnouncementWithDetailTest(unittest.TestCase):
    def test_same_published_at_is_true(self):
        result = enrichment.compare_announcement_with_detail(
            {"published_at": "2026-09-25 16:53:27"},
            {"published_at_detail": "2026-09-25 16:53:27"},
        )

        self.assertIs(result["published_at_match"], True)

    def test_different_published_at_is_false(self):
        result = enrichment.compare_announcement_with_detail(
            {"published_at": "2026-09-25 17:48:00"},
            {"published_at_detail": "2026-09-25 18:49:00"},
        )

        self.assertIs(result["published_at_match"], False)

    def test_same_deadline_after_normalization_is_true(self):
        detail = enrichment.parse_armeps_detail(ARMEPS_HTML)

        result = enrichment.compare_announcement_with_detail(
            {"deadline_at": "2026-10-05 14:30:00"}, detail,
        )

        self.assertIs(result["deadline_at_match"], True)

    def test_missing_detail_deadline_is_none(self):
        result = enrichment.compare_announcement_with_detail(
            {"deadline_at": "2026-10-05 14:30:00"}, {"deadline_at_detail": None},
        )

        self.assertIsNone(result["deadline_at_match"])

    def test_missing_list_deadline_is_none(self):
        result = enrichment.compare_announcement_with_detail(
            {"deadline_at": None}, {"deadline_at_detail": "2026-10-05 14:30:00"},
        )

        self.assertIsNone(result["deadline_at_match"])

    def test_missing_keys_are_none_not_mismatch(self):
        result = enrichment.compare_announcement_with_detail({}, {})

        self.assertEqual(result, {"published_at_match": None, "deadline_at_match": None})


class ParseEauctionDetailTest(unittest.TestCase):
    def setUp(self):
        self.result = enrichment.parse_eauction_detail(EAUCTION_HTML)

    def test_procedure_code(self):
        self.assertEqual(self.result["procedure_code"], "ԵՔ-ԷԱՃԾՁԲ-26/148")

    def test_deadline(self):
        self.assertEqual(self.result["deadline_at_detail"], "2026-10-06 09:00:00")

    def test_published_and_title(self):
        self.assertEqual(self.result["published_at_detail"], "2026-09-25 16:53:27")
        self.assertEqual(self.result["detail_title"], "արխիվացման")

    def test_status(self):
        self.assertEqual(self.result["procedure_status"], "Հրապարակված")

    def test_document_url(self):
        self.assertEqual(
            self.result["document_url"],
            "https://eauction.armeps.am/application/documents/public_invitation/tender_47686.zip",
        )

    def test_missing_fields_are_none(self):
        result = enrichment.parse_eauction_detail("<html><body><p>ничего</p></body></html>")

        self.assertEqual(set(result), set(self.result))
        self.assertTrue(all(value is None for value in result.values()))

    def test_only_present_fields_are_filled(self):
        html = '<div class="de_t">Ծածկագիր</div><div class="de_v">X-1</div>'

        result = enrichment.parse_eauction_detail(html)

        self.assertEqual(result["procedure_code"], "X-1")
        self.assertIsNone(result["deadline_at_detail"])
        self.assertIsNone(result["document_url"])


class ParseArmepsDetailTest(unittest.TestCase):
    def setUp(self):
        self.result = enrichment.parse_armeps_detail(ARMEPS_HTML)

    def test_contracting_authority(self):
        self.assertEqual(self.result["contracting_authority"], "Արաքսի համայնքապետարան")

    def test_cft_id_is_procedure_code(self):
        self.assertEqual(self.result["procedure_code"], "ԱՄԱՀ-ԱՊ-ԳՀԱՊՁԲ-26/107")

    def test_cpv_codes(self):
        self.assertEqual(
            self.result["cpv_codes"],
            [
                {"code": "44131300", "name": "թուջե կափարիչներ"},
                {"code": "44423690", "name": "դիտահոր երկաթբետոնե"},
            ],
        )

    def test_deadline_and_published(self):
        # DD/MM/YYYY HH:MM в HTML -> YYYY-MM-DD HH:MM:SS
        self.assertEqual(self.result["deadline_at_detail"], "2026-10-05 14:30:00")
        self.assertEqual(self.result["published_at_detail"], "2026-09-25 18:49:00")

    def test_titles_do_not_mix_languages(self):
        self.assertEqual(self.result["detail_title"], "ՀՀ Արմավիրի մարզի Արաքս համայնք")
        self.assertEqual(self.result["detail_title_en"], "Procurement of goods for the Araks community")
        self.assertEqual(self.result["detail_title_ru"], "Закупка товаров для нужд общины Аракс")

    def test_types_and_lots(self):
        self.assertEqual(self.result["procurement_type"], "Supplies")
        self.assertEqual(self.result["procedure_type"], "Request for Quotations")
        self.assertEqual(self.result["number_of_lots"], 2)

    def test_empty_and_missing_fields_are_none(self):
        self.assertIsNone(self.result["estimated_value_amd"])  # dd пустой
        self.assertIsNone(self.result["description"])  # dt отсутствует

    def test_empty_html_gives_all_none(self):
        result = enrichment.parse_armeps_detail("<html></html>")

        self.assertEqual(set(result), set(self.result))
        self.assertTrue(all(value is None for value in result.values()))


class ParseArmepsDocumentsTest(unittest.TestCase):
    def test_document_id_and_filename(self):
        documents = enrichment.parse_armeps_documents(ARMEPS_DOCUMENTS_HTML)

        self.assertEqual(documents[0]["document_id"], "12486850")
        self.assertEqual(documents[0]["filename"], "ԱՄԱՀ-ԱՊ-ԳՀԱՊՁԲ-26-117 անգլ..docx")

    def test_language(self):
        documents = enrichment.parse_armeps_documents(ARMEPS_DOCUMENTS_HTML)

        self.assertEqual(documents[0]["language"], "EN")
        self.assertEqual(documents[1]["language"], "HY")

    def test_several_documents_give_several_dicts(self):
        documents = enrichment.parse_armeps_documents(ARMEPS_DOCUMENTS_HTML)

        self.assertEqual([d["document_id"] for d in documents], ["12486850", "12486845"])

    def test_title_and_description(self):
        documents = enrichment.parse_armeps_documents(ARMEPS_DOCUMENTS_HTML)

        self.assertEqual(documents[0]["title"], "Procurement of goods for the Araks community")
        self.assertEqual(documents[0]["description"], "Description of the procurement")
        self.assertIsNone(documents[1]["description"])  # в HTML стоит N/A

    def test_no_documents(self):
        self.assertEqual(enrichment.parse_armeps_documents("<html></html>"), [])

    def test_rows_without_download_handler_are_ignored(self):
        html = "<table><tr><td>1</td><td><a href='#'>не документ</a></td></tr></table>"

        self.assertEqual(enrichment.parse_armeps_documents(html), [])


class ExtractResourceIdTest(unittest.TestCase):
    def test_resource_id_from_url(self):
        self.assertEqual(enrichment.extract_resource_id(ARMEPS_URL), "12486627")

    def test_no_resource_id(self):
        self.assertIsNone(enrichment.extract_resource_id("https://armeps.am/epps/cft/list.do"))


class EnrichAnnouncementTest(unittest.TestCase):
    def setUp(self):
        get_patcher = mock.patch.object(enrichment.requests, "get")
        session_patcher = mock.patch.object(enrichment.requests, "Session")
        self.get = get_patcher.start()
        self.session_class = session_patcher.start()
        self.addCleanup(get_patcher.stop)
        self.addCleanup(session_patcher.stop)

        self.session = self.session_class.return_value

    def assert_no_network(self):
        self.get.assert_not_called()
        self.session_class.assert_not_called()

    def test_direct_file_makes_no_network_request(self):
        url = "https://gnumner.minfin.am/website/images/original/36f98164.docx"
        announcement = make_announcement("direct_file", url)

        result = enrichment.enrich_announcement(announcement)

        self.assert_no_network()
        self.assertEqual(result["enrichment_status"], "not_required")
        self.assertEqual(result["direct_file_url"], url)

    def test_unknown_is_unsupported_without_network(self):
        announcement = make_announcement("unknown", "https://example.test/page")

        result = enrichment.enrich_announcement(announcement)

        self.assert_no_network()
        self.assertEqual(result["enrichment_status"], "unsupported")

    def test_original_announcement_is_not_modified(self):
        announcement = make_announcement("direct_file", "https://example.test/a.pdf")
        snapshot = dict(announcement)

        result = enrichment.enrich_announcement(announcement)

        self.assertEqual(announcement, snapshot)
        self.assertIsNot(result, announcement)
        self.assertEqual(result["title"], announcement["title"])

    def test_eauction_success(self):
        self.get.return_value = make_response(EAUCTION_HTML)
        announcement = make_announcement("eauction_tender_page", EAUCTION_URL)

        result = enrichment.enrich_announcement(announcement)

        args, kwargs = self.get.call_args
        self.assertEqual(args, (EAUCTION_URL,))
        self.assertIs(kwargs["verify"], True)
        self.assertIn("User-Agent", kwargs["headers"])
        self.assertIsNotNone(kwargs["timeout"])
        self.get.return_value.raise_for_status.assert_called_once()

        self.assertEqual(result["enrichment_status"], "success")
        self.assertEqual(result["procedure_code"], "ԵՔ-ԷԱՃԾՁԲ-26/148")
        # исходные даты list-страницы не перезаписываются
        self.assertEqual(result["deadline_at"], "2026-10-02 11:10:00")
        self.assertEqual(result["deadline_at_detail"], "2026-10-06 09:00:00")

    def test_eauction_live_like_dates_match(self):
        self.get.return_value = make_response(EAUCTION_HTML)
        announcement = make_announcement(
            "eauction_tender_page", EAUCTION_URL,
            published_at="2026-09-25 16:53:27", deadline_at="2026-10-06 09:00:00",
        )

        result = enrichment.enrich_announcement(announcement)

        self.assertEqual(
            result["consistency"], {"published_at_match": True, "deadline_at_match": True},
        )

    def test_armeps_live_like_published_differs_deadline_matches(self):
        self.session.get.return_value = make_response(ARMEPS_PAGE_HTML)
        announcement = make_announcement(
            "armeps_documents_page", ARMEPS_URL,
            published_at="2026-09-25 17:48:00", deadline_at="2026-10-05 14:30:00",
        )

        result = enrichment.enrich_announcement(announcement)

        self.assertEqual(result["published_at_detail"], "2026-09-25 18:49:00")
        self.assertEqual(result["deadline_at_detail"], "2026-10-05 14:30:00")
        self.assertEqual(
            result["consistency"], {"published_at_match": False, "deadline_at_match": True},
        )
        # исходные даты и код процедуры не «исправляются»
        self.assertEqual(result["published_at"], "2026-09-25 17:48:00")
        self.assertEqual(result["procedure_code"], "ԱՄԱՀ-ԱՊ-ԳՀԱՊՁԲ-26/107")

    def test_direct_file_and_unknown_have_no_consistency(self):
        for resource_type in ("direct_file", "unknown"):
            result = enrichment.enrich_announcement(
                make_announcement(resource_type, "https://example.test/x")
            )

            self.assertNotIn("consistency", result)

    def test_eauction_missing_fields_give_partial(self):
        self.get.return_value = make_response('<div class="de_t">Ծածկագիր</div><div class="de_v">X-1</div>')
        announcement = make_announcement("eauction_tender_page", EAUCTION_URL)

        result = enrichment.enrich_announcement(announcement)

        self.assertEqual(result["enrichment_status"], "partial")
        self.assertEqual(result["procedure_code"], "X-1")
        self.assertIsNone(result["document_url"])

    def test_eauction_http_error_is_not_hidden(self):
        self.get.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError("500")
        announcement = make_announcement("eauction_tender_page", EAUCTION_URL)

        with self.assertRaises(requests.exceptions.HTTPError):
            enrichment.enrich_announcement(announcement)

    def test_armeps_success(self):
        self.session.get.return_value = make_response(ARMEPS_PAGE_HTML)
        announcement = make_announcement("armeps_documents_page", ARMEPS_URL)

        result = enrichment.enrich_announcement(announcement)

        self.get.assert_not_called()  # запрос идёт через Session, а не requests.get
        args, kwargs = self.session.get.call_args
        self.assertEqual(args, (ARMEPS_URL,))
        self.assertIs(kwargs["verify"], True)
        self.session.close.assert_called_once()

        self.assertEqual(result["enrichment_status"], "success")
        self.assertEqual(result["resource_id"], "12486627")
        self.assertEqual(result["procedure_code"], "ԱՄԱՀ-ԱՊ-ԳՀԱՊՁԲ-26/107")
        self.assertEqual(len(result["documents"]), 2)

    def test_armeps_without_documents_gives_partial(self):
        self.session.get.return_value = make_response(ARMEPS_HTML)
        announcement = make_announcement("armeps_documents_page", ARMEPS_URL)

        result = enrichment.enrich_announcement(announcement)

        self.assertEqual(result["enrichment_status"], "partial")
        self.assertEqual(result["documents"], [])

    def test_armeps_http_error_is_not_hidden_and_session_is_closed(self):
        self.session.get.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError("503")
        announcement = make_announcement("armeps_documents_page", ARMEPS_URL)

        with self.assertRaises(requests.exceptions.HTTPError):
            enrichment.enrich_announcement(announcement)

        self.session.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
