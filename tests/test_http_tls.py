"""Тесты host-scoped TLS-обхода для gnumner.minfin.am (без сети, OpenAI и Telegram)."""

import re
import ssl
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import requests

from src import http_tls
from src.scraper import gnumner

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
CERT = http_tls.INTERMEDIATE_CERT_PATH


class HostSelectionTest(unittest.TestCase):
    def test_gnumner_is_special(self):
        self.assertTrue(http_tls.is_gnumner_url("https://gnumner.minfin.am/ru/page/x/"))
        self.assertTrue(http_tls.is_gnumner_url("https://GNUMNER.minfin.am/file.docx"))

    def test_other_hosts_are_not_special(self):
        for url in (
            "https://armeps.am/epps/x",
            "https://eauction.armeps.am/a",
            "https://api.openai.com/v1",
            "https://api.telegram.org/bot/x",
            "https://minfin.am/",
            "https://gnumner.minfin.am.evil.com/",
            "https://evil.com/gnumner.minfin.am/",
        ):
            with self.subTest(url=url):
                self.assertFalse(http_tls.is_gnumner_url(url))

    def test_session_mounts_adapter_only_for_gnumner(self):
        session = http_tls.new_session()
        self.assertIsInstance(session.get_adapter("https://gnumner.minfin.am/x"), http_tls.GnumnerAdapter)
        for url in ("https://armeps.am/", "https://eauction.armeps.am/", "https://gnumner.minfin.am.evil.com/"):
            with self.subTest(url=url):
                self.assertNotIsInstance(session.get_adapter(url), http_tls.GnumnerAdapter)

    def test_get_uses_session_for_gnumner(self):
        fake = mock.Mock()
        with mock.patch.object(http_tls, "new_session", return_value=fake) as new_session, \
                mock.patch.object(http_tls.requests, "get") as plain_get:
            http_tls.get("https://gnumner.minfin.am/a", timeout=5, verify=True)
        new_session.assert_called_once()
        fake.get.assert_called_once_with("https://gnumner.minfin.am/a", timeout=5, verify=True)
        plain_get.assert_not_called()

    def test_get_uses_plain_requests_for_other_hosts(self):
        for url in ("https://armeps.am/a", "https://eauction.armeps.am/b", "https://example.org/"):
            with self.subTest(url=url), \
                    mock.patch.object(http_tls, "new_session") as new_session, \
                    mock.patch.object(http_tls.requests, "get") as plain_get:
                http_tls.get(url, timeout=5, verify=True)
            new_session.assert_not_called()
            plain_get.assert_called_once_with(url, timeout=5, verify=True)

    def test_fetch_page_goes_through_helper(self):
        response = mock.Mock(text="<html></html>")
        with mock.patch.object(gnumner.http_tls, "get", return_value=response) as helper_get:
            gnumner.fetch_page(1)
        self.assertEqual(helper_get.call_args.args[0], gnumner.BASE_URL)
        self.assertIs(helper_get.call_args.kwargs["verify"], True)

    def test_direct_download_session_comes_from_helper(self):
        from src.parser import document_downloader as downloader
        with mock.patch.object(downloader.http_tls, "new_session") as new_session, \
                mock.patch.object(downloader, "_get", side_effect=RuntimeError("stop")):
            with self.assertRaises(RuntimeError):
                downloader.download_direct_file("https://gnumner.minfin.am/f.docx", "https://gnumner.minfin.am/")
        new_session.assert_called_once()


class ContextTest(unittest.TestCase):
    def test_context_verifies_cert_and_hostname(self):
        context = http_tls.build_gnumner_ssl_context()
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)
        self.assertNotIn("truststore", type(context).__module__)

    def test_context_trusts_pinned_intermediate(self):
        context = http_tls.build_gnumner_ssl_context()
        subjects = [dict(x[0] for x in c["subject"]) for c in context.get_ca_certs()]
        self.assertTrue(any(s.get("commonName") == "GoGetSSL RSA DV SSL CA 2" for s in subjects))

    def test_missing_cert_fails_closed(self):
        missing = Path(tempfile.gettempdir()) / "definitely-missing-ca.pem"
        with self.assertRaises(FileNotFoundError):
            http_tls.build_gnumner_ssl_context(missing)

    def test_adapter_send_fails_closed_without_cert_and_makes_no_request(self):
        adapter = http_tls.GnumnerAdapter()
        request = requests.Request("GET", "https://gnumner.minfin.am/").prepare()
        with mock.patch.object(http_tls, "INTERMEDIATE_CERT_PATH", Path("missing.pem")), \
                mock.patch.object(http_tls, "build_gnumner_ssl_context", side_effect=FileNotFoundError("no")), \
                mock.patch.object(requests.adapters.HTTPAdapter, "send") as base_send:
            with self.assertRaises(FileNotFoundError):
                adapter.send(request)
        base_send.assert_not_called()

    def test_adapter_passes_ssl_context_to_pool_after_first_send(self):
        adapter = http_tls.GnumnerAdapter()
        request = requests.Request("GET", "https://gnumner.minfin.am/").prepare()
        with mock.patch.object(requests.adapters.HTTPAdapter, "send", return_value="ok"):
            adapter.send(request)
        self.assertIn("ssl_context", adapter.poolmanager.connection_pool_kw)


class ImportOrderTest(unittest.TestCase):
    """truststore подменяет ssl.SSLContext; результат не должен зависеть от порядка."""

    SCRIPT = """
import ssl, sys
order = sys.argv[1]
import truststore
if order == "tls_first":
    from src import http_tls
    truststore.inject_into_ssl()
else:
    truststore.inject_into_ssl()
    from src import http_tls
assert ssl.SSLContext.__module__ == "truststore._api", ssl.SSLContext
ctx = http_tls.build_gnumner_ssl_context()
assert not any("truststore" in c.__module__ for c in type(ctx).__mro__), type(ctx)
ctx.verify_mode = ssl.CERT_REQUIRED; ctx.check_hostname = True
assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname
names = [dict(x[0] for x in c["subject"]).get("commonName") for c in ctx.get_ca_certs()]
assert "GoGetSSL RSA DV SSL CA 2" in names, names
print("ok")
"""

    def _run(self, order):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, "-c", self.SCRIPT, order], cwd=root,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)

    def test_http_tls_imported_before_truststore_injection(self):
        self._run("tls_first")

    def test_http_tls_imported_after_truststore_injection(self):
        self._run("inject_first")

    def test_helper_works_when_truststore_configured_in_process(self):
        import truststore
        truststore.inject_into_ssl()
        try:
            context = http_tls.build_gnumner_ssl_context()
            self.assertFalse(any("truststore" in c.__module__ for c in type(context).__mro__))
            self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        finally:
            truststore.extract_from_ssl()

    def test_gnumner_session_gets_special_context_and_others_do_not(self):
        adapter = http_tls.GnumnerAdapter()
        request = requests.Request("GET", "https://gnumner.minfin.am/").prepare()
        with mock.patch.object(requests.adapters.HTTPAdapter, "send", return_value="ok"):
            adapter.send(request)
        context = adapter.poolmanager.connection_pool_kw["ssl_context"]
        self.assertFalse(any("truststore" in c.__module__ for c in type(context).__mro__))
        other = requests.Session().get_adapter("https://armeps.am/")
        self.assertNotIn("ssl_context", other.poolmanager.connection_pool_kw)


class CertificateFileTest(unittest.TestCase):
    def test_pem_is_single_public_certificate(self):
        text = CERT.read_text(encoding="ascii")
        self.assertEqual(text.count("BEGIN CERTIFICATE"), 1)
        self.assertNotIn("PRIVATE KEY", text)


class NoVerifyFalseTest(unittest.TestCase):
    def test_helper_has_no_disabled_verification(self):
        code = (SRC_DIR / "http_tls.py").read_text(encoding="utf-8")
        code = re.sub(r'""".*?"""', "", code, flags=re.S)
        code = re.sub(r"#.*", "", code)
        self.assertNotIn("verify=False", code.replace(" ", ""))
        self.assertNotIn("CERT_NONE", code)
        self.assertNotIn("check_hostname = False", code)
        self.assertNotIn("_create_unverified_context", code)
        self.assertNotIn("REQUESTS_CA_BUNDLE", code)
        self.assertNotIn("SSL_CERT_FILE", code)


class NoVerifyFalseInSrcTest(unittest.TestCase):
    def test_no_verify_false_anywhere_in_src(self):
        offenders = []
        for path in SRC_DIR.rglob("*.py"):
            if re.search(r"verify\s*=\s*False", path.read_text(encoding="utf-8")):
                offenders.append(str(path.relative_to(SRC_DIR)))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
