"""
TLS-обход только для gnumner.minfin.am.

Сервер gnumner.minfin.am отдаёт только leaf-сертификат (*.minfin.am) и не
присылает промежуточный "GoGetSSL RSA DV SSL CA 2". Windows докачивает его сам
(AIA), Linux/OpenSSL (Railway) — нет, и проверка падает ("unable to verify the
first certificate").

Решение: для хоста ровно gnumner.minfin.am используется отдельный SSLContext
с обычными доверенными корнями (certifi) + публичный промежуточный сертификат
из certs/GoGetSSLRSADVSSLCA2.pem. Это только дополнительный материал для
построения цепочки: проверка подписи, срока действия и имени хоста остаётся
полной. Все остальные хосты идут по обычному пути requests (verify=True).
Файл сертификата не найден/не читается -> исключение (fail closed).
"""

import logging
import ssl
from pathlib import Path
from urllib.parse import urlparse

import certifi
import requests
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)

GNUMNER_HOST = "gnumner.minfin.am"
INTERMEDIATE_CERT_PATH = Path(__file__).resolve().parents[1] / "certs" / "GoGetSSLRSADVSSLCA2.pem"


def _stdlib_ssl_context_class() -> type:
    """
    Штатный ssl.SSLContext, независимо от truststore.inject_into_ssl().

    После inject_into_ssl() имя ssl.SSLContext указывает на truststore.SSLContext:
    его load_verify_locations не влияет на проверку цепочки. Оригинал находим в MRO
    на момент вызова — порядок импорта и вызова configure_tls() не важен.

    Но свойства штатного класса (verify_mode, options, ...) обращаются к глобальному
    имени ssl.SSLContext, и при подмене уходят в бесконечную рекурсию (urllib3 их
    выставляет). Поэтому возвращаем подкласс, у которого эти свойства напрямую
    делегируют C-базе _ssl._SSLContext (так же делает сам truststore).
    """
    original = next(
        (c for c in ssl.SSLContext.__mro__ if c.__name__ == "SSLContext" and c.__module__ == "ssl"),
        None,
    )
    base = next((c for c in ssl.SSLContext.__mro__ if c.__module__ == "_ssl"), None)
    if original is None or base is None:
        raise RuntimeError("Не найден штатный ssl.SSLContext")
    cached = _SAFE_CLASSES.get(original)
    if cached is not None:
        return cached

    namespace = {}
    for name, attr in vars(original).items():
        descriptor = getattr(base, name, None)
        if isinstance(attr, property) and descriptor is not None and hasattr(descriptor, "__get__"):
            fset = descriptor.__set__ if hasattr(descriptor, "__set__") else None
            namespace[name] = property(descriptor.__get__, fset)
    safe = type("GnumnerSSLContext", (original,), namespace)
    _SAFE_CLASSES[original] = safe
    return safe


_SAFE_CLASSES: dict = {}


def is_gnumner_url(url: str) -> bool:
    return (urlparse(url).hostname or "").lower() == GNUMNER_HOST


def build_gnumner_ssl_context(cert_path: Path | None = None) -> ssl.SSLContext:
    """Контекст с полной проверкой: certifi + промежуточный CA. Нет файла -> ошибка."""
    path = Path(cert_path) if cert_path is not None else INTERMEDIATE_CERT_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Не найден сертификат промежуточного CA: {path}")

    context = _stdlib_ssl_context_class()(ssl.PROTOCOL_TLS_CLIENT)
    # PROTOCOL_TLS_CLIENT по умолчанию: CERT_REQUIRED + check_hostname=True.
    context.load_verify_locations(cafile=certifi.where())
    context.load_verify_locations(cafile=str(path))
    return context


class GnumnerAdapter(HTTPAdapter):
    """HTTPS-адаптер для gnumner.minfin.am. Контекст строится при первом запросе."""

    def __init__(self, *args, **kwargs):
        self._secured = False
        super().__init__(*args, **kwargs)

    def init_poolmanager(self, *args, **kwargs):
        if self._secured:
            kwargs["ssl_context"] = build_gnumner_ssl_context()
        super().init_poolmanager(*args, **kwargs)

    def send(self, request, **kwargs):
        if not self._secured:
            context = build_gnumner_ssl_context()  # fail closed до любого соединения
            self._secured = True
            self.init_poolmanager(self._pool_connections, self._pool_maxsize, block=self._pool_block)
            logger.info("TLS: для %s подключён промежуточный CA %s", GNUMNER_HOST, INTERMEDIATE_CERT_PATH.name)
            del context
        return super().send(request, **kwargs)


def new_session() -> requests.Session:
    """Session; адаптер монтируется только на https://gnumner.minfin.am/."""
    session = requests.Session()
    session.mount(f"https://{GNUMNER_HOST}/", GnumnerAdapter())
    return session


def get(url: str, **kwargs):
    """requests.get-замена: gnumner -> специальный контекст, остальные хосты -> requests.get как есть."""
    if not is_gnumner_url(url):
        return requests.get(url, **kwargs)
    session = new_session()
    response = session.get(url, **kwargs)
    # Для stream=True соединение освобождает response.close(); сессию не закрываем,
    # чтобы не оборвать чтение тела.
    if not kwargs.get("stream"):
        session.close()
    return response
