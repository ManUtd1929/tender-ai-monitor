"""
Production-модуль безопасного скачивания тендерных документов.

Только скачивает и сохраняет файлы. Содержимое (DOC/DOCX/PDF/ZIP) не
разбирается и не распаковывается. В monitor.py и enrichment pipeline модуль
НЕ подключён; ничего не скачивается при импорте и без явного вызова функций.

Каждая функция download_* возвращает dict:

    source_url, saved_path, filename, content_type, size_bytes, sha256

sha256 считается по фактически сохранённым bytes. Имя файла — только
человекочитаемая метка, идентификатором документа оно не является.

Раскладка на диске: <root>/<sha256(resource_url)[:16]>/<безопасное имя файла>.
Название тендера и имя файла в имени директории не участвуют, поэтому
одинаковые имена, слеши, Unicode и path traversal не могут повлиять на путь.

Запись атомарная: bytes пишутся в <имя>.part и только после полного успешного
скачивания переименовываются через os.replace. При любой ошибке .part удаляется.
Существующий файл не перезаписывается: тот же sha256 — возвращается он же,
другой sha256 — новая версия name__<хеш>.ext, старая версия сохраняется.

Ошибки сети (requests.RequestException, HTTP-статусы) наружу не скрываются,
слишком большой файл или HTML вместо файла дают ValueError: их обрабатывает
вызывающий код.

Механизм ARMEPS подтверждён диагностикой (armeps_final_download_probe.py):
страница ресурса -> prepareAnonymousDownload.do -> downloadContractDocument.do
в одной requests.Session.
"""

import hashlib
import logging
import os
import re
from email.errors import HeaderParseError
from email.header import decode_header, make_header
from email.message import Message
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import requests

from src import http_tls
from src.scraper.gnumner import HEADERS, TIMEOUT

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT_DIR = PROJECT_ROOT / "data" / "documents"

DEFAULT_MAX_BYTES = 50 * 1024 * 1024
CHUNK_SIZE = 64 * 1024

RESOURCE_HASH_LENGTH = 16
VERSION_HASH_LENGTHS = (8, 16, 64)  # длины хеша в name__<hash>.ext при коллизии

DEFAULT_FILENAME = "document"
MAX_FILENAME_LENGTH = 150
MAX_EXTENSION_LENGTH = 20
HTML_EXTENSIONS = (".html", ".htm", ".xhtml")

ARMEPS_PREPARE_URL = "https://armeps.am/epps/cft/prepareAnonymousDownload.do?documentId={}"
ARMEPS_DOWNLOAD_URL = "https://armeps.am/epps/cft/downloadContractDocument.do?documentId={}&resourceId=null"

INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*]')
CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


# --------------------------------------------------------------------------
# Директория и имена файлов
# --------------------------------------------------------------------------

def build_document_directory(resource_url: str, root_dir=None) -> Path:
    """
    <root>/<первые 16 hex SHA-256 от resource_url>. Директорию не создаёт.
    """
    root = Path(root_dir) if root_dir is not None else DEFAULT_ROOT_DIR
    digest = hashlib.sha256(resource_url.encode("utf-8")).hexdigest()
    return root / digest[:RESOURCE_HASH_LENGTH]


def sanitize_filename(filename: str) -> str:
    """
    Безопасное имя файла: без директорий, ../, символов, недопустимых в Windows,
    управляющих символов, хвостовых пробелов и точек. Пустой результат даёт
    "document". Расширение сохраняется; длинное имя укорачивается по основе.
    """
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    name = CONTROL_CHARS.sub("", name)
    name = INVALID_FILENAME_CHARS.sub("_", name)
    name = name.strip().rstrip(". ")

    if len(name) > MAX_FILENAME_LENGTH:
        stem, extension = os.path.splitext(name)
        if len(extension) > MAX_EXTENSION_LENGTH:
            stem, extension = name, ""
        name = stem[: MAX_FILENAME_LENGTH - len(extension)].rstrip(". ") + extension

    if not name:
        return DEFAULT_FILENAME

    # Зарезервированные имена устройств Windows (CON, NUL.txt, ...)
    if name.split(".")[0].rstrip().upper() in WINDOWS_RESERVED_NAMES:
        name = "_" + name
    return name


def _get_header(headers, name: str) -> str | None:
    """Значение заголовка без учёта регистра (работает и с обычным dict)."""
    value = headers.get(name)
    if value is not None:
        return value
    wanted = name.lower()
    for key, value in headers.items():
        if key.lower() == wanted:
            return value
    return None


def _decode_mime_words(text: str) -> str:
    """'=?UTF8?B?...?=' -> строка (стандартный email.header.decode_header)."""
    if "=?" not in text:
        return text
    try:
        return str(make_header(decode_header(text)))
    except (LookupError, UnicodeError, HeaderParseError):
        return text


def _fix_mojibake(text: str) -> str:
    """
    requests декодирует заголовки как latin-1; если сервер прислал сырые UTF-8
    bytes, восстанавливаем исходную строку. Иначе текст возвращается как есть.
    """
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def extract_filename_from_headers(headers) -> str | None:
    """
    Имя файла из Content-Disposition (filename=, filename*= и MIME-encoded
    =?charset?B/Q?...?=). Нет заголовка или имени — None. Имя НЕ санитизируется.
    """
    disposition = _get_header(headers, "Content-Disposition")
    if not disposition:
        return None

    message = Message()
    message["Content-Disposition"] = disposition
    filename = message.get_filename()
    if not filename:
        return None

    filename = _fix_mojibake(_decode_mime_words(filename)).strip()
    return filename or None


def _filename_from_url(url: str) -> str | None:
    basename = unquote(urlparse(url).path.rsplit("/", 1)[-1]).strip()
    return basename or None


# --------------------------------------------------------------------------
# HTTP и запись файла
# --------------------------------------------------------------------------

def _get(session: requests.Session, url: str, referer: str | None = None, stream: bool = True):
    """GET с timeout, User-Agent, verify=True и проверкой статуса."""
    headers = dict(HEADERS)
    if referer:
        headers["Referer"] = referer

    response = session.get(url, headers=headers, timeout=TIMEOUT, stream=stream, verify=True)
    try:
        response.raise_for_status()
    except Exception:
        response.close()
        raise
    return response


def _parse_content_length(headers) -> int | None:
    value = _get_header(headers, "Content-Length")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_quietly(path: Path) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning("Не удалось удалить временный файл %s: %s", path, e)


def _validate_response(response, filename: str, max_bytes: int) -> None:
    """Проверки до записи: не HTML вместо файла, Content-Length в пределах лимита."""
    content_type = (_get_header(response.headers, "Content-Type") or "").lower()
    if "text/html" in content_type and not filename.lower().endswith(HTML_EXTENSIONS):
        raise ValueError("Expected file download but received HTML")

    content_length = _parse_content_length(response.headers)
    if content_length is not None and content_length > max_bytes:
        raise ValueError(f"Content-Length {content_length} exceeds max_bytes {max_bytes}")


def _write_part_file(response, part_path: Path, max_bytes: int) -> tuple[str, int]:
    """Пишет тело ответа в part_path; возвращает (sha256, size). При ошибке .part удаляется."""
    digest = hashlib.sha256()
    size = 0
    try:
        with open(part_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                size += len(chunk)
                if size > max_bytes:
                    raise ValueError(f"Downloaded size exceeds max_bytes {max_bytes}")
                digest.update(chunk)
                file.write(chunk)
    except BaseException:
        _remove_quietly(part_path)
        raise
    return digest.hexdigest(), size


def _resolve_final_path(directory: Path, filename: str, sha256: str) -> tuple[Path, bool]:
    """
    (путь, файл уже существует с теми же bytes). Существующий файл с другими
    bytes не трогается: выбирается имя name__<хеш>.ext.
    """
    final_path = directory / filename
    if not final_path.exists():
        return final_path, False
    if _sha256_of_file(final_path) == sha256:
        return final_path, True

    stem, extension = os.path.splitext(filename)
    for length in VERSION_HASH_LENGTHS:
        candidate = directory / f"{stem}__{sha256[:length]}{extension}"
        if not candidate.exists():
            return candidate, False
        if _sha256_of_file(candidate) == sha256:
            return candidate, True
    raise ValueError(f"Cannot choose a unique file name for {filename}")


def _save_response(
    response, source_url: str, resource_url: str, fallback_filename: str,
    root_dir, max_bytes: int,
) -> dict:
    """Общая часть всех download_*: имя, проверки, запись .part, коллизии, os.replace."""
    try:
        filename = sanitize_filename(
            extract_filename_from_headers(response.headers) or fallback_filename
        )
        _validate_response(response, filename, max_bytes)

        directory = build_document_directory(resource_url, root_dir)
        directory.mkdir(parents=True, exist_ok=True)
        part_path = directory / (filename + ".part")

        sha256, size = _write_part_file(response, part_path, max_bytes)
        content_type = _get_header(response.headers, "Content-Type")
    finally:
        response.close()

    try:
        final_path, already_exists = _resolve_final_path(directory, filename, sha256)
        if already_exists:
            _remove_quietly(part_path)
            logger.info("Документ уже сохранён, дубль не создан: %s", final_path)
        else:
            os.replace(part_path, final_path)
    except BaseException:
        _remove_quietly(part_path)
        raise

    logger.info(
        "Документ скачан: %s -> %s (%s, %d байт)", source_url, final_path, final_path.name, size,
    )
    return {
        "source_url": source_url,
        "saved_path": str(final_path),
        "filename": final_path.name,
        "content_type": content_type,
        "size_bytes": size,
        "sha256": sha256,
    }


# --------------------------------------------------------------------------
# Публичные функции скачивания
# --------------------------------------------------------------------------

def download_direct_file(
    url: str,
    resource_url: str,
    root_dir=None,
    session=None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> dict:
    """
    Прямая ссылка на файл. Имя: Content-Disposition -> basename URL -> "document".
    Созданную внутри функции session закрывает сама функция.
    """
    logger.info("Скачивание документа: %s", url)

    own_session = session is None
    if own_session:
        session = http_tls.new_session()
    try:
        response = _get(session, url)
        return _save_response(
            response, url, resource_url, _filename_from_url(url) or DEFAULT_FILENAME,
            root_dir, max_bytes,
        )
    finally:
        if own_session:
            session.close()


def download_eauction_document(
    document_url: str,
    resource_url: str,
    root_dir=None,
    session=None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> dict:
    """
    ZIP приглашения eAuction (.../public_invitation/tender_XXXXX.zip): обычный
    прямой файл. Архив НЕ распаковывается.
    """
    return download_direct_file(
        document_url, resource_url, root_dir=root_dir, session=session, max_bytes=max_bytes,
    )


def download_armeps_document(
    resource_url: str,
    document_id: str,
    expected_filename: str | None = None,
    root_dir=None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> dict:
    """
    Документ ARMEPS в одной requests.Session:
      1. GET resource_url (cookies/сессия);
      2. GET prepareAnonymousDownload.do (Referer = resource_url);
      3. GET downloadContractDocument.do (Referer = prepare-страница) — файл.

    Тело prepare-страницы документом не считается. Имя: Content-Disposition
    финального ответа -> expected_filename -> document_<document_id>.
    """
    document_id = str(document_id)
    prepare_url = ARMEPS_PREPARE_URL.format(quote(document_id, safe=""))
    download_url = ARMEPS_DOWNLOAD_URL.format(quote(document_id, safe=""))
    logger.info("Скачивание документа ARMEPS %s: %s", document_id, resource_url)

    session = requests.Session()
    try:
        _get(session, resource_url, stream=False).close()
        _get(session, prepare_url, referer=resource_url, stream=False).close()
        response = _get(session, download_url, referer=prepare_url)
        return _save_response(
            response, download_url, resource_url,
            expected_filename or f"document_{document_id}", root_dir, max_bytes,
        )
    finally:
        session.close()


# --------------------------------------------------------------------------
# Диагностический запуск
# --------------------------------------------------------------------------

def main():
    print("Document downloader module. Use explicit download functions.")


if __name__ == "__main__":
    main()
