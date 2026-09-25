"""
Production-модуль извлечения текста из тендерных документов.

Поддерживается только:
    - .docx (путь, file-like объект или bytes);
    - .zip, внутри которого извлекаются .docx.

НЕ поддерживается: .doc, .pdf, .xls/.xlsx, OCR, AI. В monitor.py и enrichment
pipeline модуль НЕ подключён; ничего не читается при импорте и без явного
вызова функций. HTTP не используется.

extract_docx() возвращает dict:

    file_type, text, char_count, paragraph_count, table_count

Текст собирается в порядке документа: непустые абзацы и строки таблиц идут
построчно, разделитель — "\\n". Строка таблицы: ячейки через TAB. Строки, в
которых все ячейки пустые, пропускаются. Таблицы не превращаются в структуру:
нужен читаемый текст для будущего AI. Вложенные таблицы, колонтитулы и
содержимое content controls (w:sdt) не извлекаются.
paragraph_count — число непустых абзацев верхнего уровня (вне таблиц),
table_count — число таблиц верхнего уровня.

Повреждённый DOCX в extract_docx() приводит к исключению (python-docx /
zipfile), оно не скрывается.

extract_docx_from_zip() читает ZIP в памяти, на диск ничего не распаковывает
и не использует имена member как пути файловой системы: member_name — только
метаданные. Лимиты (число member, размер member, суммарный размер) проверяются
по ZIP-заголовкам ДО чтения и ещё раз по фактически прочитанным bytes;
превышение -> ValueError. Повреждённый DOCX внутри ZIP не останавливает
обработку: он попадает в failures, остальные member обрабатываются.
"""

import io
import logging
import zipfile
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph

logger = logging.getLogger(__name__)

DOCX_EXTENSION = ".docx"

DEFAULT_MAX_MEMBERS = 50
DEFAULT_MAX_MEMBER_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_TOTAL_UNCOMPRESSED_BYTES = 100 * 1024 * 1024


def clean_text_piece(value):
    """strip() и нормализация CRLF/CR -> LF. Больше ничего не меняется."""
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def _open_arg(source):
    """Приводит source к тому, что принимают Document() и ZipFile()."""
    if isinstance(source, (bytes, bytearray)):
        return io.BytesIO(source)
    if isinstance(source, Path):
        return str(source)
    return source


def _cell_text(cell):
    parts = (clean_text_piece(p.text) for p in cell.paragraphs)
    return " ".join(p for p in parts if p)


def _table_lines(table):
    lines = []
    for row in table.rows:
        # Ячейки берём напрямую из XML строки: row.cells повторяет объединённую
        # ячейку для каждой колонки/строки, что дублировало бы текст.
        cells = [_cell_text(_Cell(tc, table)) for tc in row._tr.tc_lst]
        if any(cells):
            lines.append("\t".join(cells))
    return lines


def extract_docx(source):
    """Извлекает текст абзацев и таблиц из .docx (path / file-like / bytes)."""
    document = Document(_open_arg(source))

    lines = []
    paragraph_count = 0
    table_count = 0
    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            text = clean_text_piece(Paragraph(child, document).text)
            if text:
                lines.append(text)
                paragraph_count += 1
        elif child.tag == qn("w:tbl"):
            table_count += 1
            lines.extend(_table_lines(Table(child, document)))

    text = "\n".join(lines)
    return {
        "file_type": "docx",
        "text": text,
        "char_count": len(text),
        "paragraph_count": paragraph_count,
        "table_count": table_count,
    }


def _check_zip_limits(members, max_members, max_member_bytes, max_total_bytes):
    """Проверка лимитов по заголовкам ZIP до чтения содержимого."""
    if len(members) > max_members:
        raise ValueError(f"ZIP has too many members: {len(members)} > {max_members}")
    total = 0
    for info in members:
        if info.file_size > max_member_bytes:
            raise ValueError(
                f"ZIP member {info.filename!r} is too large: "
                f"{info.file_size} > {max_member_bytes} bytes"
            )
        total += info.file_size
    if total > max_total_bytes:
        raise ValueError(
            f"ZIP total uncompressed size is too large: {total} > {max_total_bytes} bytes"
        )


def _failure(member_name, exc):
    return {
        "member_name": member_name,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def extract_docx_from_zip(
    zip_path,
    max_members=DEFAULT_MAX_MEMBERS,
    max_member_bytes=DEFAULT_MAX_MEMBER_BYTES,
    max_total_uncompressed_bytes=DEFAULT_MAX_TOTAL_UNCOMPRESSED_BYTES,
):
    """
    Извлекает текст из всех .docx внутри ZIP без распаковки на диск.

    member_count — число member без директорий. skipped_members — имена
    member, которые не .docx. failures — повреждённые .docx и member, которые
    не удалось прочитать: {member_name, error_type, error_message}.
    """
    documents = []
    skipped_members = []
    failures = []

    with zipfile.ZipFile(_open_arg(zip_path)) as zf:
        members = [info for info in zf.infolist() if not info.is_dir()]
        _check_zip_limits(
            members, max_members, max_member_bytes, max_total_uncompressed_bytes
        )

        total_read = 0
        for info in members:
            name = info.filename
            if not name.lower().endswith(DOCX_EXTENSION):
                logger.info("ZIP member skipped (not .docx): %s", name)
                skipped_members.append(name)
                continue

            try:
                with zf.open(info) as member_file:
                    # +1 байт: заголовок file_size мог занижать реальный размер.
                    data = member_file.read(max_member_bytes + 1)
            except Exception as exc:
                logger.warning(
                    "ZIP member read failed: %s (%s: %s)", name, type(exc).__name__, exc
                )
                failures.append(_failure(name, exc))
                continue

            if len(data) > max_member_bytes:
                raise ValueError(
                    f"ZIP member {name!r} is too large: > {max_member_bytes} bytes"
                )
            total_read += len(data)
            if total_read > max_total_uncompressed_bytes:
                raise ValueError(
                    "ZIP total uncompressed size is too large: "
                    f"> {max_total_uncompressed_bytes} bytes"
                )

            try:
                result = extract_docx(data)
            except Exception as exc:
                logger.warning(
                    "DOCX extraction failed: %s (%s: %s)", name, type(exc).__name__, exc
                )
                failures.append(_failure(name, exc))
                continue

            logger.info("DOCX extracted: %s (%d chars)", name, result["char_count"])
            documents.append({"member_name": name, **result})

    return {
        "file_type": "zip",
        "member_count": len(members),
        "docx_count": len(documents),
        "documents": documents,
        "skipped_members": skipped_members,
        "failures": failures,
    }


# --------------------------------------------------------------------------

def main():
    print("Document extractor module. Use explicit extraction functions.")


if __name__ == "__main__":
    main()
