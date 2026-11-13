"""Reproducible, page-by-page PDF ingestion for the project corpus."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from pypdf import PdfReader
from pypdf.errors import FileNotDecryptedError, PdfReadError


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_FIELDS = [
    "document_id",
    "title",
    "institution",
    "category",
    "url",
    "publication_date",
    "collection_date",
    "language",
    "filename",
    "source_path",
    "ingestion_status",
    "include_in_pipeline",
    "review_reason",
]


@dataclass(frozen=True)
class IngestionSummary:
    documents: int
    documents_failed: int
    pages_processed: int
    pages_without_text: int
    page_errors: int


def _slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = "".join(
        character for character in normalized
        if not unicodedata.combining(character)
    )
    return re.sub(r"[^a-z0-9]+", "-", ascii_value.lower()).strip("-")


def _document_id(pdf_path: Path, input_dir: Path) -> str:
    relative_path = pdf_path.relative_to(input_dir).with_suffix("")
    parts = [_slug(input_dir.name), *(_slug(part) for part in relative_path.parts)]
    return "-".join(part for part in parts if part) or "pdf-document"


def _source_path(pdf_path: Path, input_dir: Path) -> str:
    return (Path(input_dir.name) / pdf_path.relative_to(input_dir)).as_posix()


def _serialize_metadata(metadata: Any) -> dict[str, str]:
    if metadata is None:
        return {}
    return {
        str(key): str(value)
        for key, value in metadata.items()
        if value is not None
    }


def _record(
    *,
    document_id: str,
    filename: str,
    source_path: str,
    record_type: str,
    status: str,
    pdf_metadata: dict[str, str],
    page_number: int | None = None,
    text: str = "",
    error: Exception | None = None,
) -> dict[str, Any]:
    return {
        "record_type": record_type,
        "document_id": document_id,
        "filename": filename,
        "source_path": source_path,
        "page_number": page_number,
        "text": text,
        "character_count": len(text),
        "pdf_metadata": pdf_metadata,
        "status": status,
        "error": (
            {"type": type(error).__name__, "message": str(error)}
            if error is not None
            else None
        ),
    }


def _iter_pdf_records(
    pdf_path: Path, input_dir: Path
) -> Iterator[dict[str, Any]]:
    document_id = _document_id(pdf_path, input_dir)
    filename = pdf_path.name
    source_path = _source_path(pdf_path, input_dir)
    metadata: dict[str, str] = {}
    pages_found = False

    try:
        with pdf_path.open("rb") as pdf_file:
            reader = PdfReader(pdf_file, strict=False)
            metadata = _serialize_metadata(reader.metadata)
            for page_number, page in enumerate(reader.pages, start=1):
                pages_found = True
                try:
                    text = page.extract_text() or ""
                except (FileNotDecryptedError, OSError, PdfReadError, ValueError) as error:
                    LOGGER.error(
                        "Falha ao extrair a página %s de %s: %s",
                        page_number,
                        filename,
                        error,
                    )
                    yield _record(
                        document_id=document_id,
                        filename=filename,
                        source_path=source_path,
                        record_type="page_error",
                        status="error",
                        pdf_metadata=metadata,
                        page_number=page_number,
                        error=error,
                    )
                    return

                status = "processed"
                if not text.strip():
                    status = "empty_text"
                    LOGGER.warning(
                        "Página %s sem texto extraível em %s; OCR não foi executado.",
                        page_number,
                        filename,
                    )
                yield _record(
                    document_id=document_id,
                    filename=filename,
                    source_path=source_path,
                    record_type="page",
                    status=status,
                    pdf_metadata=metadata,
                    page_number=page_number,
                    text=text,
                )
    except (FileNotDecryptedError, OSError, PdfReadError, ValueError) as error:
        LOGGER.error("Falha ao ler %s: %s", filename, error)
        yield _record(
            document_id=document_id,
            filename=filename,
            source_path=source_path,
            record_type="document_error",
            status="error",
            pdf_metadata=metadata,
            error=error,
        )
        return

    if not pages_found:
        error = ValueError("O PDF não contém páginas.")
        LOGGER.error("Falha ao ler %s: %s", filename, error)
        yield _record(
            document_id=document_id,
            filename=filename,
            source_path=source_path,
            record_type="document_error",
            status="error",
            pdf_metadata=metadata,
            error=error,
        )


def _update_sources_csv(
    sources_path: Path, source_records: list[dict[str, str]]
) -> None:
    existing_rows: dict[str, dict[str, str]] = {}
    fieldnames = list(SOURCE_FIELDS)

    if sources_path.exists():
        with sources_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames:
                existing_fields = [
                    field for field in reader.fieldnames if field != "status"
                ]
                fieldnames = list(dict.fromkeys([*fieldnames, *existing_fields]))
            for row in reader:
                document_id = row.get("document_id", "")
                if document_id:
                    existing = {
                        key: value or "" for key, value in row.items() if key
                    }
                    legacy_status = existing.pop("status", "")
                    if not existing.get("ingestion_status"):
                        existing["ingestion_status"] = legacy_status
                    existing_rows[document_id] = existing

    for source_record in source_records:
        document_id = source_record["document_id"]
        existing_rows.setdefault(document_id, {}).update(source_record)

    sources_path.parent.mkdir(parents=True, exist_ok=True)
    with sources_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            existing_rows[document_id]
            for document_id in sorted(existing_rows)
        )


def ingest_pdfs(
    input_dir: Path, output_path: Path, sources_path: Path
) -> IngestionSummary:
    input_dir = input_dir.resolve()
    pdf_paths = sorted(
        (
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: path.relative_to(input_dir).as_posix().casefold(),
    )
    if not pdf_paths:
        raise FileNotFoundError(f"Nenhum arquivo PDF encontrado em {input_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    documents_failed = 0
    pages_processed = 0
    pages_without_text = 0
    page_errors = 0
    source_records: list[dict[str, str]] = []

    with output_path.open("w", encoding="utf-8", newline="\n") as jsonl_file:
        for pdf_path in pdf_paths:
            document_id = _document_id(pdf_path, input_dir)
            metadata: dict[str, str] = {}
            document_failed = False

            for record in _iter_pdf_records(pdf_path, input_dir):
                jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                metadata = record["pdf_metadata"]
                if record["record_type"] in {"page", "page_error"}:
                    pages_processed += 1
                if record["status"] == "empty_text":
                    pages_without_text += 1
                if record["record_type"] == "page_error":
                    page_errors += 1
                    document_failed = True
                elif record["record_type"] == "document_error":
                    document_failed = True

            if document_failed:
                documents_failed += 1
            source_records.append(
                {
                    "document_id": document_id,
                    "title": metadata.get("/Title", ""),
                    "filename": pdf_path.name,
                    "source_path": _source_path(pdf_path, input_dir),
                    "ingestion_status": (
                        "error" if document_failed else "processed"
                    ),
                }
            )

    _update_sources_csv(sources_path, source_records)
    return IngestionSummary(
        documents=len(pdf_paths),
        documents_failed=documents_failed,
        pages_processed=pages_processed,
        pages_without_text=pages_without_text,
        page_errors=page_errors,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extrai texto página a página de PDFs e salva registros JSONL."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "NASA",
        help="Pasta de entrada (padrão: data/raw/NASA).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "nasa_pages.jsonl",
        help="Arquivo JSONL de saída.",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=PROJECT_ROOT / "sources.csv",
        help="CSV de registro das fontes.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    summary = ingest_pdfs(args.input_dir, args.output, args.sources)
    print(f"Documentos encontrados: {summary.documents}")
    print(f"Documentos com erros: {summary.documents_failed}")
    print(f"Páginas processadas: {summary.pages_processed}")
    print(f"Páginas sem texto extraível: {summary.pages_without_text}")
    print(f"Falhas de extração de página: {summary.page_errors}")
    print(f"JSONL: {args.output}")
    print(f"Fontes: {args.sources}")
    return int(summary.documents_failed > 0)


if __name__ == "__main__":
    sys.exit(main())
