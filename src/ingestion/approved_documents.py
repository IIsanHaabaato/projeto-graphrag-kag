"""Load only reviewed, successfully ingested pages for downstream pipelines."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


REQUIRED_MANIFEST_FIELDS = {
    "document_id",
    "ingestion_status",
    "include_in_pipeline",
    "review_reason",
}
VALID_INGESTION_STATUSES = {"processed", "error"}
VALID_RECORD_TYPES = {"page", "page_error", "document_error"}


@dataclass(frozen=True)
class ApprovedPage:
    document_id: str
    filename: str
    source_path: str
    page_number: int
    text: str
    character_count: int
    pdf_metadata: dict[str, str]


def _read_manifest(manifest_path: Path) -> dict[str, bool]:
    eligibility: dict[str, bool] = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("O manifesto CSV não possui cabeçalho.")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("O cabeçalho do manifesto possui colunas duplicadas.")
        missing_fields = REQUIRED_MANIFEST_FIELDS - set(reader.fieldnames)
        if missing_fields:
            raise ValueError(
                "Colunas obrigatórias ausentes no manifesto: "
                + ", ".join(sorted(missing_fields))
            )

        for line_number, row in enumerate(reader, start=2):
            document_id = row.get("document_id", "")
            if not document_id or document_id != document_id.strip():
                raise ValueError(
                    f"Identificador inválido no manifesto, linha {line_number}."
                )
            if document_id in eligibility:
                raise ValueError(
                    f"Identificador duplicado no manifesto: {document_id}"
                )

            ingestion_status = (row.get("ingestion_status") or "").strip().lower()
            if ingestion_status not in VALID_INGESTION_STATUSES:
                raise ValueError(
                    f"ingestion_status inválido para {document_id}: "
                    f"{ingestion_status!r}"
                )

            approval = (row.get("include_in_pipeline") or "").strip().lower()
            if approval not in {"", "true", "false"}:
                raise ValueError(
                    f"include_in_pipeline inválido para {document_id}: {approval!r}"
                )
            eligibility[document_id] = (
                ingestion_status == "processed" and approval == "true"
            )

    return eligibility


def _iter_jsonl_records(jsonl_path: Path) -> Iterator[tuple[int, dict[str, object]]]:
    with jsonl_path.open("r", encoding="utf-8") as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSONL inválido na linha {line_number}: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"Registro JSONL inválido na linha {line_number}: "
                    "era esperado um objeto."
                )
            yield line_number, record


def _validate_jsonl_document_ids(
    jsonl_path: Path, manifest_ids: set[str]
) -> None:
    for line_number, record in _iter_jsonl_records(jsonl_path):
        document_id = record.get("document_id")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(
                f"Identificador inválido no JSONL, linha {line_number}."
            )
        if document_id not in manifest_ids:
            raise ValueError(
                f"Identificador do JSONL ausente no manifesto, "
                f"linha {line_number}: {document_id}"
            )


def iter_approved_pages(
    manifest_path: Path, jsonl_path: Path
) -> Iterator[ApprovedPage]:
    """Yield text pages only after validating the complete manifest/JSONL ID set."""
    eligibility = _read_manifest(manifest_path)
    _validate_jsonl_document_ids(jsonl_path, set(eligibility))

    for line_number, record in _iter_jsonl_records(jsonl_path):
        document_id = record["document_id"]
        if not eligibility[document_id]:
            continue

        record_type = record.get("record_type")
        if record_type not in VALID_RECORD_TYPES:
            raise ValueError(
                f"record_type inválido no JSONL, linha {line_number}: "
                f"{record_type!r}"
            )
        if record_type != "page" or record.get("status") != "processed":
            continue

        text = record.get("text")
        if not isinstance(text, str):
            raise ValueError(f"Texto inválido no JSONL, linha {line_number}.")
        if not text.strip():
            continue

        page_number = record.get("page_number")
        if (
            not isinstance(page_number, int)
            or isinstance(page_number, bool)
            or page_number < 1
        ):
            raise ValueError(
                f"Número de página inválido no JSONL, linha {line_number}."
            )
        filename = record.get("filename")
        source_path = record.get("source_path")
        character_count = record.get("character_count")
        pdf_metadata = record.get("pdf_metadata")
        if not isinstance(filename, str) or not isinstance(source_path, str):
            raise ValueError(
                f"Proveniência inválida no JSONL, linha {line_number}."
            )
        if not isinstance(character_count, int) or isinstance(character_count, bool):
            raise ValueError(
                f"Contagem de caracteres inválida no JSONL, linha {line_number}."
            )
        if not isinstance(pdf_metadata, dict):
            raise ValueError(
                f"Metadados PDF inválidos no JSONL, linha {line_number}."
            )

        yield ApprovedPage(
            document_id=document_id,
            filename=filename,
            source_path=source_path,
            page_number=page_number,
            text=text,
            character_count=character_count,
            pdf_metadata=pdf_metadata,
        )
