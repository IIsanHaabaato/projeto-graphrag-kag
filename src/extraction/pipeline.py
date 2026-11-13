"""Run per-page extraction over the approved corpus and write auditable JSONL."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from src.config import Neo4jSettings, OllamaSettings
from src.extraction.contracts import ExtractionResult
from src.extraction.ollama_client import (
    InvalidModelResponse,
    OllamaExtractor,
)
from src.ingestion.approved_documents import ApprovedPage, iter_approved_pages
from src.knowledge.canonicalization import (
    CanonicalizedPage,
    canonicalize_page,
)
from src.knowledge.neo4j_repository import Neo4jRepository


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PageExtractor(Protocol):
    extractor_version: str

    def extract(self, page: ApprovedPage) -> ExtractionResult: ...


@dataclass(frozen=True)
class PipelineSummary:
    processed_pages: int
    invalid_pages: int
    output_path: str
    persisted_to_neo4j: bool


def _result_record(
    canonicalized: CanonicalizedPage,
) -> dict[str, object]:
    page = canonicalized.page
    return {
        "status": "extracted_pending_review",
        "document_id": page.document_id,
        "filename": page.filename,
        "source_path": page.source_path,
        "page_number": page.page_number,
        "extractor_version": canonicalized.extractor_version,
        "entities": [
            {
                "mention_id": mention.mention_id,
                "text": mention.text,
                "type": mention.entity.entity_type.value,
                "start_char": mention.start_char,
                "end_char": mention.end_char,
                "entity_id": mention.entity.entity_id,
                "canonical_name": mention.entity.canonical_name,
                "canonical_name_suggestion": (
                    mention.entity.canonical_name_suggestion
                ),
                "aliases": list(mention.entity.aliases),
                "resolution_status": mention.entity.resolution_status,
                "evidence_id": mention.evidence_id,
            }
            for mention in canonicalized.entities
        ],
        "relationships": [
            {
                "claim_id": relationship.claim_id,
                "subject_entity_id": relationship.subject_entity_id,
                "predicate": relationship.predicate.value,
                "object_entity_id": relationship.object_entity_id,
                "evidence_id": relationship.evidence_id,
                "evidence_text": relationship.evidence_text,
                "evidence_start_char": relationship.evidence_start_char,
                "evidence_end_char": relationship.evidence_end_char,
                "temporal_status": relationship.temporal_status.value,
                "qualifiers": relationship.qualifiers,
                "review_status": relationship.review_status,
            }
            for relationship in canonicalized.relationships
        ],
    }


def run_pipeline(
    manifest_path: Path,
    pages_path: Path,
    output_path: Path,
    extractor: PageExtractor,
    repository: Neo4jRepository | None = None,
) -> PipelineSummary:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_pages = 0
    invalid_pages = 0

    with output_path.open("w", encoding="utf-8") as output_file:
        for page in iter_approved_pages(manifest_path, pages_path):
            try:
                extraction = extractor.extract(page)
            except InvalidModelResponse as error:
                invalid_pages += 1
                output_file.write(
                    json.dumps(
                        {
                            "status": "invalid_model_response",
                            "document_id": page.document_id,
                            "filename": page.filename,
                            "page_number": page.page_number,
                            "extractor_version": extractor.extractor_version,
                            "error": str(error),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                output_file.flush()
                continue

            canonicalized = canonicalize_page(
                page, extraction, extractor.extractor_version
            )
            if repository is not None:
                repository.persist_page(canonicalized)
            output_file.write(
                json.dumps(
                    _result_record(canonicalized),
                    ensure_ascii=False,
                )
                + "\n"
            )
            output_file.flush()
            processed_pages += 1

    return PipelineSummary(
        processed_pages=processed_pages,
        invalid_pages=invalid_pages,
        output_path=str(output_path),
        persisted_to_neo4j=repository is not None,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrai entidades e relações das páginas aprovadas."
    )
    parser.add_argument(
        "--manifest", type=Path, default=PROJECT_ROOT / "sources.csv"
    )
    parser.add_argument(
        "--pages",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "nasa_pages.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "text_to_graph.jsonl",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Persiste os resultados pendentes no Neo4j.",
    )
    parser.add_argument(
        "--apply-schema",
        action="store_true",
        help="Cria constraints e índices antes da persistência.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    ollama_settings = OllamaSettings.from_environment()
    extractor = OllamaExtractor(ollama_settings)
    repository: Neo4jRepository | None = None
    if args.apply_schema and not args.persist:
        raise ValueError("--apply-schema requer --persist.")
    try:
        if args.persist:
            repository = Neo4jRepository.connect(
                Neo4jSettings.from_environment()
            )
            if args.apply_schema:
                repository.apply_schema()
        summary = run_pipeline(
            args.manifest,
            args.pages,
            args.output,
            extractor,
            repository,
        )
    finally:
        if repository is not None:
            repository.close()
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    return 1 if summary.invalid_pages else 0


if __name__ == "__main__":
    raise SystemExit(main())
