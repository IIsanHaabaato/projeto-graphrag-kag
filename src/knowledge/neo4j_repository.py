"""Neo4j persistence for pending, evidence-backed extraction results."""

from __future__ import annotations

import json
import re
from typing import Any

from src.config import Neo4jSettings
from src.knowledge.canonicalization import CanonicalizedPage
from src.knowledge.cypher_schema import load_schema_statements


_SAFE_LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _page_id(document_id: str, page_number: int) -> str:
    return f"page:{document_id}:{page_number}"


class Neo4jRepository:
    def __init__(self, driver: Any, database: str) -> None:
        self._driver = driver
        self._database = database

    @classmethod
    def connect(cls, settings: Neo4jSettings) -> "Neo4jRepository":
        try:
            from neo4j import GraphDatabase
        except ImportError as error:
            raise RuntimeError(
                "O driver Neo4j não está instalado. Instale requirements.txt "
                "antes de habilitar persistência."
            ) from error
        driver = GraphDatabase.driver(
            settings.uri,
            auth=(settings.username, settings.password),
        )
        return cls(driver, settings.database)

    def close(self) -> None:
        self._driver.close()

    def apply_schema(self) -> None:
        with self._driver.session(database=self._database) as session:
            for statement in load_schema_statements():
                session.run(statement).consume()

    def persist_page(self, canonicalized: CanonicalizedPage) -> None:
        page = canonicalized.page

        def write(tx: Any) -> None:
            tx.run(
                """
                MERGE (d:Document {document_id: $document_id})
                SET d.filename = $filename,
                    d.source_path = $source_path,
                    d.pdf_metadata_json = $pdf_metadata_json
                """,
                document_id=page.document_id,
                filename=page.filename,
                source_path=page.source_path,
                pdf_metadata_json=json.dumps(
                    page.pdf_metadata, ensure_ascii=False, sort_keys=True
                ),
            ).consume()
            tx.run(
                """
                MERGE (p:Page {page_id: $page_id})
                SET p.document_id = $document_id,
                    p.page_number = $page_number,
                    p.text = $text,
                    p.character_count = $character_count,
                    p.extractor_version = $extractor_version
                WITH p
                MATCH (d:Document {document_id: $document_id})
                MERGE (d)-[:HAS_PAGE]->(p)
                """,
                page_id=_page_id(page.document_id, page.page_number),
                document_id=page.document_id,
                page_number=page.page_number,
                text=page.text,
                character_count=page.character_count,
                extractor_version=canonicalized.extractor_version,
            ).consume()

            entity_ids: set[str] = set()
            for mention in canonicalized.entities:
                entity = mention.entity
                label = entity.entity_type.value
                if not _SAFE_LABEL.fullmatch(label):
                    raise ValueError(f"Rótulo de entidade inválido: {label!r}")
                tx.run(
                    f"""
                    MERGE (e:Entity {{id: $entity_id}})
                    SET e:`{label}`,
                        e.entity_type = $entity_type,
                        e.canonical_name = $canonical_name,
                        e.normalized_name = $normalized_name,
                        e.aliases = reduce(
                            acc = [],
                            alias IN coalesce(e.aliases, []) + $aliases |
                            CASE WHEN alias IN acc THEN acc ELSE acc + alias END
                        ),
                        e.resolution_status = CASE
                            WHEN e.resolution_status = 'resolved'
                                OR $resolution_status = 'resolved'
                            THEN 'resolved'
                            ELSE 'pending_review'
                        END,
                        e.canonical_name_suggestion = $canonical_name_suggestion
                    """,
                    entity_id=entity.entity_id,
                    entity_type=entity.entity_type.value,
                    canonical_name=entity.canonical_name,
                    normalized_name=entity.normalized_name,
                    aliases=list(entity.aliases),
                    resolution_status=entity.resolution_status,
                    canonical_name_suggestion=entity.canonical_name_suggestion,
                ).consume()
                entity_ids.add(entity.entity_id)
                tx.run(
                    """
                    MERGE (ev:Evidence {evidence_id: $evidence_id})
                    SET ev.evidence_type = 'entity_mention',
                        ev.document_id = $document_id,
                        ev.page_number = $page_number,
                        ev.quote = $quote,
                        ev.start_char = $start_char,
                        ev.end_char = $end_char,
                        ev.extractor_version = $extractor_version
                    WITH ev
                    MATCH (p:Page {page_id: $page_id})
                    MATCH (e:Entity {id: $entity_id})
                    MERGE (p)-[:CONTAINS_EVIDENCE]->(ev)
                    MERGE (ev)-[:MENTIONS]->(e)
                    """,
                    evidence_id=mention.evidence_id,
                    document_id=page.document_id,
                    page_number=page.page_number,
                    quote=mention.text,
                    start_char=mention.start_char,
                    end_char=mention.end_char,
                    extractor_version=canonicalized.extractor_version,
                    page_id=_page_id(page.document_id, page.page_number),
                    entity_id=mention.entity.entity_id,
                ).consume()

            for relationship in canonicalized.relationships:
                if (
                    relationship.subject_entity_id not in entity_ids
                    or relationship.object_entity_id not in entity_ids
                ):
                    raise ValueError(
                        "A relação aponta para uma entidade que não foi persistida "
                        "nesta página."
                    )
                tx.run(
                    """
                    MERGE (c:Claim {claim_id: $claim_id})
                    SET c.predicate = $predicate,
                        c.temporal_status = $temporal_status,
                        c.qualifiers_json = $qualifiers_json,
                        c.review_status = $review_status,
                        c.document_id = $document_id,
                        c.page_number = $page_number,
                        c.extractor_version = $extractor_version
                    WITH c
                    MATCH (subject:Entity {id: $subject_entity_id})
                    MATCH (object:Entity {id: $object_entity_id})
                    MERGE (c)-[:SUBJECT]->(subject)
                    MERGE (c)-[:OBJECT]->(object)
                    """,
                    claim_id=relationship.claim_id,
                    predicate=relationship.predicate.value,
                    temporal_status=relationship.temporal_status.value,
                    qualifiers_json=json.dumps(
                        relationship.qualifiers,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    review_status=relationship.review_status,
                    document_id=page.document_id,
                    page_number=page.page_number,
                    extractor_version=canonicalized.extractor_version,
                    subject_entity_id=relationship.subject_entity_id,
                    object_entity_id=relationship.object_entity_id,
                ).consume()
                tx.run(
                    """
                    MERGE (ev:Evidence {evidence_id: $evidence_id})
                    SET ev.evidence_type = 'relationship',
                        ev.document_id = $document_id,
                        ev.page_number = $page_number,
                        ev.quote = $quote,
                        ev.start_char = $start_char,
                        ev.end_char = $end_char,
                        ev.extractor_version = $extractor_version
                    WITH ev
                    MATCH (p:Page {page_id: $page_id})
                    MATCH (c:Claim {claim_id: $claim_id})
                    MERGE (p)-[:CONTAINS_EVIDENCE]->(ev)
                    MERGE (ev)-[:SUPPORTS]->(c)
                    """,
                    evidence_id=relationship.evidence_id,
                    document_id=page.document_id,
                    page_number=page.page_number,
                    quote=relationship.evidence_text,
                    start_char=relationship.evidence_start_char,
                    end_char=relationship.evidence_end_char,
                    extractor_version=canonicalized.extractor_version,
                    page_id=_page_id(page.document_id, page.page_number),
                    claim_id=relationship.claim_id,
                ).consume()

        with self._driver.session(database=self._database) as session:
            session.execute_write(write)
