"""Conservative, explicit alias resolution for extracted entity mentions."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass

from src.extraction.contracts import EntityMention, ExtractionResult
from src.ingestion.approved_documents import ApprovedPage
from src.knowledge.ontology import EntityType, Predicate, TemporalStatus


@dataclass(frozen=True)
class EntityIdentity:
    entity_id: str
    entity_type: EntityType
    canonical_name: str
    normalized_name: str
    aliases: tuple[str, ...]
    resolution_status: str
    canonical_name_suggestion: str


@dataclass(frozen=True)
class CanonicalEntityMention:
    mention_id: str
    entity: EntityIdentity
    text: str
    start_char: int | None
    end_char: int | None
    evidence_id: str


@dataclass(frozen=True)
class CanonicalRelationship:
    claim_id: str
    predicate: Predicate
    subject_entity_id: str
    object_entity_id: str
    evidence_id: str
    evidence_text: str
    evidence_start_char: int
    evidence_end_char: int
    temporal_status: TemporalStatus
    qualifiers: dict[str, str]
    review_status: str = "pending"


@dataclass(frozen=True)
class CanonicalizedPage:
    page: ApprovedPage
    entities: tuple[CanonicalEntityMention, ...]
    relationships: tuple[CanonicalRelationship, ...]
    extractor_version: str


def normalize_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(" \t\r\n.,;:()[]{}")


def _stable_id(prefix: str, *parts: object) -> str:
    material = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


# Only unambiguous forms are mapped. Ambiguous forms such as "Orion" remain
# unresolved, and the Mars 2020 mission is distinct from its Perseverance rover.
_EXPLICIT_ALIASES: dict[
    tuple[EntityType, str], tuple[str, str, tuple[str, ...]]
] = {
    (EntityType.MISSION, "artemis i"): (
        "entity:mission:artemis-i",
        "Artemis I",
        ("Artemis I", "Artemis 1"),
    ),
    (EntityType.MISSION, "artemis 1"): (
        "entity:mission:artemis-i",
        "Artemis I",
        ("Artemis I", "Artemis 1"),
    ),
    (EntityType.MISSION, "mars 2020 mission"): (
        "entity:mission:mars-2020",
        "Mars 2020 Mission",
        ("Mars 2020 Mission",),
    ),
    (EntityType.SPACECRAFT, "perseverance"): (
        "entity:spacecraft:perseverance-rover",
        "Perseverance rover",
        ("Perseverance", "Perseverance rover"),
    ),
    (EntityType.SPACECRAFT, "perseverance rover"): (
        "entity:spacecraft:perseverance-rover",
        "Perseverance rover",
        ("Perseverance", "Perseverance rover"),
    ),
    (EntityType.SPACECRAFT, "orion crew module"): (
        "entity:spacecraft:orion-crew-module",
        "Orion Crew Module",
        ("Orion Crew Module",),
    ),
}


def _identity_for(page: ApprovedPage, mention: EntityMention) -> EntityIdentity:
    normalized = normalize_name(mention.text)
    explicit = _EXPLICIT_ALIASES.get((mention.entity_type, normalized))
    if explicit is not None:
        entity_id, canonical_name, aliases = explicit
        observed_aliases = tuple(dict.fromkeys((*aliases, mention.text)))
        return EntityIdentity(
            entity_id=entity_id,
            entity_type=mention.entity_type,
            canonical_name=canonical_name,
            normalized_name=normalize_name(canonical_name),
            aliases=observed_aliases,
            resolution_status="resolved",
            canonical_name_suggestion=mention.canonical_name_suggestion,
        )

    entity_id = _stable_id(
        "entity:pending",
        page.document_id,
        page.page_number,
        mention.entity_type.value,
        normalized,
        mention.start_char,
        mention.end_char,
        mention.mention_id if mention.start_char is None else "",
    )
    return EntityIdentity(
        entity_id=entity_id,
        entity_type=mention.entity_type,
        canonical_name=mention.text,
        normalized_name=normalized,
        aliases=(mention.text,),
        resolution_status="pending_review",
        canonical_name_suggestion=mention.canonical_name_suggestion,
    )


def canonicalize_page(
    page: ApprovedPage,
    extraction: ExtractionResult,
    extractor_version: str,
) -> CanonicalizedPage:
    """Resolve explicit aliases; retain all other mentions as separate candidates."""
    mentions: dict[str, CanonicalEntityMention] = {}
    for mention in extraction.entities:
        identity = _identity_for(page, mention)
        evidence_id = _stable_id(
            "evidence",
            page.document_id,
            page.page_number,
            "entity_mention",
            mention.entity_type.value,
            mention.start_char,
            mention.end_char,
            mention.text,
            mention.mention_id if mention.start_char is None else "",
        )
        mentions[mention.mention_id] = CanonicalEntityMention(
            mention_id=mention.mention_id,
            entity=identity,
            text=mention.text,
            start_char=mention.start_char,
            end_char=mention.end_char,
            evidence_id=evidence_id,
        )

    relationships: list[CanonicalRelationship] = []
    for relationship in extraction.relationships:
        subject = mentions[relationship.subject_mention_id]
        object_mention = mentions[relationship.object_mention_id]
        evidence_id = _stable_id(
            "evidence",
            page.document_id,
            page.page_number,
            "relationship",
            relationship.predicate.value,
            relationship.evidence_start_char,
            relationship.evidence_end_char,
            relationship.evidence_text,
        )
        claim_id = _stable_id(
            "claim",
            page.document_id,
            page.page_number,
            relationship.predicate.value,
            subject.entity.entity_id,
            object_mention.entity.entity_id,
            evidence_id,
        )
        relationships.append(
            CanonicalRelationship(
                claim_id=claim_id,
                predicate=relationship.predicate,
                subject_entity_id=subject.entity.entity_id,
                object_entity_id=object_mention.entity.entity_id,
                evidence_id=evidence_id,
                evidence_text=relationship.evidence_text,
                evidence_start_char=relationship.evidence_start_char,
                evidence_end_char=relationship.evidence_end_char,
                temporal_status=relationship.temporal_status,
                qualifiers=relationship.qualifiers,
            )
        )

    return CanonicalizedPage(
        page=page,
        entities=tuple(mentions.values()),
        relationships=tuple(relationships),
        extractor_version=extractor_version,
    )
