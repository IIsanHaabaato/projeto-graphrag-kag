"""Strict JSON contract and page-grounded validation for NER/RE responses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.knowledge.ontology import (
    EntityType,
    Predicate,
    TemporalStatus,
    relation_endpoints_are_valid,
)


ENTITY_TYPES = [entity_type.value for entity_type in EntityType]
PREDICATES = [predicate.value for predicate in Predicate]
TEMPORAL_STATUSES = [status.value for status in TemporalStatus]

EXTRACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["entities", "relationships"],
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "mention_id",
                    "text",
                    "type",
                    "start_char",
                    "end_char",
                    "canonical_name_suggestion",
                ],
                "properties": {
                    "mention_id": {"type": "string"},
                    "text": {"type": "string"},
                    "type": {"type": "string", "enum": ENTITY_TYPES},
                    "start_char": {"type": ["integer", "null"]},
                    "end_char": {"type": ["integer", "null"]},
                    "canonical_name_suggestion": {"type": "string"},
                },
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "subject_mention_id",
                    "predicate",
                    "object_mention_id",
                    "evidence_text",
                    "evidence_start_char",
                    "evidence_end_char",
                    "temporal_status",
                    "qualifiers",
                ],
                "properties": {
                    "subject_mention_id": {"type": "string"},
                    "predicate": {"type": "string", "enum": PREDICATES},
                    "object_mention_id": {"type": "string"},
                    "evidence_text": {"type": "string"},
                    "evidence_start_char": {"type": ["integer", "null"]},
                    "evidence_end_char": {"type": ["integer", "null"]},
                    "temporal_status": {
                        "type": "string",
                        "enum": TEMPORAL_STATUSES,
                    },
                    "qualifiers": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                },
            },
        },
    },
}


class ExtractionValidationError(ValueError):
    """Raised when a model response violates the contract or page evidence."""


@dataclass(frozen=True)
class EntityMention:
    mention_id: str
    text: str
    entity_type: EntityType
    start_char: int | None
    end_char: int | None
    canonical_name_suggestion: str


@dataclass(frozen=True)
class RelationshipMention:
    subject_mention_id: str
    predicate: Predicate
    object_mention_id: str
    evidence_text: str
    evidence_start_char: int
    evidence_end_char: int
    temporal_status: TemporalStatus
    qualifiers: dict[str, str]


@dataclass(frozen=True)
class ExtractionResult:
    entities: tuple[EntityMention, ...]
    relationships: tuple[RelationshipMention, ...]


def _validate_pair_of_offsets(
    start: object, end: object, text: str, description: str
) -> tuple[int, int] | None:
    if start is None and end is None:
        return None
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
    ):
        raise ExtractionValidationError(
            f"Offsets inválidos para {description}: ambos devem ser inteiros."
        )
    if start < 0 or end <= start or end > len(text):
        raise ExtractionValidationError(
            f"Intervalo de offsets inválido para {description}."
        )
    return start, end


def _find_unique_span(text: str, phrase: str, description: str) -> tuple[int, int]:
    first = text.find(phrase)
    if first < 0:
        raise ExtractionValidationError(
            f"{description} não aparece literalmente na página."
        )
    if text.find(phrase, first + 1) >= 0:
        raise ExtractionValidationError(
            f"{description} aparece mais de uma vez; offsets explícitos são necessários."
        )
    return first, first + len(phrase)


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExtractionValidationError(f"Campo {field} deve ser texto não vazio.")
    return value


def parse_extraction_response(content: str, page_text: str) -> ExtractionResult:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise ExtractionValidationError(
            f"Resposta não é JSON válido: {error.msg}"
        ) from error
    if not isinstance(payload, dict) or set(payload) != {"entities", "relationships"}:
        raise ExtractionValidationError(
            "A resposta deve conter somente entities e relationships."
        )
    raw_entities = payload["entities"]
    raw_relationships = payload["relationships"]
    if not isinstance(raw_entities, list) or not isinstance(raw_relationships, list):
        raise ExtractionValidationError(
            "entities e relationships devem ser listas."
        )

    entities: list[EntityMention] = []
    mentions_by_id: dict[str, EntityMention] = {}
    for raw_entity in raw_entities:
        required = {
            "mention_id",
            "text",
            "type",
            "start_char",
            "end_char",
            "canonical_name_suggestion",
        }
        if not isinstance(raw_entity, dict) or set(raw_entity) != required:
            raise ExtractionValidationError(
                "Cada entidade deve seguir exatamente o schema definido."
            )
        mention_id = _require_string(raw_entity["mention_id"], "mention_id")
        if mention_id in mentions_by_id:
            raise ExtractionValidationError(f"mention_id duplicado: {mention_id}")
        surface = _require_string(raw_entity["text"], "text")
        if surface not in page_text:
            raise ExtractionValidationError(
                f"Menção {surface!r} não aparece literalmente na página."
            )
        try:
            entity_type = EntityType(raw_entity["type"])
        except (ValueError, TypeError) as error:
            raise ExtractionValidationError(
                f"Tipo de entidade não permitido: {raw_entity['type']!r}"
            ) from error

        offsets = _validate_pair_of_offsets(
            raw_entity["start_char"],
            raw_entity["end_char"],
            page_text,
            f"menção {mention_id}",
        )
        if offsets is not None:
            start_char, end_char = offsets
            if page_text[start_char:end_char] != surface:
                raise ExtractionValidationError(
                    f"Offsets da menção {mention_id} não correspondem ao texto."
                )
        else:
            first = page_text.find(surface)
            if page_text.find(surface, first + 1) < 0:
                offsets = (first, first + len(surface))

        mention = EntityMention(
            mention_id=mention_id,
            text=surface,
            entity_type=entity_type,
            start_char=offsets[0] if offsets else None,
            end_char=offsets[1] if offsets else None,
            canonical_name_suggestion=_require_string(
                raw_entity["canonical_name_suggestion"],
                "canonical_name_suggestion",
            ),
        )
        entities.append(mention)
        mentions_by_id[mention_id] = mention

    relationships: list[RelationshipMention] = []
    for raw_relationship in raw_relationships:
        required = {
            "subject_mention_id",
            "predicate",
            "object_mention_id",
            "evidence_text",
            "evidence_start_char",
            "evidence_end_char",
            "temporal_status",
            "qualifiers",
        }
        if not isinstance(raw_relationship, dict) or set(raw_relationship) != required:
            raise ExtractionValidationError(
                "Cada relação deve seguir exatamente o schema definido."
            )
        subject_id = _require_string(
            raw_relationship["subject_mention_id"], "subject_mention_id"
        )
        object_id = _require_string(
            raw_relationship["object_mention_id"], "object_mention_id"
        )
        subject = mentions_by_id.get(subject_id)
        object_mention = mentions_by_id.get(object_id)
        if subject is None or object_mention is None:
            raise ExtractionValidationError(
                "Relação referencia endpoint sem menção declarada."
            )
        if subject_id == object_id:
            raise ExtractionValidationError(
                "Uma relação não pode ligar uma menção a ela própria."
            )
        try:
            predicate = Predicate(raw_relationship["predicate"])
        except (ValueError, TypeError) as error:
            raise ExtractionValidationError(
                f"Predicado não permitido: {raw_relationship['predicate']!r}"
            ) from error
        if not relation_endpoints_are_valid(
            predicate, subject.entity_type, object_mention.entity_type
        ):
            raise ExtractionValidationError(
                f"Tipos incompatíveis com {predicate.value}: "
                f"{subject.entity_type.value} -> {object_mention.entity_type.value}."
            )

        evidence = _require_string(
            raw_relationship["evidence_text"], "evidence_text"
        )
        offsets = _validate_pair_of_offsets(
            raw_relationship["evidence_start_char"],
            raw_relationship["evidence_end_char"],
            page_text,
            "evidência",
        )
        if offsets is None:
            offsets = _find_unique_span(page_text, evidence, "Evidência")
        elif page_text[offsets[0] : offsets[1]] != evidence:
            raise ExtractionValidationError(
                "Os offsets da evidência não correspondem ao trecho literal."
            )

        try:
            temporal_status = TemporalStatus(raw_relationship["temporal_status"])
        except (ValueError, TypeError) as error:
            raise ExtractionValidationError(
                "Modalidade temporal não permitida."
            ) from error
        qualifiers = raw_relationship["qualifiers"]
        if not isinstance(qualifiers, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in qualifiers.items()
        ):
            raise ExtractionValidationError(
                "qualifiers deve ser um objeto de textos."
            )

        relationships.append(
            RelationshipMention(
                subject_mention_id=subject_id,
                predicate=predicate,
                object_mention_id=object_id,
                evidence_text=evidence,
                evidence_start_char=offsets[0],
                evidence_end_char=offsets[1],
                temporal_status=temporal_status,
                qualifiers=qualifiers,
            )
        )

    return ExtractionResult(tuple(entities), tuple(relationships))
