"""Exact-match metrics for manually annotated entity and relation examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class PRFScore:
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class AccuracyScore:
    correct: int
    total: int
    accuracy: float | None


def _prf(gold: set[tuple[object, ...]], predicted: set[tuple[object, ...]]) -> PRFScore:
    true_positive = len(gold & predicted)
    false_positive = len(predicted - gold)
    false_negative = len(gold - predicted)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return PRFScore(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _entity_key(entity: Mapping[str, object]) -> tuple[object, ...]:
    start = entity.get("start_char")
    end = entity.get("end_char")
    entity_type = entity.get("type")
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or not isinstance(entity_type, str)
    ):
        raise ValueError("Anotações de entidade exigem offsets e type válidos.")
    return (start, end, entity_type)


def _relation_key(relation: Mapping[str, object]) -> tuple[object, ...]:
    subject_span = relation.get("subject_span")
    object_span = relation.get("object_span")
    predicate = relation.get("predicate")
    if (
        not isinstance(subject_span, list)
        or len(subject_span) != 2
        or not isinstance(object_span, list)
        or len(object_span) != 2
        or not isinstance(predicate, str)
        or any(
            not isinstance(offset, int) or isinstance(offset, bool)
            for offset in subject_span + object_span
        )
    ):
        raise ValueError(
            "Anotações de relação exigem subject_span, object_span e predicate."
        )
    return (*subject_span, predicate, *object_span)


def entity_scores(
    gold: Iterable[Mapping[str, object]],
    predicted: Iterable[Mapping[str, object]],
) -> PRFScore:
    return _prf(
        {_entity_key(entity) for entity in gold},
        {_entity_key(entity) for entity in predicted},
    )


def relation_scores(
    gold: Iterable[Mapping[str, object]],
    predicted: Iterable[Mapping[str, object]],
) -> PRFScore:
    return _prf(
        {_relation_key(relation) for relation in gold},
        {_relation_key(relation) for relation in predicted},
    )


def canonicalization_accuracy(
    gold: Iterable[Mapping[str, object]],
    predicted: Iterable[Mapping[str, object]],
) -> AccuracyScore:
    gold_by_key = {
        _entity_key(entity): entity.get("canonical_entity_id")
        for entity in gold
        if entity.get("canonical_entity_id") is not None
    }
    predicted_by_key = {
        _entity_key(entity): entity.get("canonical_entity_id")
        for entity in predicted
    }
    comparable = set(gold_by_key) & set(predicted_by_key)
    correct = sum(
        gold_by_key[key] == predicted_by_key[key] for key in comparable
    )
    return AccuracyScore(
        correct=correct,
        total=len(comparable),
        accuracy=correct / len(comparable) if comparable else None,
    )


def evidence_accuracy(
    relationships: Iterable[Mapping[str, object]], page_text: str
) -> AccuracyScore:
    relation_list = list(relationships)
    correct = 0
    for relation in relation_list:
        quote = relation.get("evidence_text")
        start = relation.get("evidence_start_char")
        end = relation.get("evidence_end_char")
        if (
            isinstance(quote, str)
            and isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
            and 0 <= start < end <= len(page_text)
            and page_text[start:end] == quote
        ):
            correct += 1
    total = len(relation_list)
    return AccuracyScore(
        correct=correct,
        total=total,
        accuracy=correct / total if total else None,
    )


def temporal_status_accuracy(
    gold: Iterable[Mapping[str, object]],
    predicted: Iterable[Mapping[str, object]],
) -> AccuracyScore:
    gold_by_key = {
        _relation_key(relation): relation.get("temporal_status")
        for relation in gold
        if relation.get("temporal_status") is not None
    }
    predicted_by_key = {
        _relation_key(relation): relation.get("temporal_status")
        for relation in predicted
    }
    comparable = set(gold_by_key) & set(predicted_by_key)
    correct = sum(
        gold_by_key[key] == predicted_by_key[key] for key in comparable
    )
    return AccuracyScore(
        correct=correct,
        total=len(comparable),
        accuracy=correct / len(comparable) if comparable else None,
    )
