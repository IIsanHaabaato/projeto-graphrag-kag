from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import ConfigurationError, Neo4jSettings, OllamaSettings
from src.evaluation.metrics import (
    canonicalization_accuracy,
    entity_scores,
    evidence_accuracy,
    relation_scores,
    temporal_status_accuracy,
)
from src.extraction.contracts import (
    EXTRACTION_JSON_SCHEMA,
    ExtractionValidationError,
    parse_extraction_response,
)
from src.extraction.ollama_client import (
    InvalidModelResponse,
    OllamaExtractor,
    OllamaServiceError,
)
from src.extraction.pipeline import run_pipeline
from src.ingestion.approved_documents import ApprovedPage
from src.knowledge.canonicalization import canonicalize_page
from src.knowledge.cypher_schema import load_schema_statements
from src.knowledge.neo4j_repository import Neo4jRepository
from src.knowledge.ontology import Predicate


def _entity(
    mention_id: str,
    text: str,
    entity_type: str,
    page_text: str,
    occurrence: int = 0,
) -> dict[str, object]:
    start = -1
    for _ in range(occurrence + 1):
        start = page_text.find(text, start + 1)
    if start < 0:
        raise AssertionError(f"{text!r} occurrence {occurrence} not found")
    return {
        "mention_id": mention_id,
        "text": text,
        "type": entity_type,
        "start_char": start,
        "end_char": start + len(text),
        "canonical_name_suggestion": text,
    }


def _relation(
    subject_id: str,
    predicate: str,
    object_id: str,
    evidence: str,
    page_text: str,
    occurrence: int = 0,
) -> dict[str, object]:
    start = -1
    for _ in range(occurrence + 1):
        start = page_text.find(evidence, start + 1)
    if start < 0:
        raise AssertionError(f"{evidence!r} occurrence {occurrence} not found")
    return {
        "subject_mention_id": subject_id,
        "predicate": predicate,
        "object_mention_id": object_id,
        "evidence_text": evidence,
        "evidence_start_char": start,
        "evidence_end_char": start + len(evidence),
        "temporal_status": "historical",
        "qualifiers": {},
    }


def _page(text: str = "Artemis I uses Space Launch System.") -> ApprovedPage:
    return ApprovedPage(
        document_id="nasa-artemis-i",
        filename="artemis.pdf",
        source_path="NASA/artemis.pdf",
        page_number=1,
        text=text,
        character_count=len(text),
        pdf_metadata={"title": "Artemis"},
    )


def _valid_payload(page_text: str) -> dict[str, object]:
    evidence = "Artemis I uses Space Launch System."
    return {
        "entities": [
            _entity("mission-1", "Artemis I", "Mission", page_text),
            _entity(
                "vehicle-1",
                "Space Launch System",
                "LaunchVehicle",
                page_text,
            ),
        ],
        "relationships": [
            _relation(
                "mission-1",
                "USES_LAUNCH_VEHICLE",
                "vehicle-1",
                evidence,
                page_text,
            )
        ],
    }


class ExtractionContractTests(unittest.TestCase):
    def test_valid_page_grounded_entities_and_relationships_parse(self) -> None:
        page_text = _page().text
        result = parse_extraction_response(
            json.dumps(_valid_payload(page_text)), page_text
        )

        self.assertEqual(len(result.entities), 2)
        self.assertEqual(len(result.relationships), 1)
        self.assertEqual(
            result.relationships[0].predicate,
            Predicate.USES_LAUNCH_VEHICLE,
        )
        self.assertEqual(EXTRACTION_JSON_SCHEMA["type"], "object")

    def test_invalid_json_and_nonliteral_evidence_are_rejected(self) -> None:
        with self.assertRaisesRegex(ExtractionValidationError, "JSON válido"):
            parse_extraction_response("{not json", _page().text)

        payload = _valid_payload(_page().text)
        payload["relationships"][0]["evidence_text"] = "Artemis is connected."
        with self.assertRaisesRegex(
            ExtractionValidationError, "offsets da evidência"
        ):
            parse_extraction_response(json.dumps(payload), _page().text)

    def test_relationships_require_known_endpoints_and_valid_types(self) -> None:
        page_text = _page().text
        payload = _valid_payload(page_text)
        payload["relationships"][0]["object_mention_id"] = "missing"
        with self.assertRaisesRegex(ExtractionValidationError, "endpoint"):
            parse_extraction_response(json.dumps(payload), page_text)

        payload = _valid_payload(page_text)
        payload["relationships"][0]["predicate"] = "TARGETS"
        with self.assertRaisesRegex(ExtractionValidationError, "Tipos incompatíveis"):
            parse_extraction_response(json.dumps(payload), page_text)

    def test_unlisted_fields_and_duplicate_mention_ids_are_rejected(self) -> None:
        page_text = _page().text
        payload = _valid_payload(page_text)
        payload["surprise"] = True
        with self.assertRaisesRegex(ExtractionValidationError, "somente"):
            parse_extraction_response(json.dumps(payload), page_text)

        payload = _valid_payload(page_text)
        payload["entities"][1]["mention_id"] = "mission-1"
        with self.assertRaisesRegex(ExtractionValidationError, "duplicado"):
            parse_extraction_response(json.dumps(payload), page_text)


class CanonicalizationTests(unittest.TestCase):
    def test_explicit_aliases_resolve_and_ambiguous_names_stay_distinct(self) -> None:
        page_text = (
            "Artemis 1, Orion, Orion Crew Module, Mars 2020 Mission, "
            "Perseverance rover"
        )
        entities = [
            _entity("artemis", "Artemis 1", "Mission", page_text),
            _entity("orion", "Orion", "Spacecraft", page_text),
            _entity(
                "crew-module",
                "Orion Crew Module",
                "Spacecraft",
                page_text,
            ),
            _entity(
                "mars-mission",
                "Mars 2020 Mission",
                "Mission",
                page_text,
            ),
            _entity(
                "perseverance",
                "Perseverance rover",
                "Spacecraft",
                page_text,
            ),
        ]
        parsed = parse_extraction_response(
            json.dumps({"entities": entities, "relationships": []}),
            page_text,
        )
        result = canonicalize_page(_page(page_text), parsed, "test")
        by_mention = {item.mention_id: item.entity for item in result.entities}

        self.assertEqual(by_mention["artemis"].entity_id, "entity:mission:artemis-i")
        self.assertEqual(by_mention["artemis"].resolution_status, "resolved")
        self.assertEqual(by_mention["orion"].resolution_status, "pending_review")
        self.assertNotEqual(
            by_mention["orion"].entity_id,
            by_mention["crew-module"].entity_id,
        )
        self.assertNotEqual(
            by_mention["mars-mission"].entity_id,
            by_mention["perseverance"].entity_id,
        )


class OllamaClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.error_log = Path(self.temporary_directory.name) / "errors.jsonl"
        self.settings = OllamaSettings(
            base_url="http://localhost:11434",
            model="local-test-model",
            timeout_seconds=1,
            invalid_response_log=str(self.error_log),
        )
        self.extractor = OllamaExtractor(self.settings)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_invalid_response_is_logged_then_corrected_once(self) -> None:
        content = json.dumps(_valid_payload(_page().text))
        with patch.object(
            self.extractor, "_chat", side_effect=["not json", content]
        ) as chat:
            result = self.extractor.extract(_page())

        self.assertEqual(len(result.entities), 2)
        self.assertEqual(chat.call_count, 2)
        log_records = [
            json.loads(line)
            for line in self.error_log.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(log_records), 1)
        self.assertEqual(log_records[0]["document_id"], "nasa-artemis-i")
        self.assertEqual(log_records[0]["page_number"], 1)

    def test_two_invalid_responses_are_logged_and_rejected(self) -> None:
        with patch.object(self.extractor, "_chat", return_value="not json"):
            with self.assertRaises(InvalidModelResponse):
                self.extractor.extract(_page())

        records = self.error_log.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(records), 2)

    def test_service_errors_are_not_hidden_or_retried(self) -> None:
        with patch.object(
            self.extractor,
            "_chat",
            side_effect=OllamaServiceError("offline"),
        ) as chat:
            with self.assertRaisesRegex(OllamaServiceError, "offline"):
                self.extractor.extract(_page())
        chat.assert_called_once()

    def test_chat_request_uses_configured_local_model_and_json_schema(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args: object) -> None:
                del args

            def read(self) -> bytes:
                return json.dumps(
                    {"message": {"content": '{"entities":[],"relationships":[]}'}}
                ).encode("utf-8")

        with patch(
            "src.extraction.ollama_client.urlopen",
            return_value=FakeResponse(),
        ) as open_url:
            content = self.extractor._chat(
                [{"role": "user", "content": "page"}]
            )

        request = open_url.call_args.args[0]
        request_body = json.loads(request.data)
        self.assertEqual(request.full_url, "http://localhost:11434/api/chat")
        self.assertEqual(request_body["model"], "local-test-model")
        self.assertEqual(request_body["format"], EXTRACTION_JSON_SCHEMA)
        self.assertIn('"entities":[]', content)


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.manifest_path = self.root / "sources.csv"
        self.pages_path = self.root / "pages.jsonl"
        self.output_path = self.root / "result.jsonl"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_inputs(self) -> None:
        fields = [
            "document_id",
            "ingestion_status",
            "include_in_pipeline",
            "review_reason",
        ]
        with self.manifest_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "document_id": "approved",
                        "ingestion_status": "processed",
                        "include_in_pipeline": "true",
                        "review_reason": "approved",
                    },
                    {
                        "document_id": "rejected",
                        "ingestion_status": "processed",
                        "include_in_pipeline": "false",
                        "review_reason": "rejected",
                    },
                ]
            )
        records = []
        for document_id in ("approved", "rejected"):
            records.append(
                {
                    "record_type": "page",
                    "document_id": document_id,
                    "filename": f"{document_id}.pdf",
                    "source_path": f"NASA/{document_id}.pdf",
                    "page_number": 1,
                    "text": "Artemis I uses Space Launch System.",
                    "character_count": len("Artemis I uses Space Launch System."),
                    "pdf_metadata": {},
                    "status": "processed",
                    "error": None,
                }
            )
        self.pages_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    def test_only_approved_pages_reach_extractor_and_output(self) -> None:
        self._write_inputs()

        class FakeExtractor:
            extractor_version = "test"

            def __init__(self) -> None:
                self.document_ids: list[str] = []

            def extract(self, page: ApprovedPage):
                self.document_ids.append(page.document_id)
                return parse_extraction_response(
                    json.dumps(_valid_payload(page.text)), page.text
                )

        extractor = FakeExtractor()
        summary = run_pipeline(
            self.manifest_path,
            self.pages_path,
            self.output_path,
            extractor,
        )
        output = json.loads(self.output_path.read_text(encoding="utf-8"))

        self.assertEqual(extractor.document_ids, ["approved"])
        self.assertEqual(summary.processed_pages, 1)
        self.assertFalse(summary.persisted_to_neo4j)
        self.assertEqual(output["status"], "extracted_pending_review")
        self.assertEqual(output["document_id"], "approved")

    def test_invalid_page_is_recorded_but_not_persisted(self) -> None:
        self._write_inputs()

        class FakeExtractor:
            extractor_version = "test"

            def extract(self, page: ApprovedPage):
                del page
                raise InvalidModelResponse("invalid page response")

        summary = run_pipeline(
            self.manifest_path,
            self.pages_path,
            self.output_path,
            FakeExtractor(),
        )
        output = json.loads(self.output_path.read_text(encoding="utf-8"))

        self.assertEqual(summary.processed_pages, 0)
        self.assertEqual(summary.invalid_pages, 1)
        self.assertEqual(output["status"], "invalid_model_response")


class Neo4jRepositoryTests(unittest.TestCase):
    class _Result:
        def consume(self) -> None:
            return None

    class _Transaction:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def run(self, query: str, **parameters: object):
            self.calls.append((query, parameters))
            return Neo4jRepositoryTests._Result()

    class _Session:
        def __init__(self) -> None:
            self.transaction = Neo4jRepositoryTests._Transaction()
            self.schema_calls: list[str] = []

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            del args
            return None

        def execute_write(self, callback):
            return callback(self.transaction)

        def run(self, query: str):
            self.schema_calls.append(query)
            return Neo4jRepositoryTests._Result()

    class _Driver:
        def __init__(self) -> None:
            self.current_session = Neo4jRepositoryTests._Session()

        def session(self, database: str):
            del database
            return self.current_session

        def close(self) -> None:
            return None

    def test_persistence_keeps_distinct_evidence_and_pending_claims(self) -> None:
        text = "Artemis I uses Space Launch System. Artemis I uses Space Launch System."
        entities = [
            _entity("mission-1", "Artemis I", "Mission", text, 0),
            _entity("vehicle-1", "Space Launch System", "LaunchVehicle", text, 0),
            _entity("mission-2", "Artemis I", "Mission", text, 1),
            _entity("vehicle-2", "Space Launch System", "LaunchVehicle", text, 1),
        ]
        relationships = [
            _relation(
                "mission-1",
                "USES_LAUNCH_VEHICLE",
                "vehicle-1",
                "Artemis I uses Space Launch System.",
                text,
                0,
            ),
            _relation(
                "mission-2",
                "USES_LAUNCH_VEHICLE",
                "vehicle-2",
                "Artemis I uses Space Launch System.",
                text,
                1,
            ),
        ]
        extraction = parse_extraction_response(
            json.dumps({"entities": entities, "relationships": relationships}),
            text,
        )
        canonicalized = canonicalize_page(_page(text), extraction, "test-version")
        driver = self._Driver()
        repository = Neo4jRepository(driver, "neo4j")

        repository.persist_page(canonicalized)

        calls = driver.current_session.transaction.calls
        claim_calls = [params for query, params in calls if "MERGE (c:Claim" in query]
        evidence_calls = [
            params for query, params in calls if "MERGE (ev:Evidence" in query
        ]
        relationship_evidence_calls = [
            params
            for query, params in calls
            if "SET ev.evidence_type = 'relationship'" in query
        ]
        self.assertEqual(len(claim_calls), 2)
        self.assertEqual(len({params["claim_id"] for params in claim_calls}), 2)
        self.assertEqual(len(evidence_calls), 6)
        self.assertEqual(len(relationship_evidence_calls), 2)
        self.assertTrue(all(params["review_status"] == "pending" for params in claim_calls))

    def test_schema_includes_range_fulltext_vector_and_not_legacy_btree(self) -> None:
        statements = load_schema_statements()
        schema = "\n".join(statements).upper()
        self.assertIn("CREATE RANGE INDEX", schema)
        self.assertIn("CREATE FULLTEXT INDEX", schema)
        self.assertIn("CREATE VECTOR INDEX", schema)
        self.assertNotIn("BTREE", schema)
        self.assertNotIn("VECTOR.DIMENSIONS", schema)


class ConfigurationAndEvaluationTests(unittest.TestCase):
    def test_settings_require_model_and_neo4j_credentials_only_when_requested(self) -> None:
        with self.assertRaises(ConfigurationError):
            OllamaSettings.from_environment({})
        with self.assertRaises(ConfigurationError):
            Neo4jSettings.from_environment({})
        with self.assertRaises(ConfigurationError):
            OllamaSettings.from_environment(
                {"OLLAMA_MODEL": "model", "OLLAMA_TIMEOUT_SECONDS": "inf"}
            )
        with self.assertRaises(ConfigurationError):
            Neo4jSettings.from_environment(
                {
                    "NEO4J_URI": "http://localhost:7687",
                    "NEO4J_USERNAME": "neo4j",
                    "NEO4J_PASSWORD": "test",
                }
            )

        settings = OllamaSettings.from_environment(
            {"OLLAMA_MODEL": "user-selected-model"}
        )
        self.assertEqual(settings.model, "user-selected-model")
        self.assertEqual(settings.base_url, "http://localhost:11434")

    def test_exact_metrics_cover_entities_relations_evidence_and_temporality(self) -> None:
        gold_entity = {
            "start_char": 0,
            "end_char": 9,
            "type": "Mission",
            "canonical_entity_id": "entity:mission:artemis-i",
        }
        predicted_entity = dict(gold_entity)
        self.assertEqual(entity_scores([gold_entity], [predicted_entity]).f1, 1.0)
        self.assertEqual(
            canonicalization_accuracy(
                [gold_entity], [predicted_entity]
            ).accuracy,
            1.0,
        )
        relation = {
            "subject_span": [0, 9],
            "predicate": "USES_LAUNCH_VEHICLE",
            "object_span": [15, 34],
            "temporal_status": "historical",
            "evidence_text": "Artemis I uses Space Launch System.",
            "evidence_start_char": 0,
            "evidence_end_char": len("Artemis I uses Space Launch System."),
        }
        self.assertEqual(relation_scores([relation], [relation]).f1, 1.0)
        self.assertEqual(
            evidence_accuracy([relation], "Artemis I uses Space Launch System.").accuracy,
            1.0,
        )
        self.assertEqual(
            temporal_status_accuracy([relation], [relation]).accuracy,
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
