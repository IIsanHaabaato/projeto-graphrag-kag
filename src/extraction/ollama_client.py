"""Local Ollama structured-output client with bounded validation retry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.config import OllamaSettings
from src.extraction.contracts import (
    EXTRACTION_JSON_SCHEMA,
    ExtractionResult,
    ExtractionValidationError,
    parse_extraction_response,
)
from src.ingestion.approved_documents import ApprovedPage
from src.knowledge.ontology import EntityType, Predicate, TemporalStatus


EXTRACTOR_VERSION = "ollama-json-schema-v1"


class OllamaServiceError(RuntimeError):
    """Raised when the Ollama endpoint cannot produce an API response."""


class InvalidModelResponse(RuntimeError):
    """Raised after the model response fails strict validation twice."""


def _system_prompt() -> str:
    entity_types = ", ".join(entity_type.value for entity_type in EntityType)
    predicates = ", ".join(predicate.value for predicate in Predicate)
    temporal_statuses = ", ".join(status.value for status in TemporalStatus)
    return (
        "You extract only facts explicitly stated in the supplied page. "
        "Return the required JSON object, with no prose or markdown. "
        "Do not infer missing relationships. Each entity text and relationship "
        "evidence_text must be a verbatim substring of the page, with zero-based "
        "end-exclusive character offsets. Use only these entity types: "
        f"{entity_types}. Use only these predicates: {predicates}. "
        f"Use only these temporal statuses: {temporal_statuses}. "
        "Every relationship endpoint must refer to a declared mention_id. "
        "canonical_name_suggestion is only a suggestion; do not merge entities."
    )


def _user_prompt(page: ApprovedPage) -> str:
    return (
        f"Document ID: {page.document_id}\n"
        f"Filename: {page.filename}\n"
        f"Page number: {page.page_number}\n"
        "Extract entities and relationships from this page only:\n"
        f"<page>\n{page.text}\n</page>"
    )


class OllamaExtractor:
    def __init__(self, settings: OllamaSettings) -> None:
        self.settings = settings
        self.extractor_version = EXTRACTOR_VERSION

    def _chat(self, messages: list[dict[str, str]]) -> str:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "stream": False,
            "format": EXTRACTION_JSON_SCHEMA,
            "messages": messages,
        }
        request = Request(
            f"{self.settings.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.settings.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise OllamaServiceError(
                f"Ollama retornou HTTP {error.code}: {detail}"
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            raise OllamaServiceError(f"Falha ao consultar Ollama: {error}") from error
        except json.JSONDecodeError as error:
            raise OllamaServiceError("Ollama retornou JSON de API inválido.") from error

        message = body.get("message") if isinstance(body, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise OllamaServiceError(
                "Resposta da API Ollama não contém message.content textual."
            )
        return content

    def _record_invalid_response(
        self,
        page: ApprovedPage,
        attempt: int,
        response: str,
        error: ExtractionValidationError,
    ) -> None:
        log_path = Path(self.settings.invalid_response_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "document_id": page.document_id,
            "filename": page.filename,
            "page_number": page.page_number,
            "attempt": attempt,
            "error": str(error),
            "response": response,
        }
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def extract(self, page: ApprovedPage) -> ExtractionResult:
        messages = [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": _user_prompt(page)},
        ]
        for attempt in (1, 2):
            response = self._chat(messages)
            try:
                return parse_extraction_response(response, page.text)
            except ExtractionValidationError as error:
                self._record_invalid_response(page, attempt, response, error)
                if attempt == 2:
                    raise InvalidModelResponse(
                        f"Resposta inválida duas vezes para "
                        f"{page.document_id}, página {page.page_number}: {error}"
                    ) from error
                messages.extend(
                    [
                        {"role": "assistant", "content": response},
                        {
                            "role": "user",
                            "content": (
                                "The previous response failed validation: "
                                f"{error}. Return a corrected JSON object that "
                                "strictly follows the supplied schema and page."
                            ),
                        },
                    ]
                )
        raise AssertionError("bounded extraction loop should always return or raise")
