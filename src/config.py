"""Environment-backed configuration without implicit dotenv or service startup."""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlparse


class ConfigurationError(ValueError):
    """Raised when required service configuration is absent or invalid."""


@dataclass(frozen=True)
class OllamaSettings:
    base_url: str
    model: str
    timeout_seconds: float
    invalid_response_log: str

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> "OllamaSettings":
        values = os.environ if environ is None else environ
        model = values.get("OLLAMA_MODEL", "").strip()
        if not model:
            raise ConfigurationError(
                "Defina OLLAMA_MODEL com uma tag de modelo Ollama local."
            )
        base_url = values.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        ).strip().rstrip("/")
        parsed_url = urlparse(base_url)
        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.netloc
            or parsed_url.query
            or parsed_url.fragment
        ):
            raise ConfigurationError("OLLAMA_BASE_URL deve ser uma URL HTTP(S).")
        try:
            timeout_seconds = float(values.get("OLLAMA_TIMEOUT_SECONDS", "120"))
        except ValueError as error:
            raise ConfigurationError(
                "OLLAMA_TIMEOUT_SECONDS deve ser um número positivo."
            ) from error
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ConfigurationError(
                "OLLAMA_TIMEOUT_SECONDS deve ser um número positivo."
            )
        invalid_response_log = values.get(
            "EXTRACTION_ERROR_LOG",
            "data/processed/extraction_errors.jsonl",
        ).strip()
        if not invalid_response_log:
            raise ConfigurationError("EXTRACTION_ERROR_LOG não pode ser vazio.")
        return cls(
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
            invalid_response_log=invalid_response_log,
        )


@dataclass(frozen=True)
class Neo4jSettings:
    uri: str
    username: str
    password: str
    database: str

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> "Neo4jSettings":
        values = os.environ if environ is None else environ
        required = ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")
        missing = [key for key in required if not values.get(key, "").strip()]
        if missing:
            raise ConfigurationError(
                "Configuração Neo4j ausente: " + ", ".join(missing)
            )
        uri = values["NEO4J_URI"].strip()
        parsed_uri = urlparse(uri)
        if parsed_uri.scheme not in {
            "bolt",
            "bolt+s",
            "bolt+ssc",
            "neo4j",
            "neo4j+s",
            "neo4j+ssc",
        } or not parsed_uri.netloc:
            raise ConfigurationError("NEO4J_URI deve ser uma URI Neo4j válida.")
        return cls(
            uri=uri,
            username=values["NEO4J_USERNAME"].strip(),
            password=values["NEO4J_PASSWORD"],
            database=values.get("NEO4J_DATABASE", "neo4j").strip() or "neo4j",
        )
