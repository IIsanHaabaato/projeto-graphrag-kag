"""Read idempotent Neo4j 5 schema statements from the reviewed Cypher files."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONSTRAINTS_PATH = PROJECT_ROOT / "neo4j" / "constraints.cypher"
INDEXES_PATH = PROJECT_ROOT / "neo4j" / "indexes.cypher"


def load_cypher_statements(path: Path) -> list[str]:
    statements = [
        statement.strip()
        for statement in path.read_text(encoding="utf-8").split(";")
        if statement.strip()
    ]
    return statements


def load_schema_statements() -> list[str]:
    return [
        *load_cypher_statements(CONSTRAINTS_PATH),
        *load_cypher_statements(INDEXES_PATH),
    ]
