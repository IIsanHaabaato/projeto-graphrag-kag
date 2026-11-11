"""Protótipo executável de GraphRAG/KAG para Ciência da Computação.

O programa funciona sem serviços externos e pode gerar um script Cypher para
Neo4j. Ele implementa os quatro pilares do enunciado: Text-to-Graph,
modelagem LPG, recuperação híbrida com multi-hop e rastreabilidade.

Exemplos:
    python "PROJETO RICARDO.py" --demo
    python "PROJETO RICARDO.py" --file documentos.txt
    python "PROJETO RICARDO.py" --file documentos --cypher schema.cypher
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


STOPWORDS = {
    "com", "como", "dos", "das", "entre", "para", "que", "uma", "por",
    "sobre", "qual", "quais", "onde", "está", "esta", "este", "são", "tem",
}


def normalize(value: str) -> str:
    """Cria uma chave estável para resolução de entidades equivalentes."""
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def terms(value: str) -> set[str]:
    return {
        normalize(term)
        for term in re.findall(r"\b[\wÀ-ÿ]{3,}\b", value.lower())
        if normalize(term) not in STOPWORDS
    }


@dataclass(frozen=True)
class Document:
    identifier: str
    text: str
    terms: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", frozenset(terms(self.text)))


@dataclass(frozen=True)
class Node:
    identifier: str
    label: str
    name: str
    source: str


@dataclass(frozen=True)
class Relationship:
    source: str
    relation: str
    target: str
    source_document: str


@dataclass(frozen=True)
class RetrievalResult:
    answer: str
    documents: list[str]
    nodes: list[str]
    paths: list[str]
    cypher: str
    confidence: float


class KnowledgeGraph:
    """LPG local com canonicalização e travessia orientada de dois ou mais hops."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.relationships: set[Relationship] = set()
        self._outgoing: defaultdict[str, set[str]] = defaultdict(set)

    def add_node(self, label: str, name: str, source: str) -> str:
        identifier = f"{label.lower()}:{normalize(name)}"
        self.nodes.setdefault(identifier, Node(identifier, label, name, source))
        return identifier

    def add_relationship(
        self, source: str, relation: str, target: str, source_document: str
    ) -> None:
        item = Relationship(source, relation.upper(), target, source_document)
        self.relationships.add(item)
        self._outgoing[source].add(target)

    def neighbors(self, seed: str, hops: int = 2) -> set[str]:
        if hops < 1:
            raise ValueError("A travessia precisa ter ao menos um salto.")
        visited = {seed}
        frontier = {seed}
        for _ in range(hops):
            frontier = {
                target
                for node_id in frontier
                for target in self._outgoing[node_id]
                if target not in visited
            }
            visited.update(frontier)
        visited.remove(seed)
        return visited

    def paths(self, selected: set[str]) -> list[str]:
        return [
            f"{self.nodes[item.source].name} -[{item.relation}]-> "
            f"{self.nodes[item.target].name}"
            for item in sorted(self.relationships, key=lambda rel: (rel.source, rel.target))
            if item.source in selected and item.target in selected
        ]

    def cypher_context(self, seeds: list[str], hops: int = 2) -> str:
        escaped = [seed.replace("'", "\\'") for seed in seeds]
        return (
            "MATCH path=(seed)-[*1..%d]-(context) "
            "WHERE seed.id IN [%s] "
            "RETURN path LIMIT 50"
        ) % (hops, ", ".join(f"'{seed}'" for seed in escaped))

    def schema_cypher(self) -> str:
        return """// Schema LPG GraphRAG/KAG
CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE;
CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name);
CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
FOR (n:Entity) ON EACH [n.name, n.description];
// Crie o índice vetorial após definir a dimensão do embedding do grupo:
// CREATE VECTOR INDEX entity_embedding IF NOT EXISTS FOR (n:Entity)
// ON (n.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 384,
// `vector.similarity_function`: 'cosine'}};
"""

    def data_cypher(self) -> str:
        """Exporta o grafo local para nós e relações compatíveis com Neo4j."""
        lines = ["// Dados extraídos pelo pipeline Text-to-Graph"]
        for node in sorted(self.nodes.values(), key=lambda item: item.identifier):
            lines.append(
                "MERGE (n:%s {id: %s}) SET n.name = %s, n.source = %s;"
                % (
                    node.label,
                    self._cypher_string(node.identifier),
                    self._cypher_string(node.name),
                    self._cypher_string(node.source),
                )
            )
        for relationship in sorted(
            self.relationships, key=lambda item: (item.source, item.relation, item.target)
        ):
            lines.append(
                "MATCH (source {id: %s}), (target {id: %s}) "
                "MERGE (source)-[:%s {source_document: %s}]->(target);"
                % (
                    self._cypher_string(relationship.source),
                    self._cypher_string(relationship.target),
                    relationship.relation,
                    self._cypher_string(relationship.source_document),
                )
            )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _cypher_string(value: str) -> str:
        return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


class TextToGraph:
    """Extrator determinístico de entidades e relações para a demonstração."""

    RELATION_PATTERNS = (
        (r"(\b[\wÀ-ÿ ]+?)\s+(?:combina|usa|utiliza|emprega)\s+([\wÀ-ÿ -]+)", "USA"),
        (r"(\b[\wÀ-ÿ ]+?)\s+(?:melhora|reduz|evita)\s+([\wÀ-ÿ -]+)", "MELHORA"),
        (r"(\b[\wÀ-ÿ ]+?)\s+(?:e|está relacionado a)\s+([\wÀ-ÿ -]+)", "RELACIONA"),
    )

    def build(self, documents: Iterable[Document]) -> KnowledgeGraph:
        graph = KnowledgeGraph()
        for document in documents:
            graph.add_node("Document", document.identifier, document.identifier)
            sentences = re.split(r"(?<=[.!?])\s+", document.text)
            for sentence in sentences:
                entities = self._entities(sentence)
                entity_ids = [graph.add_node("Entity", entity, document.identifier)
                              for entity in entities]
                for entity_id in entity_ids:
                    graph.add_relationship(
                        entity_id, "MENTIONED_IN",
                        f"document:{normalize(document.identifier)}", document.identifier
                    )
                for source, target, relation in self._relations(sentence, entities):
                    graph.add_relationship(
                        graph.add_node("Entity", source, document.identifier),
                        relation,
                        graph.add_node("Entity", target, document.identifier),
                        document.identifier,
                    )
        return graph

    @staticmethod
    def _entities(sentence: str) -> list[str]:
        candidates = re.findall(
            r"\b(?:GraphRAG|KAG|Neo4j|LLMs?|Python|Cypher|grafos?|"
            r"conhecimento|recuperação|alucinações|documentos?)\b",
            sentence,
            flags=re.IGNORECASE,
        )
        return list(dict.fromkeys(candidate.strip() for candidate in candidates))

    def _relations(
        self, sentence: str, entities: list[str]
    ) -> list[tuple[str, str, str]]:
        found: list[tuple[str, str, str]] = []
        for pattern, relation in self.RELATION_PATTERNS:
            match = re.search(pattern, sentence, flags=re.IGNORECASE)
            if match:
                source, target = match.groups()
                source = self._nearest_entity(source, entities)
                target = self._nearest_entity(target, entities)
                if source and target and normalize(source) != normalize(target):
                    found.append((source, target, relation))
        return found

    @staticmethod
    def _nearest_entity(value: str, entities: list[str]) -> str:
        normalized = normalize(value)
        return next((entity for entity in entities if normalize(entity) in normalized), "")


class HybridRetriever:
    """Busca lexical combinada com expansão estrutural e contexto auditável."""

    def __init__(self, documents: list[Document], graph: KnowledgeGraph) -> None:
        self.documents = documents
        self.graph = graph

    def search(self, question: str, limit: int = 5, hops: int = 2) -> RetrievalResult:
        if not question.strip():
            raise ValueError("A pergunta não pode ser vazia.")
        query_terms = terms(question)
        ranked = sorted(
            self.documents,
            key=lambda doc: (len(doc.terms & query_terms), doc.identifier),
            reverse=True,
        )
        selected_documents = [doc for doc in ranked if doc.terms & query_terms][:limit]
        seeds = [
            node.identifier for node in self.graph.nodes.values()
            if node.label == "Entity" and terms(node.name) & query_terms
        ]
        selected_nodes = set(seeds)
        for seed in seeds[:limit]:
            selected_nodes.update(self.graph.neighbors(seed, hops))
        paths = self.graph.paths(selected_nodes)
        evidence = len(selected_documents) + len(paths)
        confidence = min(0.98, 0.25 + evidence / 20)
        if not selected_documents:
            answer = "Nenhum contexto fundamentado foi encontrado na base."
        else:
            answer = (
                f"Encontrei {len(selected_documents)} documento(s) e {len(paths)} "
                "caminho(s) verificável(is). A resposta está limitada ao contexto "
                "recuperado; valide os caminhos antes de afirmar novos fatos."
            )
        return RetrievalResult(
            answer,
            [doc.identifier for doc in selected_documents],
            sorted(selected_nodes),
            paths[:20],
            self.graph.cypher_context(seeds[:limit], hops),
            round(confidence, 2),
        )


def load_documents(path: Path) -> list[Document]:
    if path.is_file() and path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        records = data if isinstance(data, list) else data.get("documents", [])
        return [Document(str(item["id"]), str(item["text"])) for item in records]
    files = [path] if path.is_file() else sorted(path.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo .txt encontrado em {path}")
    return [Document(file.stem, file.read_text(encoding="utf-8")) for file in files]


def demo_documents() -> list[Document]:
    return [
        Document("fundamentos", "Grafos representam entidades e relações entre conceitos."),
        Document("graphrag", "GraphRAG combina recuperação de documentos e conhecimento estruturado."),
        Document("kag", "KAG utiliza um grafo de conhecimento para melhorar respostas."),
        Document("confiabilidade", "Neo4j usa Cypher e reduz alucinações com contexto verificável."),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="GraphRAG/KAG educacional local.")
    parser.add_argument("question", nargs="?", help="Pergunta em linguagem natural.")
    parser.add_argument("--demo", action="store_true", help="Usa os documentos de demonstração.")
    parser.add_argument("--file", type=Path, help="Arquivo JSON ou arquivo/pasta com .txt.")
    parser.add_argument("--cypher", type=Path, help="Salva o schema Neo4j e a consulta.")
    parser.add_argument("--hops", type=int, default=2, help="Quantidade de saltos (mínimo 2).")
    args = parser.parse_args()
    if args.hops < 2:
        parser.error("--hops deve ser no mínimo 2 para atender ao multi-hop.")
    documents = demo_documents() if args.demo or not args.file else load_documents(args.file)
    graph = TextToGraph().build(documents)
    question = args.question or input("Pergunta: ")
    result = HybridRetriever(documents, graph).search(question, hops=args.hops)
    if args.cypher:
        export = (
            graph.schema_cypher()
            + "\n"
            + graph.data_cypher()
            + "\n// Consulta de contexto gerada para esta pergunta\n"
            + result.cypher
            + "\n"
        )
        args.cypher.write_text(export, encoding="utf-8")
        print(f"Schema, dados e consulta Neo4j salvos em: {args.cypher}")
    print(f"\n{result.answer}\nConfiança: {result.confidence:.2f}")
    print(f"Documentos: {', '.join(result.documents) or 'nenhum'}")
    print(f"Nós consultados: {len(result.nodes)}")
    print("Caminhos do grafo:")
    for path in result.paths:
        print(f"  {path}")
    print(f"Cypher de contexto: {result.cypher}")


if __name__ == "__main__":
    main()
