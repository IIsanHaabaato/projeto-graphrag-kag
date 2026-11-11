"""Testes de regressão do protótipo local GraphRAG/KAG."""

from __future__ import annotations

import unittest
import sys

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("PROJETO RICARDO.py")
SPEC = spec_from_file_location("projeto_ricardo", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Não foi possível carregar {MODULE_PATH}")
projeto = module_from_spec(SPEC)
sys.modules[SPEC.name] = projeto
SPEC.loader.exec_module(projeto)


class GraphRAGTests(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = [
            projeto.Document(
                "doc1", "GraphRAG combina recuperação e conhecimento estruturado."
            ),
            projeto.Document(
                "doc2", "KAG utiliza um grafo de conhecimento para melhorar respostas."
            ),
        ]
        self.graph = projeto.TextToGraph().build(self.documents)

    def test_canonicaliza_entidade_repetida(self) -> None:
        entity_ids = [
            node_id
            for node_id, node in self.graph.nodes.items()
            if node.label == "Entity" and projeto.normalize(node.name) == "conhecimento"
        ]
        self.assertEqual(entity_ids, ["entity:conhecimento"])

    def test_retrieval_exibe_multi_hop_e_cypher(self) -> None:
        result = projeto.HybridRetriever(self.documents, self.graph).search(
            "Como KAG utiliza conhecimento?"
        )
        self.assertIn("doc2", result.documents)
        self.assertGreaterEqual(len(result.nodes), 2)
        self.assertIn("[*1..2]", result.cypher)
        self.assertGreaterEqual(result.confidence, 0.25)

    def test_exporta_dados_e_relacionamentos_em_cypher(self) -> None:
        exported = self.graph.data_cypher()
        self.assertIn("MERGE (n:Entity", exported)
        self.assertIn("MENTIONED_IN", exported)


if __name__ == "__main__":
    unittest.main()
