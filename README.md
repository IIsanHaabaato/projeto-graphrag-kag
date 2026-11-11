# Projeto GraphRAG/KAG

Protótipo educacional baseado no documento do projeto final da disciplina
Tendências em Ciência da Computação. O script implementa uma versão local e
determinística dos quatro pilares pedidos:

1. **Text-to-Graph:** extração simples de entidades e relações com
   canonicalização de nomes.
2. **LPG/Neo4j:** modelo de nós e relações local, com geração de constraints,
   índice BTree, Fulltext e modelo de Vector Index em Cypher.
3. **Recuperação híbrida:** ranking lexical, expansão estrutural e travessia
   multi-hop (mínimo de dois saltos).
4. **Explicabilidade:** documentos, nós, caminhos, consulta Cypher e confiança.

## Executar

```powershell
python "PROJETO RICARDO.py" --demo "Como GraphRAG reduz alucinações?"
```

Para ingerir textos, coloque arquivos `.txt` em uma pasta ou use um JSON no
formato `[{"id": "doc1", "text": "..."}]`:

```powershell
python "PROJETO RICARDO.py" --file .\documentos --cypher .\schema.cypher "O que KAG utiliza?"
```

O arquivo indicado em `--cypher` inclui o schema, os nós e relacionamentos
extraídos e a consulta de contexto da pergunta. Para executar os testes:

```powershell
python -m unittest -v
```

O extrator é propositalmente transparente para fins acadêmicos. Para a entrega
final, ele deve ser substituído ou comparado com um pipeline NER/RE baseado no
domínio escolhido pelo grupo e validado com métricas e testes.
