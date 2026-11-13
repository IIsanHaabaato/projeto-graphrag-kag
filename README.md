# Projeto GraphRAG/KAG

Projeto educacional baseado no documento do projeto final da disciplina
Tendências em Ciência da Computação. `PROJETO RICARDO.py` é o protótipo legado:
ele demonstra extração determinística, busca lexical e geração de Cypher, mas
não conecta a um Neo4j nem executa embeddings ou busca vetorial. A evolução
modular fica separada em `src/`:

1. **Ingestão e curadoria:** extração página a página e carregamento apenas de
   fontes aprovadas.
2. **Preparação Text-to-Graph:** contrato NER/RE, cliente Ollama local,
   canonicalização conservadora e persistência Neo4j opcional.
3. **Avaliação:** métricas exatas para comparar resultados com anotações humanas.

Busca GraphRAG híbrida, embeddings, navegação multi-hop, algoritmos de
comunidade/ranking e interface ainda não fazem parte desta etapa.

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

## Ingestão de PDFs

O módulo `src/ingestion/pdf_ingest.py` extrai o texto página a página dos PDFs
em `data/raw/NASA/`, sem executar OCR. Cada linha de
`data/processed/nasa_pages.jsonl` representa uma página e contém o identificador
estável do documento (derivado da pasta e do caminho relativo), nome do arquivo,
caminho de origem, número da página, texto, quantidade de caracteres,
metadados existentes no PDF e estado da extração. Páginas sem texto geram um
aviso; PDFs inválidos também geram um registro de erro, em vez de serem
descartados.

Instale a dependência e execute a ingestão:

```powershell
python -m pip install -r requirements.txt
python -m src.ingestion.pdf_ingest
```

Por padrão, a execução lê `data/raw/NASA/`, sobrescreve o JSONL processado e
atualiza `sources.csv` por `document_id`, sem criar linhas duplicadas. Os
caminhos podem ser alterados com `--input-dir`, `--output` e `--sources`.
Documentos com falhas ficam registrados e fazem o comando terminar com código
de saída diferente de zero, após salvar os resultados disponíveis.

Para validar a ingestão junto com os testes de regressão existentes:

```powershell
python -m unittest discover -v
```

## Curadoria e carregamento de páginas aprovadas

`sources.csv` separa o resultado técnico (`ingestion_status`) da decisão
editorial (`include_in_pipeline`) e registra a justificativa em `review_reason`.
Somente o valor explícito `true`, combinado com `ingestion_status=processed`,
autoriza um documento. Campo de aprovação vazio não autoriza uso.

O módulo `src/ingestion/approved_documents.py` fornece
`iter_approved_pages(manifest_path, jsonl_path)`. Ele verifica IDs duplicados no
manifesto e IDs do JSONL ausentes no manifesto antes de disponibilizar páginas.
Retorna apenas páginas de documentos aprovados e processados que tenham texto;
páginas vazias, documentos rejeitados e documentos com erro são ignorados. Os
PDFs originais e o JSONL completo permanecem preservados.

## Preparação Text-to-Graph (NER + RE)

O fluxo novo é implementado em módulos separados:

- `src/knowledge/ontology.py`: tipos de entidade, predicados e combinações de
  endpoints permitidas.
- `src/extraction/contracts.py`: schema JSON e validação de menções, relações,
  offsets e evidências contra o texto literal da página.
- `src/extraction/ollama_client.py`: chamada HTTP local ao endpoint Ollama,
  com uma tentativa de correção para resposta inválida. Respostas inválidas
  são registradas em `EXTRACTION_ERROR_LOG` e não chegam ao Neo4j.
- `src/knowledge/canonicalization.py`: resolve apenas aliases explícitos;
  nomes desconhecidos ou ambíguos ficam `pending_review`, sem fusão aproximada.
- `src/knowledge/neo4j_repository.py`: persistência transacional opcional de
  páginas, entidades, claims e evidências.
- `src/extraction/pipeline.py`: percorre exclusivamente
  `iter_approved_pages()` e grava resultados JSONL em estado pendente de revisão.

As relações extraídas são representadas como nós `Claim` com propriedade
`predicate` e conexões `SUBJECT`/`OBJECT`. Evidências distintas são nós
`Evidence` separados, ligados à página e ao claim; assim, uma citação não é
sobrescrita por outra. Entidades e claims extraídos permanecem pendentes de
revisão humana, e a modalidade temporal distingue `planned`, `expected`,
`occurred`, `historical` e `unknown`.

O modelo LPG implementado usa estes nós e conexões de proveniência:

- `Entity` mais um label de tipo: `Mission`, `Organization`, `Spacecraft`,
  `LaunchVehicle`, `Equipment`, `CelestialBody`, `Location`, `Facility`,
  `Event`, `TestCampaign`, `TestTechnique` ou `EngineeringFinding`. As
  propriedades comuns são `id`, `entity_type`, `canonical_name`,
  `normalized_name`, `aliases`, `resolution_status` e
  `canonical_name_suggestion`.
- `Document`: `document_id`, `filename`, `source_path` e `pdf_metadata_json`.
- `Page`: `page_id`, `document_id`, `page_number`, `text`,
  `character_count` e `extractor_version`.
- `Evidence`: `evidence_id`, tipo, documento/página, citação literal, offsets e
  versão do extrator.
- `Claim`: `claim_id`, `predicate`, `temporal_status`, `qualifiers_json`,
  `review_status`, documento/página e versão do extrator.
- Proveniência: `Document-[:HAS_PAGE]->Page`,
  `Page-[:CONTAINS_EVIDENCE]->Evidence`,
  `Evidence-[:MENTIONS]->Entity`,
  `Evidence-[:SUPPORTS]->Claim`,
  `Claim-[:SUBJECT]->Entity` e `Claim-[:OBJECT]->Entity`.

Os predicados de domínio permitidos são `HAS_EVENT`, `TESTS`, `HAS_COMPONENT`,
`CARRIES`, `BUILT_BY`, `DEVELOPED_BY`, `PROVIDED_BY`,
`USES_LAUNCH_VEHICLE`, `LAUNCHED_FROM`, `LANDS_AT`, `TARGETS`, `LOCATED_IN`,
`CONDUCTED_AT`, `TESTED_ARTICLE`, `USES_TECHNIQUE`, `COMMUNICATES_WITH`,
`RELAYS_DATA_TO`, `HAS_FINDING` e `ABOUT_VEHICLE`. Seus endpoints tipados são
validados por `src/knowledge/ontology.py`. Datas e detalhes temporais são
mantidos como propriedades/qualificadores do evento ou claim quando extraídos;
não se cria automaticamente um nó de data.

### Ollama local

Instale e inicie o Ollama separadamente, baixe localmente o modelo escolhido
pela equipe e configure a tag desse modelo no ambiente. Nenhum modelo é
escolhido, baixado ou iniciado pelo projeto. A saída é solicitada em JSON
Schema; a validação local também verifica citações literais, tipos permitidos,
endpoints declarados e offsets na página. Documento e página são anexados pelo
código e não são aceitos como metadados fornecidos pelo modelo.

No PowerShell, defina os valores para sua instalação (o arquivo `.env.example`
é apenas um modelo e não é carregado automaticamente):

```powershell
$env:OLLAMA_BASE_URL = "http://localhost:11434"
$env:OLLAMA_MODEL = "<tag-local-escolhida>"
python -m src.extraction.pipeline
```

O comando processa somente páginas aprovadas e grava
`data/processed/text_to_graph.jsonl`. O serviço Ollama precisa estar ativo no
endpoint configurado. Erros de transporte são apresentados ao usuário; uma
resposta de modelo inválida após a única correção fica registrada no JSONL de
erros e o pipeline informa a página como inválida.

### Neo4j 5 e índices

`neo4j/constraints.cypher` e `neo4j/indexes.cypher` contêm as constraints de
unicidade e os índices RANGE, Fulltext e Vector para Neo4j 5. O termo B-tree é
legado: no Neo4j 5 o tipo equivalente é RANGE, e os arquivos não declaram um
índice chamado `BTREE`. A dimensão do índice vetorial não é fixada antes da
escolha de um modelo de embeddings.

Para persistir, configure `NEO4J_URI`, `NEO4J_USERNAME` e `NEO4J_PASSWORD` como
variáveis de ambiente e execute:

```powershell
$env:NEO4J_URI = "bolt://localhost:7687"
$env:NEO4J_USERNAME = "<usuario>"
$env:NEO4J_PASSWORD = "<senha>"
python -m src.extraction.pipeline --persist --apply-schema
```

`--apply-schema` executa constraints e índices no banco antes de processar as
páginas. Sem `--persist`, o driver Neo4j não é carregado e não há conexão ao
banco. Não coloque credenciais no código nem as versione.

### Avaliação manual

`data/evaluation/annotation_template.json` é somente um template vazio
identificado como `pending_human_annotation`; não contém rótulos gold. Para
cada página anotada, preencha os IDs e offsets conferidos manualmente, tipos,
IDs canônicos revisados, relações, evidência literal e modalidade temporal.
As funções em `src/evaluation/metrics.py` calculam precisão, recall e F1 por
correspondência exata de spans/tipos e relações, além da acurácia de
canonicalização, evidência e modalidade temporal. A comparação exige que gold
e predição usem os mesmos offsets de página.

### Testes

Instale as dependências Python declaradas e execute a suíte completa:

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -v
```

Os testes Text-to-Graph usam respostas e driver simulados; não iniciam Ollama,
não conectam ao Neo4j e não alteram os documentos aprovados.
