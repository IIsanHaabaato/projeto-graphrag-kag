// Neo4j 5 uses RANGE indexes in place of the deprecated B-tree index type.
CREATE RANGE INDEX entity_canonical_name IF NOT EXISTS
FOR (n:Entity) ON (n.canonical_name);

CREATE RANGE INDEX page_source_location IF NOT EXISTS
FOR (n:Page) ON (n.document_id, n.page_number);

CREATE RANGE INDEX claim_predicate IF NOT EXISTS
FOR (n:Claim) ON (n.predicate);

CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
FOR (n:Entity) ON EACH [n.canonical_name, n.aliases];

CREATE FULLTEXT INDEX page_text_fulltext IF NOT EXISTS
FOR (n:Page) ON EACH [n.text];

// Dimensions are intentionally unspecified until an embedding model is selected.
CREATE VECTOR INDEX entity_embedding IF NOT EXISTS
FOR (n:Entity) ON (n.embedding)
OPTIONS {indexConfig: {`vector.similarity_function`: 'cosine'}};
