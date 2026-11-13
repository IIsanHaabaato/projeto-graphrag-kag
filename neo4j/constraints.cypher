// Neo4j 5 schema constraints; Entity is the shared identity label.
CREATE CONSTRAINT entity_id IF NOT EXISTS
FOR (n:Entity) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT document_id IF NOT EXISTS
FOR (n:Document) REQUIRE n.document_id IS UNIQUE;

CREATE CONSTRAINT page_id IF NOT EXISTS
FOR (n:Page) REQUIRE n.page_id IS UNIQUE;

CREATE CONSTRAINT evidence_id IF NOT EXISTS
FOR (n:Evidence) REQUIRE n.evidence_id IS UNIQUE;

CREATE CONSTRAINT claim_id IF NOT EXISTS
FOR (n:Claim) REQUIRE n.claim_id IS UNIQUE;
