from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.ingestion.approved_documents import iter_approved_pages


class ApprovedDocumentsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.manifest_path = self.root / "sources.csv"
        self.jsonl_path = self.root / "pages.jsonl"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_manifest(self, rows: list[dict[str, str]]) -> None:
        fields = [
            "document_id",
            "ingestion_status",
            "include_in_pipeline",
            "review_reason",
        ]
        with self.manifest_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def _write_jsonl(self, records: list[dict[str, object]]) -> None:
        self.jsonl_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    @staticmethod
    def _manifest_row(
        document_id: str,
        *,
        ingestion_status: str = "processed",
        include_in_pipeline: str = "true",
    ) -> dict[str, str]:
        return {
            "document_id": document_id,
            "ingestion_status": ingestion_status,
            "include_in_pipeline": include_in_pipeline,
            "review_reason": "Reviewed source",
        }

    @staticmethod
    def _page_record(
        document_id: str,
        *,
        text: str = "Extracted page text.",
        status: str = "processed",
        page_number: int = 1,
    ) -> dict[str, object]:
        return {
            "record_type": "page",
            "document_id": document_id,
            "filename": f"{document_id}.pdf",
            "source_path": f"NASA/{document_id}.pdf",
            "page_number": page_number,
            "text": text,
            "character_count": len(text),
            "pdf_metadata": {},
            "status": status,
            "error": None,
        }

    def test_loads_approved_processed_pages_with_text(self) -> None:
        self._write_manifest([self._manifest_row("approved")])
        self._write_jsonl([self._page_record("approved")])

        pages = list(iter_approved_pages(self.manifest_path, self.jsonl_path))

        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0].document_id, "approved")
        self.assertEqual(pages[0].text, "Extracted page text.")

    def test_rejected_documents_are_never_loaded(self) -> None:
        self._write_manifest(
            [
                self._manifest_row("approved"),
                self._manifest_row("rejected", include_in_pipeline="false"),
            ]
        )
        self._write_jsonl(
            [self._page_record("approved"), self._page_record("rejected")]
        )

        pages = list(iter_approved_pages(self.manifest_path, self.jsonl_path))

        self.assertEqual([page.document_id for page in pages], ["approved"])

    def test_pages_without_text_are_ignored(self) -> None:
        self._write_manifest([self._manifest_row("approved")])
        self._write_jsonl(
            [
                self._page_record("approved", text="", status="empty_text"),
                self._page_record("approved", text="   "),
                self._page_record("approved", page_number=3),
            ]
        )

        pages = list(iter_approved_pages(self.manifest_path, self.jsonl_path))

        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0].page_number, 3)

    def test_jsonl_ids_missing_from_manifest_raise_before_yielding_pages(self) -> None:
        self._write_manifest([self._manifest_row("approved")])
        self._write_jsonl(
            [
                self._page_record("approved"),
                self._page_record("not-in-manifest"),
            ]
        )

        pages = iter_approved_pages(self.manifest_path, self.jsonl_path)
        with self.assertRaisesRegex(ValueError, "ausente no manifesto"):
            next(pages)

    def test_duplicate_manifest_ids_raise(self) -> None:
        row = self._manifest_row("duplicate")
        self._write_manifest([row, row])
        self._write_jsonl([])

        with self.assertRaisesRegex(ValueError, "duplicado"):
            list(iter_approved_pages(self.manifest_path, self.jsonl_path))

    def test_manifest_rows_without_valid_ids_raise(self) -> None:
        self._write_manifest([self._manifest_row("   ")])
        self._write_jsonl([])

        with self.assertRaisesRegex(ValueError, "Identificador inválido"):
            list(iter_approved_pages(self.manifest_path, self.jsonl_path))

    def test_documents_with_ingestion_errors_are_not_loaded(self) -> None:
        self._write_manifest(
            [self._manifest_row("failed", ingestion_status="error")]
        )
        self._write_jsonl([self._page_record("failed")])

        pages = list(iter_approved_pages(self.manifest_path, self.jsonl_path))

        self.assertEqual(pages, [])

    def test_missing_explicit_approval_does_not_authorize_documents(self) -> None:
        self._write_manifest([self._manifest_row("pending", include_in_pipeline="")])
        self._write_jsonl([self._page_record("pending")])

        pages = list(iter_approved_pages(self.manifest_path, self.jsonl_path))

        self.assertEqual(pages, [])


if __name__ == "__main__":
    unittest.main()
