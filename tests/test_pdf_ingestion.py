from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)

from src.ingestion.pdf_ingest import ingest_pdfs


def _create_text_pdf(path: Path) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=144)

    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_reference = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): font_reference}
            )
        }
    )
    content = DecodedStreamObject()
    content.set_data(b"BT /F1 12 Tf 72 72 Td (First page text.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(content)
    writer.add_blank_page(width=300, height=144)
    writer.add_metadata({"/Title": "PDF fixture"})
    with path.open("wb") as pdf_file:
        writer.write(pdf_file)


class PdfIngestionTests(unittest.TestCase):
    def test_extracts_each_page_and_preserves_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "NASA"
            input_dir.mkdir()
            _create_text_pdf(input_dir / "two_pages.pdf")

            with self.assertLogs("src.ingestion.pdf_ingest", level="WARNING") as logs:
                summary = ingest_pdfs(
                    input_dir, root / "pages.jsonl", root / "sources.csv"
                )

            records = [
                json.loads(line)
                for line in (root / "pages.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual([record["page_number"] for record in records], [1, 2])
            self.assertEqual(records[0]["text"], "First page text.")
            self.assertEqual(records[0]["character_count"], len("First page text."))
            self.assertEqual(records[0]["pdf_metadata"]["/Title"], "PDF fixture")
            self.assertEqual(records[1]["text"], "")
            self.assertEqual(records[1]["status"], "empty_text")
            self.assertEqual(summary.documents, 1)
            self.assertEqual(summary.pages_processed, 2)
            self.assertEqual(summary.pages_without_text, 1)
            self.assertEqual(summary.documents_failed, 0)
            self.assertTrue(any("sem texto extraível" in message for message in logs.output))

    def test_invalid_pdf_is_recorded_and_registered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "NASA"
            input_dir.mkdir()
            (input_dir / "invalid.pdf").write_bytes(b"not a PDF")

            with self.assertLogs("src.ingestion.pdf_ingest", level="ERROR"):
                summary = ingest_pdfs(
                    input_dir, root / "pages.jsonl", root / "sources.csv"
                )

            records = [
                json.loads(line)
                for line in (root / "pages.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["record_type"], "document_error")
            self.assertEqual(records[0]["status"], "error")
            self.assertTrue(records[0]["error"]["message"])
            self.assertEqual(summary.documents_failed, 1)

            with (root / "sources.csv").open(
                encoding="utf-8", newline=""
            ) as csv_file:
                sources = list(csv.DictReader(csv_file))
            self.assertEqual(len(sources), 1)
            self.assertEqual(sources[0]["filename"], "invalid.pdf")
            self.assertEqual(sources[0]["ingestion_status"], "error")
            self.assertEqual(sources[0]["include_in_pipeline"], "")

    def test_rerunning_preserves_existing_curation_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "NASA"
            input_dir.mkdir()
            _create_text_pdf(input_dir / "sample.pdf")
            sources_path = root / "sources.csv"
            sources_path.write_text(
                "document_id,ingestion_status,include_in_pipeline,review_reason\n"
                "nasa-sample,processed,false,Reviewed and excluded\n",
                encoding="utf-8",
            )

            ingest_pdfs(input_dir, root / "pages.jsonl", sources_path)

            with sources_path.open(encoding="utf-8", newline="") as csv_file:
                sources = list(csv.DictReader(csv_file))
            self.assertEqual(len(sources), 1)
            self.assertEqual(sources[0]["ingestion_status"], "processed")
            self.assertEqual(sources[0]["include_in_pipeline"], "false")
            self.assertEqual(sources[0]["review_reason"], "Reviewed and excluded")
            self.assertNotIn("status", sources[0])

    def test_rerunning_updates_source_instead_of_duplicating_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "NASA"
            input_dir.mkdir()
            _create_text_pdf(input_dir / "sample.pdf")

            ingest_pdfs(input_dir, root / "pages.jsonl", root / "sources.csv")
            ingest_pdfs(input_dir, root / "pages.jsonl", root / "sources.csv")

            with (root / "sources.csv").open(
                encoding="utf-8", newline=""
            ) as csv_file:
                sources = list(csv.DictReader(csv_file))
            self.assertEqual(len(sources), 1)
            self.assertEqual(sources[0]["document_id"], "nasa-sample")


if __name__ == "__main__":
    unittest.main()
