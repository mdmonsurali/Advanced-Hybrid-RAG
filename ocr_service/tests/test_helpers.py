"""Small helpers: page counts, scan-type probe, OCR options, stored metadata."""

from __future__ import annotations

import json
import uuid
import zipfile

import docling_engine
import main
import pytest
import storage


def make_docx(path, app_xml: str | None) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("word/document.xml", "<w:document/>")
        if app_xml is not None:
            zf.writestr("docProps/app.xml", app_xml)


def test_docx_page_count_from_word_metadata(tmp_path):
    f = tmp_path / "a.docx"
    make_docx(f, '<Properties xmlns:ep="x"><ep:Pages>7</ep:Pages></Properties>')
    assert main._docx_page_count(f) == 7


@pytest.mark.parametrize("app_xml", [None, "<Properties><Pages>0</Pages></Properties>"])
def test_docx_page_count_falls_back_to_one(tmp_path, app_xml):
    f = tmp_path / "a.docx"
    make_docx(f, app_xml)
    assert main._docx_page_count(f) == 1


def test_docx_page_count_survives_a_broken_file(tmp_path):
    f = tmp_path / "broken.docx"
    f.write_bytes(b"not a zip")
    assert main._docx_page_count(f) == 1


def test_scan_type_by_extension(tmp_path):
    for name, expected in [("a.png", "scanned"), ("a.JPG", "scanned"), ("a.docx", "native")]:
        f = tmp_path / name
        f.write_bytes(b"")
        assert main._detect_scan_type(f) == expected
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.7 truncated")
    assert main._detect_scan_type(broken) == "scanned"  # unreadable PDFs are OCR'd


def test_pipeline_options_cpu(monkeypatch):
    monkeypatch.setattr(docling_engine, "DOCLING_OCR_LANG", ["en", "de"])
    opts = docling_engine._pipeline_options("cpu")
    assert opts.do_ocr is True
    assert opts.ocr_options.lang == ["en", "de"]


def test_unknown_ocr_engine_is_rejected(monkeypatch):
    monkeypatch.setattr(docling_engine, "DOCLING_OCR_ENGINE", "magic")
    with pytest.raises(ValueError, match="DOCLING_OCR_ENGINE"):
        docling_engine._pipeline_options("cpu")


def test_meta_ids_are_decoded_to_uuids():
    doc_id, owner = uuid.uuid4(), uuid.uuid4()
    meta = storage._decode_meta(json.dumps({"id": str(doc_id), "owner_id": str(owner)}).encode())
    assert meta["id"] == doc_id
    assert meta["owner_id"] == owner
