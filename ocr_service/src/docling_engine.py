"""Docling document conversion + exporters.

Wraps a single cached `docling.DocumentConverter` running the standard PDF
pipeline (layout model + TableFormer + OCR for bitmap regions; the PDF text
layer is used wherever it exists). Images go through the same pipeline; DOCX
is parsed natively by Docling's Word backend.

With GPU_SWAP=true, PDFs and images are converted on CUDA by the GPU worker
process (gpu_worker.py) under an `ocr` lease from the orchestrator's GPU swap;
the in-process converter below stays on CPU for DOCX and as the fallback.

Outputs per document:
    markdown  `export_to_markdown`, pictures referenced as images/picture-N.png
    json      `export_to_dict` (DoclingDocument schema)
    docx      `export_to_html` (pictures embedded) -> LibreOffice -> DOCX
    pictures  (filename, PIL image) pairs for the MinIO upload
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    OcrAutoOptions,
    PdfPipelineOptions,
    RapidOcrOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import (
    DocumentConverter,
    ImageFormatOption,
    PdfFormatOption,
    WordFormatOption,
)
from docling_core.types.doc import DoclingDocument, ImageRef, ImageRefMode, PictureItem
from PIL import Image

log = logging.getLogger("ocr_service")


def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


INCLUDE_PICTURES = _env_bool("INCLUDE_PICTURES", "true")
DOCLING_DEVICE = os.getenv("DOCLING_DEVICE", "cpu")
DOCLING_NUM_THREADS = int(os.getenv("DOCLING_NUM_THREADS", "4"))
GPU_SWAP = _env_bool("GPU_SWAP", "false")
# RapidOCR backend inside the GPU worker; `torch` runs on CUDA (models are
# baked into the image next to the onnx ones).
DOCLING_GPU_OCR_BACKEND = os.getenv("DOCLING_GPU_OCR_BACKEND", "torch").strip().lower()
GPU_EXTS = {".pdf", ".png", ".jpg", ".jpeg"}
# auto | rapidocr | easyocr (easyocr needs the docling[easyocr] extra).
DOCLING_OCR_ENGINE = os.getenv("DOCLING_OCR_ENGINE", "rapidocr").strip().lower()
# Comma-separated OCR languages; empty keeps the engine default (RapidOCR: ch,
# which also covers Latin text).
DOCLING_OCR_LANG = [l.strip() for l in os.getenv("DOCLING_OCR_LANG", "").split(",") if l.strip()]
DOCLING_TABLE_MODE = os.getenv("DOCLING_TABLE_MODE", "accurate").strip().lower()
DOCLING_IMAGES_SCALE = float(os.getenv("DOCLING_IMAGES_SCALE", "2.0"))
DOCLING_DO_FORMULA_ENRICHMENT = _env_bool("DOCLING_DO_FORMULA_ENRICHMENT", "false")
DOCLING_DO_CODE_ENRICHMENT = _env_bool("DOCLING_DO_CODE_ENRICHMENT", "false")
DOCLING_ARTIFACTS_PATH = os.getenv("DOCLING_ARTIFACTS_PATH") or None
# Documents converted at once. The models share one GPU, so 1 is the default.
OCR_CONCURRENCY = int(os.getenv("OCR_CONCURRENCY", "1"))

_OCR_OPTION_CLASSES = {
    "auto": OcrAutoOptions,
    "rapidocr": RapidOcrOptions,
    "easyocr": EasyOcrOptions,
}

_converter: DocumentConverter | None = None
_sem: asyncio.Semaphore | None = None


def _pipeline_options(device: str = DOCLING_DEVICE) -> PdfPipelineOptions:
    ocr_cls = _OCR_OPTION_CLASSES.get(DOCLING_OCR_ENGINE)
    if ocr_cls is None:
        raise ValueError(f"Unsupported DOCLING_OCR_ENGINE: {DOCLING_OCR_ENGINE}")
    ocr_kwargs: dict[str, Any] = {"lang": DOCLING_OCR_LANG} if DOCLING_OCR_LANG else {}
    if device.startswith("cuda") and ocr_cls is RapidOcrOptions:
        ocr_kwargs["backend"] = DOCLING_GPU_OCR_BACKEND

    return PdfPipelineOptions(
        artifacts_path=DOCLING_ARTIFACTS_PATH,
        accelerator_options=AcceleratorOptions(device=device, num_threads=DOCLING_NUM_THREADS),
        do_ocr=True,
        ocr_options=ocr_cls(**ocr_kwargs),
        do_table_structure=True,
        table_structure_options=TableStructureOptions(
            mode=TableFormerMode(DOCLING_TABLE_MODE),
            do_cell_matching=True,
        ),
        do_formula_enrichment=DOCLING_DO_FORMULA_ENRICHMENT,
        do_code_enrichment=DOCLING_DO_CODE_ENRICHMENT,
        generate_picture_images=INCLUDE_PICTURES,
        images_scale=DOCLING_IMAGES_SCALE,
    )


def build_converter(device: str = DOCLING_DEVICE) -> DocumentConverter:
    """A converter with the PDF pipeline models loaded (blocking)."""
    opts = _pipeline_options(device)
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.IMAGE, InputFormat.DOCX],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=opts),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=opts),
            InputFormat.DOCX: WordFormatOption(),
        },
    )
    converter.initialize_pipeline(InputFormat.PDF)
    return converter


def init_converter() -> None:
    """Build the converter and load the PDF pipeline models (blocking; call
    once at startup so the first request doesn't pay the model load)."""
    global _converter, _sem
    _converter = build_converter()
    _sem = asyncio.Semaphore(max(1, OCR_CONCURRENCY))
    log.info(
        "[docling] converter ready (device=%s, ocr=%s, table=%s)",
        DOCLING_DEVICE,
        DOCLING_OCR_ENGINE,
        DOCLING_TABLE_MODE,
    )


def is_ready() -> bool:
    return _converter is not None


def convert_with(converter: DocumentConverter, path: Path) -> DoclingDocument:
    result = converter.convert(str(path), raises_on_error=True)
    if result.status == ConversionStatus.PARTIAL_SUCCESS:
        log.warning(
            "[docling] %s converted partially: %s",
            path.name,
            [e.error_message for e in result.errors],
        )
    return result.document


def _convert_sync(path: Path) -> DoclingDocument:
    assert _converter is not None, "init_converter() not called"
    return convert_with(_converter, path)


async def convert(path: Path) -> DoclingDocument:
    """Convert one file off the event loop, bounded by OCR_CONCURRENCY.
    PDFs/images go to the GPU worker when GPU_SWAP is on; if the GPU path is
    unavailable (orchestrator down, worker crashed) the CPU converter runs."""
    assert _sem is not None, "init_converter() not called"
    async with _sem:
        if GPU_SWAP and path.suffix.lower() in GPU_EXTS:
            import gpu_worker

            try:
                async with gpu_worker.ocr_lease():
                    return await gpu_worker.convert(path)
            except gpu_worker.GpuUnavailable as e:
                log.warning("[docling] GPU path unavailable for %s, using CPU: %s", path.name, e)
        return await asyncio.to_thread(_convert_sync, path)


# Exporters


def export_html(doc: DoclingDocument) -> str:
    """HTML with pictures embedded as data URIs (input for the DOCX export).
    Call before `reference_pictures`, which swaps the image URIs."""
    return doc.export_to_html(image_mode=ImageRefMode.EMBEDDED)


def reference_pictures(doc: DoclingDocument) -> list[tuple[str, Image.Image]]:
    """Point every picture at images/picture-N.png and return the
    (filename, image) pairs to upload. Mutates `doc` in place so the markdown
    and JSON exports reference the stored crops instead of inline base64."""
    out: list[tuple[str, Image.Image]] = []
    if not INCLUDE_PICTURES:
        return out
    for item, _level in doc.iterate_items(with_groups=False):
        if not isinstance(item, PictureItem):
            continue
        img = item.get_image(doc)
        if img is None:
            continue
        fname = f"picture-{len(out) + 1}.png"
        if item.image is None:
            scale = img.size[0] / item.prov[0].bbox.width if item.prov else 1.0
            item.image = ImageRef.from_pil(image=img, dpi=round(72 * scale))
        item.image.uri = Path(f"images/{fname}")
        out.append((fname, img))
    return out


def export_markdown(doc: DoclingDocument) -> str:
    mode = ImageRefMode.REFERENCED if INCLUDE_PICTURES else ImageRefMode.PLACEHOLDER
    return doc.export_to_markdown(image_mode=mode)


def export_json(doc: DoclingDocument) -> dict:
    return doc.export_to_dict()
