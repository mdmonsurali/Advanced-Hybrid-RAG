FROM python:3.12-slim

WORKDIR /workspace

# libreoffice: DOCX export (HTML -> DOCX) and the DOCX -> PDF preview.
# libgl1/libglib2.0-0: OpenCV runtime used by Docling's OCR/layout stack.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    procps \
    libreoffice-writer \
    libgl1 \
    libglib2.0-0 \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch: Docling runs on CPU so the GPU stays free for llm_service.
RUN pip3 install --no-cache-dir torch torchvision         --index-url https://download.pytorch.org/whl/cpu

COPY ocr_service/requirements.txt /workspace/ocr_service/
RUN pip3 install --no-cache-dir -r /workspace/ocr_service/requirements.txt

# Pre-bake the Docling models (layout, TableFormer, RapidOCR, ...) into the
# image so cold start does not depend on a runtime HuggingFace download.
ENV DOCLING_ARTIFACTS_PATH=/opt/docling-models
RUN docling-tools models download -o "${DOCLING_ARTIFACTS_PATH}" && \
    echo "docling models cached under ${DOCLING_ARTIFACTS_PATH}"

COPY ocr_service/src/ /workspace/ocr_service/src/

ENV PYTHONPATH=/workspace/ocr_service/src
ENV OCR_SERVICE_PORT=8001
ENV HF_HUB_OFFLINE=1

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --retries=5 --start-period=120s \
    CMD curl -f http://localhost:8001/health || exit 1

WORKDIR /workspace/ocr_service/src

CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${OCR_SERVICE_PORT}"]
