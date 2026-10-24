FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    OCR_SERVICE_URL=http://ocr_service:8001 \
    # Chainlit reads .chainlit/config.toml, public/ and chainlit.md from here
    # and writes uploaded chat files to $CHAINLIT_APP_ROOT/.files.
    CHAINLIT_APP_ROOT=/app

# upgrade: pick up Debian security fixes newer than the base image.
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY .chainlit /app/.chainlit
COPY public /app/public
COPY chainlit.md /app/chainlit.md
COPY src /app/src

WORKDIR /app/src

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=5 --start-period=30s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
