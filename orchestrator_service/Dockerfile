FROM python:3.12-slim

WORKDIR /workspace

# upgrade: pick up Debian security fixes newer than the base image.
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY orchestrator_service/requirements.txt /workspace/orchestrator_service/
RUN pip3 install --no-cache-dir -r /workspace/orchestrator_service/requirements.txt

COPY orchestrator_service/src/ /workspace/orchestrator_service/src/

ENV PYTHONPATH=/workspace/orchestrator_service/src
ENV ORCHESTRATOR_SERVICE_PORT=8004

EXPOSE 8004

HEALTHCHECK --interval=30s --timeout=5s --retries=5 --start-period=20s \
    CMD curl -f http://localhost:8004/health || exit 1

WORKDIR /workspace/orchestrator_service/src

CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${ORCHESTRATOR_SERVICE_PORT}"]
