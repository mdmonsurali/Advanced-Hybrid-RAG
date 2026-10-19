FROM python:3.12-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch: embeddings and reranking run on CPU so the GPU stays free
# for llm_service.
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY rag_service/requirements.txt /workspace/rag_service/
RUN pip3 install --no-cache-dir -r /workspace/rag_service/requirements.txt

# Pre-bake the embedding, reranker and BM25 models into the image so cold
# start does not depend on a runtime HuggingFace download.
ARG EMBED_MODEL=Qwen/Qwen3-Embedding-0.6B
ARG RERANK_MODEL=Qwen/Qwen3-Reranker-0.6B
ENV HF_HOME=/opt/hf-cache
ENV FASTEMBED_CACHE_PATH=/opt/fastembed-cache
RUN python3 -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
from fastembed import SparseTextEmbedding; \
SentenceTransformer('${EMBED_MODEL}', device='cpu'); \
CrossEncoder('${RERANK_MODEL}', device='cpu'); \
SparseTextEmbedding('Qdrant/bm25'); \
print('models cached')"

COPY rag_service/src/ /workspace/rag_service/src/

ENV PYTHONPATH=/workspace/rag_service/src
ENV HF_HUB_OFFLINE=1
ENV EMBED_MODEL=${EMBED_MODEL}
ENV RERANK_MODEL=${RERANK_MODEL}
ENV RAG_SERVICE_PORT=8003

EXPOSE 8003

HEALTHCHECK --interval=30s --timeout=10s --retries=5 --start-period=180s \
    CMD curl -f http://localhost:8003/health || exit 1

WORKDIR /workspace/rag_service/src

CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${RAG_SERVICE_PORT}"]
