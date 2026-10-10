FROM lmsysorg/sglang:v0.5.10.post1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY llm_service/requirements.txt /workspace/llm_service/
RUN pip3 install --no-cache-dir -r /workspace/llm_service/requirements.txt

# Pre-bake the Qwen3.5 weights into the image so cold start does not depend on
# a runtime HuggingFace download. The MTP head is skipped (no speculative
# decoding on a 6 GB card).
ARG HF_TOKEN=""
ARG LLM_MODEL=LostGentoo/Qwen3.5-4B-AWQ
ENV HF_HOME=/opt/hf-cache
# huggingface_hub >= 1.0 ships the CLI as `hf`; older images only have
# `huggingface-cli`.
RUN export HF_TOKEN="${HF_TOKEN}"; \
    (hf download "${LLM_MODEL}" --exclude "model_mtp.safetensors" \
     || huggingface-cli download "${LLM_MODEL}" --exclude "model_mtp.safetensors") && \
    echo "llm weights cached under ${HF_HOME}"

COPY llm_service/src/ /workspace/llm_service/src/

ENV PYTHONPATH=/workspace/llm_service/src
ENV HF_HUB_OFFLINE=1
ENV LLM_MODEL=${LLM_MODEL}
ENV LLM_SERVICE_PORT=8005
ENV SGLANG_PORT=30000
ENV SGLANG_URL=http://localhost:30000
# Served-model name SGLang advertises; must match LLM_SERVED_NAME in the client.
ENV LLM_SERVED_NAME=qwen

EXPOSE 8005

HEALTHCHECK --interval=30s --timeout=10s --retries=5 --start-period=300s \
    CMD curl -f http://localhost:8005/health || exit 1

WORKDIR /workspace/llm_service/src

# SGLang knobs (LLM_MEM_FRACTION, LLM_CONTEXT_LEN, LLM_MAX_RUNNING, LLM_EXTRA_ARGS)
# come from .env via docker-compose. The checkpoint is compressed-tensors
# W4A16, which SGLang detects from config.json, so no --quantization flag.
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["\
     python3 -m sglang.launch_server \
         --model-path $LLM_MODEL \
         --served-model-name $LLM_SERVED_NAME \
         --mem-fraction-static $LLM_MEM_FRACTION \
         --context-length $LLM_CONTEXT_LEN \
         --max-running-requests $LLM_MAX_RUNNING \
         --cuda-graph-max-bs $LLM_MAX_RUNNING \
         --reasoning-parser qwen3 \
         --grammar-backend xgrammar \
         --host 127.0.0.1 --port $SGLANG_PORT \
         $LLM_EXTRA_ARGS & \
     SGL_PID=$!; \
     echo \"Waiting for SGLang (PID $SGL_PID) to become ready on :$SGLANG_PORT...\"; \
     for i in $(seq 1 600); do \
         if ! kill -0 $SGL_PID 2>/dev/null; then \
             echo \"SGLang process died before becoming ready\" >&2; \
             exit 1; \
         fi; \
         if curl -sf http://localhost:$SGLANG_PORT/health >/dev/null 2>&1; then \
             echo \"SGLang ready after ${i}s\"; \
             break; \
         fi; \
         sleep 1; \
     done; \
     if ! curl -sf http://localhost:$SGLANG_PORT/health >/dev/null 2>&1; then \
         echo \"SGLang did not become ready within 600s\" >&2; \
         exit 1; \
     fi; \
     exec uvicorn main:app --host 0.0.0.0 --port $LLM_SERVICE_PORT"]
