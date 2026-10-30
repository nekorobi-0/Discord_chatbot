# syntax=docker/dockerfile:1
FROM python:3.13-slim AS download
RUN pip install --no-cache-dir 'huggingface-hub>=1,<2' 'python-dotenv>=1.1,<2'
COPY deploy/embedding/download_model.py /download_model.py
# .env is mounted only for this step; neither it nor HF credentials enter a layer.
RUN --mount=type=secret,id=hf_env,required=true \
    --mount=type=tmpfs,target=/root/.cache \
    python /download_model.py

FROM ghcr.io/ggml-org/llama.cpp:server@sha256:b32890309c1a33a534a01b0b461e953dc2c5a6bac20213abcf48e22847a19c85
COPY --from=download --chmod=0444 /models/embeddinggemma-2.gguf /models/embeddinggemma-2.gguf
USER 10001:10001
ENTRYPOINT ["/app/llama-server"]
CMD ["--model", "/models/embeddinggemma-2.gguf", "--alias", "embeddinggemma-2", "--embeddings", "--host", "127.0.0.1", "--port", "11434", "--n-gpu-layers", "0", "--threads", "8", "--ctx-size", "8192"]
