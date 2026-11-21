FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 DATA_DIR=/app/data TIKTOKEN_CACHE_DIR=/app/tokenizer-cache
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 10001 bot
COPY pyproject.toml README.md ./
COPY ai_kun ./ai_kun
RUN pip install --no-cache-dir '.[voice]' \
    && python -c "from ai_kun.text_files import file_encoding; file_encoding()" \
    && mkdir -p /app/data \
    && chown bot:bot /app/data \
    && apt-get purge -y --auto-remove git
USER bot
CMD ["python", "-m", "ai_kun"]
