FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 DATA_DIR=/app/data TIKTOKEN_CACHE_DIR=/app/tokenizer-cache
WORKDIR /app
RUN useradd --create-home --uid 10001 bot
COPY pyproject.toml README.md ./
COPY ai_kun ./ai_kun
RUN pip install --no-cache-dir . \
    && python -c "from ai_kun.text_files import file_encoding; file_encoding()" \
    && mkdir -p /app/data && chown bot:bot /app/data
USER bot
CMD ["python", "-m", "ai_kun"]
