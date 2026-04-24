FROM python:3.11-slim

WORKDIR /app

# INSTALAR UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# DEPENDÊNCIAS
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache

# COPIAR CÓDIGO
COPY src/       ./src/
COPY api/       ./api/
COPY app/       ./app/
COPY models/    ./models/
COPY config.yaml .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]