# One image for web, worker and one-off commands (the compose file picks the command).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.10.11 /uv /usr/local/bin/uv

WORKDIR /app

# dependencies first (cached layer), then the project
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY config ./config
COPY data/eval ./data/eval
COPY data/models ./data/models
COPY data/snapshot ./data/snapshot
RUN uv sync --frozen --no-dev \
 && /app/.venv/bin/python -c "from fli.analyze.embed import warm_up; warm_up()"

ENV PATH="/app/.venv/bin:$PATH"

# non-root runtime user
RUN useradd --create-home --uid 10001 fli && mkdir -p /app/data/raw && chown -R fli /app/data
USER fli

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD curl -fsS http://localhost:8000/healthz || exit 1
CMD ["fli", "serve", "--host", "0.0.0.0", "--port", "8000"]
