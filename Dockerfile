FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /uvx /bin/

COPY pyproject.toml uv.lock README.md ./
COPY rocket_landing ./rocket_landing
COPY envs ./envs
COPY agents ./agents
COPY controllers ./controllers
COPY configs ./configs
COPY scripts ./scripts
RUN uv sync --locked --no-dev --no-editable

RUN useradd --create-home --uid 10001 researcher
USER researcher

CMD ["python", "scripts/smoke.py"]
