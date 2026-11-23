# syntax=docker/dockerfile:1.7
FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.9.4 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN uv venv /opt/venv

# RTX 50-series (Blackwell, sm_120) needs CUDA 12.8 wheels; use .../whl/cpu for CPU-only images.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install torch==2.7.1 --index-url ${TORCH_INDEX_URL}

WORKDIR /workspace
COPY requirements.txt pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements.txt

COPY . .
RUN uv pip install -e . --no-deps

CMD ["bash"]
