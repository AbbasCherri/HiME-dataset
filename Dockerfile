# syntax=docker/dockerfile:1
#
# Toolchain image for the notodata pipeline. Every command in this repo runs in it
# (through scripts/dc), so the host never needs Python or uv.
#
# Pins (also recorded in DATASET.md):
#   python:3.12-slim-bookworm  sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70
#   ghcr.io/astral-sh/uv:0.13.5
FROM ghcr.io/astral-sh/uv:0.13.5 AS uv

FROM python:3.12-slim-bookworm@sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70

COPY --from=uv /uv /uvx /usr/local/bin/

ARG UID=1000
ARG GID=1000
RUN groupadd --gid "${GID}" app \
    && useradd --create-home --uid "${UID}" --gid "${GID}" --shell /bin/bash app \
    && mkdir -p /opt/venv /opt/project /work \
    && chown -R app:app /opt/venv /opt/project /work

# The environment lives in /opt/venv, outside the /work bind mount, so the mount cannot shadow it.
ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON_PREFERENCE=only-system \
    UV_CACHE_DIR=/home/app/.cache/uv \
    PATH=/opt/venv/bin:${PATH} \
    PYTHONPATH=/work/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NOTODATA_HOME=/work/data

USER app
WORKDIR /opt/project

# Dependency layer: depends on pyproject.toml and uv.lock only.
COPY --chown=app:app pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

WORKDIR /work
