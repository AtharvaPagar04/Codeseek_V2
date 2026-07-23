# syntax=docker/dockerfile:1.7

###############################
# Builder Stage
###############################
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Packages required only for compiling Python dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    pkg-config \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*
ARG BUILD_PROFILE=cpu

COPY requirements-base.txt .
COPY requirements-${BUILD_PROFILE}.txt ./requirements.txt

# Upgrade pip and build wheels
RUN pip install --upgrade pip wheel setuptools

RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel \
    --wheel-dir=/wheels \
    -r requirements.txt


###############################
# Runtime Stage
###############################
FROM python:3.11-slim-bookworm

LABEL maintainer="CodeSeek"
LABEL stage="production"

ARG BACKEND_PORT=8000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/var/cache/huggingface \
    TIKTOKEN_CACHE_DIR=/var/cache/tiktoken \
    PORT=${BACKEND_PORT}

WORKDIR /app

####################################
# Runtime packages only
####################################
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

####################################
# Create non-root user
####################################
RUN groupadd --gid 1000 codeseek && \
    useradd \
    --uid 1000 \
    --gid codeseek \
    --create-home \
    --shell /usr/sbin/nologin \
    codeseek

####################################
# Install Python dependencies
####################################
COPY --from=builder /wheels /wheels

RUN pip install --no-cache-dir /wheels/* && \
    rm -rf /wheels

####################################
# Copy application
####################################
COPY . /app

####################################
# Runtime directories
####################################
RUN mkdir -p \
    /data/repo_workspace \
    /tmp/rag_ingestion \
    /var/cache/huggingface \
    /var/cache/tiktoken \
    && chown -R codeseek:codeseek \
    /app \
    /data \
    /tmp/rag_ingestion \
    /var/cache/huggingface \
    /var/cache/tiktoken

USER codeseek

####################################
# Expose application port
####################################
EXPOSE ${BACKEND_PORT}

####################################
# Healthcheck
####################################
HEALTHCHECK --interval=30s \
    --timeout=5s \
    --start-period=20s \
    --retries=3 \
    CMD curl -fsS http://127.0.0.1:${PORT}/api/v1/health || exit 1

####################################
# Start application
####################################
CMD uvicorn retrieval.api_service:app --host 0.0.0.0 --port ${PORT}