# Email Agent Framework - Multi-service Dockerfile
# Serves: Agent API, LangGraph Worker, and LangGraph Server

# Build stage
FROM python:3.11-slim AS builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpoppler-cpp-dev \
    libjpeg-dev \
    zlib1g-dev \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies only (not the project itself, for layer caching)
RUN uv sync --frozen --no-dev --no-install-project


# Runtime stage
FROM python:3.11-slim AS runtime

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpoppler-cpp-dev \
    libjpeg-dev \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Set up PATH and PYTHONPATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
# Prevent Python from writing bytecode (.pyc) and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy application code
COPY email_agent/ ./email_agent/

# Copy entrypoint scripts
COPY docker/scripts/ ./scripts/
RUN chmod +x ./scripts/*.sh

# Expose ports for Agent API (8000) and LangGraph Server (2024)
EXPOSE 8000 2024

# Health check for Agent API (default service)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command runs the Agent API
CMD ["uvicorn", "email_agent.agent_api.server:app", "--host", "0.0.0.0", "--port", "8000"]
