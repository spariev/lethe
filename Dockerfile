# Lethe Container - Safe Mode
# Access restricted to /workspace only

FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create workspace directory (this will be mounted from host)
RUN mkdir -p /workspace /app

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY config/ ./config/

# Install dependencies
RUN uv sync --frozen

# Create non-root user for safety
RUN useradd -m -s /bin/bash lethe
RUN chown -R lethe:lethe /app /workspace

USER lethe

# Environment
ENV LETHE_WORKSPACE_DIR=/workspace
ENV LETHE_DATA_DIR=/workspace/data
ENV LETHE_CONFIG_DIR=/app/config

# Run
CMD ["uv", "run", "lethe"]
