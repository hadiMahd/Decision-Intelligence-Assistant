FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Copy dependency manifests first to maximize Docker layer caching.
COPY pyproject.toml uv.lock ./

# Create a local virtual environment and install locked dependencies.
RUN uv sync --frozen --no-dev

# Copy application code after dependencies so code changes do not invalidate dependency layers.
COPY . .

# Make sure logs are streamed immediately to container stdout/stderr.
ENV PYTHONUNBUFFERED=1

# Use the project virtual environment by default.
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uv", "run", "main.py"]
