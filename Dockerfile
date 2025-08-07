# Dockerfile for development with hot reload
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Set working directory
WORKDIR /app

# Copy dependency definitions
COPY pyproject.toml poetry.lock ./

# Configure and install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Download required NLTK data for tokenization
RUN python3 - <<EOF
import nltk
nltk.download('punkt_tab')
EOF

# Copy application code
COPY . .

# Expose port for Uvicorn
EXPOSE 8002

# Default command with auto-reload for development
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]
