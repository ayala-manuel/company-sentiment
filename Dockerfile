FROM python:3.11-slim
WORKDIR /app

# Copiamos y instalamos dependencias
COPY pyproject.toml uv.lock  ./
RUN pip install uv --no-cache-dir --upgrade pip \
    && uv sync --locked


# Copiamos el resto de la app
COPY app/ ./app/
COPY model/ ./model/
COPY utils/ ./utils/
COPY model.pt ./model.pt

EXPOSE 8002
ENTRYPOINT ["uv", "run", "--", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
