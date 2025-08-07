# 1. Etapa de build con Poetry
FROM python:3.11-slim AS builder
WORKDIR /app

# Instalamos Poetry
RUN pip install uv

# Copiamos archivos de proyecto (excluyendo training/)
COPY pyproject.toml uv.lock ./
COPY app/ ./app/
COPY model/ ./model/
COPY utils/ ./utils/
COPY model.pt ./model.pt

# Instalamos dependencias en un virtual env aislado
RUN uv install

# 2. Etapa de producción
FROM python:3.11-slim AS runner
WORKDIR /app

# Copiamos el entorno creado por Poetry
COPY --from=builder /root/.cache/pypoetry/virtualenvs /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copiamos la aplicación y el modelo
COPY app/ ./app/
COPY model/ ./model/
COPY utils/ ./utils/
COPY model.pt ./model.pt

# Exponemos el puerto de FastAPI
EXPOSE 8002

# Entrypoint: iniciamos con uv
ENTRYPOINT ["uv", "run", "python -m app.main", "--host", "0.0.0.0", "--port", "8002"]
