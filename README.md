# Análisis de Sentimientos Headlines

Este repositorio contiene una **API REST** construida con **FastAPI** para realizar análisis de sentimientos sobre texto, junto a los componentes necesarios para entrenar y procesar modelos de clasificación.

## Estructura del proyecto

```text
.
├── README.md                   # Documentación del proyecto
├── app/
│   ├── __init__.py             # Inicialización del paquete
│   ├── config.py               # Configuraciones y constantes
│   ├── main.py                 # Punto de entrada de FastAPI
│   ├── routers/
│   │   └── predict.py          # Definición del endpoint /predict
│   ├── schemas.py              # Modelos Pydantic de entrada/salida
│   └── services/
│       └── prediction_service.py  # Lógica de predicción encapsulada
├── model/
│   ├── __init__.py             # Inicialización del paquete
│   ├── classifier.py           # Wrapper del modelo de clasificación
│   ├── embeddings.py           # Cálculo de embeddings (BGE)
│   └── predictor.py            # Integración genérica del modelo
├── model.pt                    # Archivo de pesos del modelo pre-entrenado
├── poetry.lock                 # Lockfile de dependencias (Poetry)
├── pyproject.toml              # Configuración de Poetry y dependencias
├── training/                   # Scripts para procesar datos y entrenar modelo
│   ├── __init__.py
│   ├── dataset_processor.py    # Preprocesamiento de datasets
│   └── train.py                # Script de entrenamiento
├── utils/
│   ├── __init__.py             # Inicialización del paquete
│   └── text_cleaner.py         # Funciones de limpieza de texto
└── uv.lock                     # Lockfile para uv (gestión de entorno)
```

> **Nota:** El directorio `training/` alberga únicamente herramientas para entrenamiento y no es requerido en tiempo de ejecución de la API.

## Tecnologías y dependencias

* Python 3.8+
* FastAPI
* Uvicorn
* PyTorch
* Pydantic
* NLTK
* uv para gestión de dependencias

## Instalación

1. Clona el repositorio:

   ```bash
   git clone https://tu-repositorio-url.com/afi-sentiment-api.git
   cd afi-sentiment-api
   ```

2. Instala dependencias con Poetry:

   ```bash
   poetry install
   ```

3. Asegúrate de tener descargados los recursos de NLTK:

   ```bash
   python -c "import nltk; nltk.download('punkt')"
   ```

## Ejecución de la API

Utiliza `uv` para iniciar el servidor en un entorno controlado:

```bash
uv run python -m app.main
```

* Por defecto se expondrá en `http://127.0.0.1:8000`.
* Para desarrollo, añade la opción `--reload` tras `uv run`.

## Endpoints

### POST `/predict`

**Descripción:**
Recibe un JSON con el texto a analizar y devuelve el sentimiento global y por oración.

**Request**:

```json
{
  "text": "Tu texto aquí."
}
```

**Response**:

```json
{
  "overall_sentiment": {
    "label": "positive",
    "mean_score": 1.33
  },
  "sentence_sentiments": [
    {
      "sentence": "Primera oración.",
      "label": "neutral",
      "score": 1
    },
    {
      "sentence": "Segunda oración.",
      "label": "positive",
      "score": 2
    }
  ]
}
```

## Entrenamiento del modelo

Para entrenar un nuevo modelo desde cero o afinar uno existente:

```bash
# Desde el directorio raíz
python -m training.train --data-path data/tu_dataset.csv --output model.pt
```

El script `training/dataset_processor.py` contiene utilidades para preparar el dataset (tokenización, limpieza, split).

## Docker
El repositorio contiene un DockerFile y docker-compose para levantar el modelo en api. 
```bash
  docker-compose up -d
  docker run sentiment-model
```

#TODO: Insertar en dockerfile la instalación de punkt_tab para ntlk, de momento es manual:
```bash
  docker exec -it sentiment-model bash .
  pip install ntlk
```

```python
  import ntlk
  ntlk.download('punkt_tab')
```

