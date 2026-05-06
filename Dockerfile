# ===========================================================================
# Dockerfile — Chem RAG Assistant
#
# Imagen de producción para despliegue en Hugging Face Spaces (Docker SDK).
#
# Etapas:
#   1. builder: instala dependencias en un entorno limpio
#   2. runtime: imagen final mínima con solo lo necesario para ejecutar la API
#
# El patrón multi-stage reduce el tamaño final de la imagen eliminando
# las herramientas de build que no son necesarias en producción.
# ===========================================================================

# ---------------------------------------------------------------------------
# Etapa 1 — Builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Directorio de trabajo para la instalación de dependencias
WORKDIR /install

# Copiar solo requirements para aprovechar la caché de Docker
# Si requirements.txt no cambia, esta capa se reutiliza en builds posteriores
COPY requirements.txt .

# Instalar dependencias en un directorio aislado
# --no-cache-dir: evita almacenar caché de pip en la imagen
# --prefix: instala en /install/deps para copiar limpiamente a la imagen final
RUN pip install --no-cache-dir --prefix=/install/deps -r requirements.txt

# ---------------------------------------------------------------------------
# Etapa 2 — Runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Metadatos de la imagen
LABEL maintainer="Jesús Rodríguez"
LABEL description="Chem RAG Assistant — API REST para consulta de literatura científica"
LABEL version="0.1.0"

# HF Spaces ejecuta contenedores como usuario no root por seguridad
# Creamos un usuario dedicado para la aplicación
RUN useradd --create-home --shell /bin/bash appuser

# Directorio de trabajo de la aplicación
WORKDIR /app

# Copiar dependencias instaladas desde la etapa builder
COPY --from=builder /install/deps /usr/local

# Copiar el código fuente de la aplicación
# El .dockerignore excluye venv/, notebooks/, tests/, data/ y .env
COPY . .

# Crear las carpetas de datos necesarias en tiempo de ejecución
# ChromaDB y los PDFs se persistirán aquí dentro del contenedor
RUN mkdir -p data/raw data/chroma_db data/processed data/embeddings

# Cambiar la propiedad de los archivos al usuario de la aplicación
RUN chown -R appuser:appuser /app

# Ejecutar como usuario no root. HF Spaces requiere que los contenedores no se ejecuten como root por seguridad
USER appuser

# Puerto en el que escucha la aplicación
# HF Spaces espera que la aplicación escuche en el puerto 7860. En local se puede seguir usando puerto 8000
EXPOSE 7860

# Variables de entorno por defecto
# HF_TOKEN se pasa como secreto desde HF Spaces — nunca en el Dockerfile
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# Comando de arranque
# --host 0.0.0.0: necesario para que HF Spaces pueda acceder al contenedor
# --port 7860: puerto estándar de HF Spaces
# --workers 1: un único worker para evitar conflictos con ChromaDB
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
