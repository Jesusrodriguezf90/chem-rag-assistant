"""
main.py

Punto de entrada de la API REST Chem RAG Assistant.

Responsabilidad:
    Inicializar la aplicación FastAPI, registrar los routers de ingesta
    y consulta, gestionar el ciclo de vida del servidor (startup/shutdown)
    y exponer el endpoint de salud.

    Usa el patrón lifespan de FastAPI (recomendado desde v0.93) en lugar
    de los eventos on_event deprecated para gestionar la inicialización
    y limpieza de recursos.

Autor:   Jesús Rodríguez
Versión: 1.0.0
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.routers import ingestion, query
from app.schemas.models import HealthResponse
from app.state import get_embedder, get_vector_store, inicializar_estado
from src.embeddings import EMBEDDING_DIMENSION

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

DIR_CHROMA = Path("data/chroma_db")

# ---------------------------------------------------------------------------
# Ciclo de vida del servidor
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestiona el ciclo de vida de la aplicación.

    startup: inicializa BGE-M3 y ChromaDB una única vez al arrancar.
    shutdown: cierra ChromaDB correctamente para liberar archivos en Windows.
    """
    # Startup — inicializar dependencias compartidas
    print("Iniciando Chem RAG Assistant...")
    inicializar_estado(persist_path=DIR_CHROMA)
    print("Modelo BGE-M3 cargado y ChromaDB conectado.")

    yield

    # Shutdown — liberar recursos
    print("Cerrando Chem RAG Assistant...")
    get_vector_store().close()
    print("ChromaDB cerrado correctamente.")

# ---------------------------------------------------------------------------
# Aplicación FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Chem RAG Assistant",
    description=(
        "API REST para consulta de literatura científica de química "
        "organometálica mediante Generación Aumentada por Recuperación (RAG). "
        "Procesa PDFs científicos, los indexa semánticamente con BGE-M3 "
        "y genera respuestas fundamentadas con Qwen3."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# Registro de routers
# ---------------------------------------------------------------------------

app.include_router(ingestion.router)
app.include_router(query.router)

# ---------------------------------------------------------------------------
# Endpoints de salud
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Sistema"],
    summary="Estado del servicio",
)
async def health() -> HealthResponse:
    """Endpoint GET /health — verifica que el servicio está operativo."""
    return HealthResponse(
        estado              ="ok",
        modelo_embeddings   =f"BAAI/bge-m3 ({EMBEDDING_DIMENSION}d)",
        modelo_generacion   ="Qwen/Qwen3-8B",
        documentos_indexados=get_vector_store().count(),
    )
