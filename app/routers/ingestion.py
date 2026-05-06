"""
ingestion.py

Router de FastAPI para el endpoint de ingesta de documentos PDF.

Responsabilidad:
    Gestionar la recepción de archivos PDF, persistirlos en disco
    y lanzar el pipeline de procesamiento completo en segundo plano:
    DocumentLoader → DocumentChunker → DocumentEmbedder → VectorStore.

    El endpoint devuelve confirmación inmediata al cliente mientras
    el procesamiento ocurre en segundo plano mediante BackgroundTasks
    combinado con run_in_threadpool, evitando que las operaciones
    CPU-intensivas (Docling, BGE-M3) bloqueen el event loop principal
    de FastAPI e impidan atender otros requests durante la ingesta.

Autor:   Jesús Rodríguez
Versión: 1.1.0
"""

import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.concurrency import run_in_threadpool

from app.schemas.models import UploadResponse, ErrorResponse
from app.state import get_embedder, get_vector_store
from src.ingestion import DocumentLoader, DocumentChunker


# ---------------------------------------------------------------------------
# Configuración del router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/upload", tags=["Ingesta"])

# Directorio donde se persisten los PDFs recibidos
# En producción usar almacenamiento en la nube (S3, GCS)
DIR_RAW = Path("data/raw")
DIR_RAW.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Función de procesamiento en segundo plano
# ---------------------------------------------------------------------------

def procesar_pdf(ruta_pdf: Path) -> None:
    """Procesa un PDF completo: extracción, chunking, embeddings e indexación.

    Función síncrona ejecutada en un thread separado mediante
    run_in_threadpool para no bloquear el event loop principal de FastAPI.
    Las operaciones Docling y BGE-M3 son CPU-intensivas y síncronas —
    ejecutarlas directamente en BackgroundTasks bloquearía todos los
    endpoints mientras dura la ingesta.

    Args:
        ruta_pdf: ruta al archivo PDF ya persistido en data/raw/.
    """
    try:
        # Fase 1: Extracción y limpieza del texto
        loader = DocumentLoader()
        texto  = loader.load(ruta_pdf)

        # Fase 2: División en chunks contextuales
        chunker = DocumentChunker()
        chunks  = chunker.split(texto)

        # Fase 3: Generación de embeddings
        embedder   = get_embedder()
        embeddings = embedder.embed(chunks)

        # Fase 4: Indexación en ChromaDB
        store = get_vector_store()
        store.index(
            chunks=chunks,
            embeddings=embeddings,
            fuente=ruta_pdf.stem,
        )

        print(f"[OK] Procesamiento completado: {ruta_pdf.name} — {len(chunks)} chunks indexados")

    except Exception as e:
        # En producción: registrar en sistema de logging centralizado
        print(f"[ERROR] Procesamiento fallido para {ruta_pdf.name}: {e}")


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/",
    response_model=UploadResponse,
    responses={400: {"model": ErrorResponse}},
    summary="Subir e indexar un documento PDF",
    description=(
        "Recibe un archivo PDF, lo persiste en disco y lanza el pipeline "
        "de procesamiento en segundo plano (extracción → chunking → "
        "embeddings → indexación). El cliente recibe confirmación inmediata "
        "sin esperar a que termine el procesamiento."
    ),
)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(
        ...,
        description="Archivo PDF a indexar.",
    ),
) -> UploadResponse:
    """Endpoint POST /upload — ingesta de documentos PDF."""

    # Validar que el archivo es un PDF
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Solo se aceptan archivos PDF.",
        )

    # Persistir el archivo en disco
    ruta_pdf = DIR_RAW / file.filename
    with ruta_pdf.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Lanzar el procesamiento en un thread separado mediante run_in_threadpool
    # para que Docling y BGE-M3 no bloqueen el event loop principal de FastAPI
    background_tasks.add_task(run_in_threadpool, procesar_pdf, ruta_pdf)

    return UploadResponse(
        mensaje        ="Archivo recibido. El procesamiento ha comenzado en segundo plano.",
        nombre_archivo =file.filename,
        estado         ="procesando",
    )