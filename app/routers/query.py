"""
query.py

Router de FastAPI para el endpoint de consulta en lenguaje natural.

Responsabilidad:
    Recibir una pregunta del usuario, ejecutar el pipeline RAG completo
    (recuperación semántica + generación con Qwen3) y devolver la respuesta
    fundamentada en el documento científico indexado.

Autor:   Jesús Rodríguez
Versión: 1.0.0
"""

from fastapi import APIRouter, HTTPException

from app.schemas.models import QueryRequest, QueryResponse, ErrorResponse
from app.state import get_embedder, get_vector_store, get_hf_token
from src.pipeline.rag import RAGPipeline

# ---------------------------------------------------------------------------
# Configuración del router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/query", tags=["Consulta"])

# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Consultar el documento indexado",
    description=(
        "Recibe una pregunta en lenguaje natural y devuelve una respuesta "
        "fundamentada en el documento científico indexado. Responde en el "
        "mismo idioma de la pregunta."
    ),
)
async def query_documento(request: QueryRequest) -> QueryResponse:
    """Endpoint POST /query — consulta en lenguaje natural."""

    # Verificar que hay documentos indexados antes de consultar
    store = get_vector_store()
    if store.count() == 0:
        raise HTTPException(
            status_code=503,
            detail=(
                "No hay documentos indexados. "
                "Usa POST /upload para indexar un documento primero."
            ),
        )

    # Inicializar el pipeline RAG con las dependencias compartidas
    pipeline = RAGPipeline(
        vector_store=store,
        embedder=get_embedder(),
        hf_token=get_hf_token(),
    )

    # Ejecutar el pipeline RAG completo
    resultado = pipeline.query(request.pregunta)

    # Similitud top-1 — 0.0 si no se recuperó ningún chunk
    similitud_top1 = (
        resultado.similitudes[0] if resultado.similitudes else 0.0
    )

    return QueryResponse(
        pregunta      =resultado.pregunta,
        respuesta     =resultado.respuesta,
        n_chunks      =resultado.n_chunks,
        similitud_top1=similitud_top1,
        tiempo_total_s=resultado.tiempo_total_s,
    )
