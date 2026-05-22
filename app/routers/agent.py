"""
agent.py

Router de FastAPI para el endpoint de consulta con agente RAG LangGraph.

Responsabilidad:
    Recibir una pregunta del usuario, ejecutar el agente RAG con LangGraph
    (Router → Retriever → Grader → Rewriter → Generator) y devolver la
    respuesta junto con los metadatos de ejecución del agente.

    A diferencia de POST /query que siempre recupera del ChromaDB,
    POST /agent_query decide de forma autónoma si recuperar, responder
    con conocimiento general o responder directamente según la naturaleza
    de la query.

Autor:   Jesús Rodríguez
Versión: 1.0.0
"""

from fastapi import APIRouter, HTTPException

from app.schemas.models import AgentQueryRequest, AgentQueryResponse, ErrorResponse
from app.state import get_embedder, get_vector_store, get_hf_token
from src.agent import AgentGraph

# ---------------------------------------------------------------------------
# Configuración del router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/agent", tags=["Agente"])

# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/query",
    response_model=AgentQueryResponse,
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Consultar el documento con el agente RAG LangGraph",
    description=(
        "Recibe una pregunta en lenguaje natural y la procesa con el agente "
        "RAG adaptativo. El agente decide de forma autónoma si recuperar "
        "información del documento indexado, responder con conocimiento "
        "general de química, o responder directamente si la pregunta está "
        "fuera del dominio. Responde en el mismo idioma de la pregunta."
    ),
)
async def agent_query(request: AgentQueryRequest) -> AgentQueryResponse:
    """Endpoint POST /agent/query — consulta con agente RAG LangGraph."""

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

    # Inicializar el agente con las dependencias compartidas
    # Reutiliza las mismas instancias de VectorStore y DocumentEmbedder
    # que el pipeline RAG clásico — sin duplicar la carga de modelos
    agente = AgentGraph(
        vector_store=store,
        embedder=get_embedder(),
        hf_token=get_hf_token(),
    )

    # Ejecutar el agente RAG completo
    resultado = agente.query(request.pregunta)

    return AgentQueryResponse(
        pregunta      =resultado.pregunta,
        respuesta     =resultado.respuesta,
        ruta          =resultado.ruta,
        n_chunks      =len(resultado.chunks_usados),
        intentos      =resultado.intentos,
        tiempo_total_s=resultado.tiempo_total_s,
    )
