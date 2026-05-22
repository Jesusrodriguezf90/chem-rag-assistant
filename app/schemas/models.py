"""
models.py

Modelos Pydantic para validación de requests y responses de la API.

Responsabilidad:
    Definir los esquemas de datos para todos los endpoints de la API.
    Pydantic valida automáticamente los tipos y valores de entrada
    y serializa las respuestas a JSON.

    Incluye modelos para:
      - Pipeline RAG clásico: QueryRequest, QueryResponse
      - Agente RAG LangGraph: AgentQueryRequest, AgentQueryResponse
      - Sistema: UploadResponse, HealthResponse, ErrorResponse

Autor:   Jesús Rodríguez
Versión: 1.1.0
"""

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Modelos de request
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Request para el endpoint POST /query.

    Attributes:
        pregunta: consulta del usuario en cualquier idioma.
        top_k: número máximo de chunks a recuperar. Opcional —
               si no se especifica usa el valor por defecto del pipeline.
    """
    pregunta: str = Field(
        ...,
        min_length=1,
        description="Consulta del usuario sobre el documento científico.",
        examples=["What metals are used in the NHC complexes studied?"],
    )
    top_k: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Número máximo de chunks a recuperar (1-10).",
    )

class AgentQueryRequest(BaseModel):
    """Request para el endpoint POST /agent/query.

    Attributes:
        pregunta: consulta del usuario en cualquier idioma.
    """
    pregunta: str = Field(
        ...,
        min_length=1,
        description="Consulta del usuario sobre el documento científico.",
        examples=["What is the yield of AuL9 in the synthesis described in the paper?"],
    )

# ---------------------------------------------------------------------------
# Modelos de response
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    """Response para el endpoint POST /upload.

    Attributes:
        mensaje: confirmación de recepción del archivo.
        nombre_archivo: nombre del PDF recibido.
        estado: estado del procesamiento en segundo plano.
    """
    mensaje       : str
    nombre_archivo: str
    estado        : str = Field(
        description="Estado del procesamiento: 'procesando' o 'error'.",
    )

class QueryResponse(BaseModel):
    """Response para el endpoint POST /query.

    Attributes:
        pregunta: consulta original del usuario.
        respuesta: respuesta generada por el LLM.
        n_chunks: número de chunks usados como contexto.
        similitud_top1: similitud del chunk más relevante recuperado.
        tiempo_total_s: tiempo de ejecución end-to-end en segundos.
    """
    pregunta      : str
    respuesta     : str
    n_chunks      : int
    similitud_top1: float = Field(
        description="Similitud coseno del chunk más relevante (0.0-1.0).",
    )
    tiempo_total_s: float

class HealthResponse(BaseModel):
    """Response para el endpoint GET /health.

    Attributes:
        estado: estado del servicio ('ok' o 'error').
        modelo_embeddings: nombre del modelo de embeddings activo.
        modelo_generacion: nombre del modelo de generación activo.
        documentos_indexados: número de chunks en el índice vectorial.
    """
    estado              : str
    modelo_embeddings   : str
    modelo_generacion   : str
    documentos_indexados: int


class ErrorResponse(BaseModel):
    """Response estándar para errores de la API.

    Attributes:
        error: descripción del error.
        detalle: información adicional sobre el error (opcional).
    """
    error  : str
    detalle: str = ""

class AgentQueryResponse(BaseModel):
    """Response para el endpoint POST /agent/query.

    Attributes:
        pregunta      : consulta original del usuario.
        respuesta     : respuesta generada por el LLM.
        ruta          : ruta seguida por el agente
                        (paper_especifico / general / fuera_dominio).
        n_chunks      : número de chunks usados como contexto.
                        0 si la ruta fue general o fuera_dominio.
        intentos      : número de intentos de recuperación realizados.
                        0 si los chunks fueron relevantes en el primer intento.
        tiempo_total_s: tiempo de ejecución end-to-end en segundos.
    """
    pregunta      : str
    respuesta     : str
    ruta          : str = Field(
        description="Ruta seguida: paper_especifico, general o fuera_dominio.",
    )
    n_chunks      : int
    intentos      : int = Field(
        description="Número de intentos de reescritura de query (0-2).",
    )
    tiempo_total_s: float