"""
models.py

Modelos Pydantic para validación de requests y responses de la API.

Responsabilidad:
    Definir los esquemas de datos para todos los endpoints de la API.
    Pydantic valida automáticamente los tipos y valores de entrada
    y serializa las respuestas a JSON.

Autor:   Jesús Rodríguez
Versión: 1.0.0
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
