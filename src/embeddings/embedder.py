"""
embedder.py

Módulo de generación de embeddings semánticos para el pipeline RAG.

Responsabilidad:
    Recibir una lista de chunks de texto y devolver sus representaciones
    vectoriales densas usando el modelo BGE-M3 (BAAI), seleccionado como
    modelo de embeddings óptimo por su rendimiento multilingüe y soporte
    de hasta 8192 tokens por chunk.

    Este módulo es la tercera etapa del pipeline RAG y su output
    es la entrada de src/retrieval/vector_store.py.

Autor:   Jesús Rodríguez
Versión: 1.0.0
"""

from dataclasses import dataclass

import numpy as np
from FlagEmbedding import BGEM3FlagModel

# ---------------------------------------------------------------------------
# Constantes por defecto
# ---------------------------------------------------------------------------

DEFAULT_MODEL_NAME    = "BAAI/bge-m3"
DEFAULT_BATCH_SIZE    = 4
DEFAULT_MAX_LENGTH    = 512
DEFAULT_USE_FP16      = True
EMBEDDING_DIMENSION   = 1024

# ---------------------------------------------------------------------------
# Dataclass de configuración
# ---------------------------------------------------------------------------

@dataclass
class EmbedderConfig:
    """Parámetros de configuración para el embedder.

    Attributes:
        model_name: nombre del modelo en HuggingFace.
        batch_size: número de chunks procesados simultáneamente.
                    Valor conservador para CPU — aumentar a 8-16 con GPU.
        max_length: límite de tokens por chunk antes de truncar.
        use_fp16: usar precisión media para reducir memoria a la mitad
                  sin pérdida significativa de calidad.
        device: dispositivo de cómputo ('cpu' o 'cuda').
    """
    model_name: str  = DEFAULT_MODEL_NAME
    batch_size : int  = DEFAULT_BATCH_SIZE
    max_length : int  = DEFAULT_MAX_LENGTH
    use_fp16   : bool = DEFAULT_USE_FP16
    device     : str  = "cpu"

# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class DocumentEmbedder:
    """Genera embeddings semánticos densos usando BGE-M3.

    Encapsula la carga del modelo y la generación de vectores,
    garantizando que el mismo modelo se usa tanto para indexar
    los chunks del documento como para vectorizar las consultas
    del usuario — requisito fundamental para que la búsqueda
    por similitud sea coherente.

    Attributes:
        config: parámetros de configuración del embedder.
        _modelo: instancia cargada de BGEM3FlagModel.
    """

    def __init__(self, config: EmbedderConfig | None = None) -> None:
        """Inicializa el embedder y carga el modelo BGE-M3.

        La carga del modelo descarga ~2.2 GB en la primera ejecución.
        Las ejecuciones posteriores usan la caché local de HuggingFace.

        Args:
            config: parámetros de configuración. Si es None, usa los
                    valores por defecto definidos en EmbedderConfig.
        """
        self.config = config or EmbedderConfig()
        self._modelo = self._cargar_modelo()

    # -----------------------------------------------------------------------
    # API pública
    # -----------------------------------------------------------------------

    def embed(self, chunks: list[str]) -> np.ndarray:
        """Genera embeddings densos para una lista de chunks.

        Args:
            chunks: lista de textos a vectorizar. Cada chunk debe ser
                    el output de DocumentChunker.split().

        Returns:
            Array numpy de forma (n_chunks, 1024) con los vectores
            densos normalizados listos para indexar en ChromaDB.

        Raises:
            ValueError: si la lista de chunks está vacía.
            ValueError: si la dimensión del resultado no es 1024.
        """
        if not chunks:
            raise ValueError("La lista de chunks no puede estar vacía.")

        resultado = self._modelo.encode(
            chunks,
            batch_size=self.config.batch_size,
            max_length=self.config.max_length,
            return_dense=True,
            return_sparse=False,      # No necesario para este pipeline
            return_colbert_vecs=False, # Re-ranking avanzado — no necesario aquí
        )

        embeddings = resultado["dense_vecs"]

        if embeddings.shape[1] != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Dimensión inesperada: {embeddings.shape[1]}, "
                f"se esperaban {EMBEDDING_DIMENSION}"
            )

        return embeddings

    def embed_query(self, query: str) -> list[float]:
        """Genera el embedding de una consulta de usuario.

        El modelo debe ser idéntico al usado para indexar los chunks
        para garantizar que los vectores están en el mismo espacio
        semántico y la búsqueda por similitud es coherente.

        Args:
            query: texto de la consulta del usuario.

        Returns:
            Vector denso de 1024 dimensiones como lista de floats,
            listo para consultar ChromaDB.

        Raises:
            ValueError: si la consulta está vacía.
        """
        if not query.strip():
            raise ValueError("La consulta no puede estar vacía.")

        resultado = self._modelo.encode(
            [query],
            batch_size=1,
            max_length=self.config.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        return resultado["dense_vecs"][0].tolist()

    # -----------------------------------------------------------------------
    # Métodos privados
    # -----------------------------------------------------------------------

    def _cargar_modelo(self) -> BGEM3FlagModel:
        """Carga el modelo BGE-M3 desde HuggingFace.

        Returns:
            Instancia de BGEM3FlagModel lista para inferencia.
        """
        return BGEM3FlagModel(
            self.config.model_name,
            use_fp16=self.config.use_fp16,
            device=self.config.device,
        )
