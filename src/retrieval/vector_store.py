"""
vector_store.py

Módulo de indexación y recuperación semántica con ChromaDB.

Responsabilidad:
    Encapsular toda la interacción con ChromaDB: creación de colecciones,
    indexación de embeddings con metadatos, búsqueda por similitud coseno
    y filtrado por umbral mínimo de similitud.

    Este módulo es la cuarta etapa del pipeline RAG y su output
    es la entrada de src/pipeline/rag.py.

Notas para migración a producción:
    - Añadir logging para trazabilidad de cada operación de indexación
      y recuperación
    - Añadir filtrado por metadatos (fuente=) para soporte multi-documento
    - Considerar Qdrant como alternativa a ChromaDB para mayor escala

Autor:   Jesús Rodríguez
Versión: 1.0.0
"""

from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np

# ---------------------------------------------------------------------------
# Constantes por defecto
# ---------------------------------------------------------------------------

DEFAULT_COLLECTION_NAME      = "chem_rag"
DEFAULT_TOP_K                = 4
DEFAULT_SIMILARITY_THRESHOLD = 0.50

# ---------------------------------------------------------------------------
# Dataclass de resultado de recuperación
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """Resultado de una búsqueda semántica en ChromaDB.

    Attributes:
        chunks: lista de textos de los chunks recuperados.
        similitudes: puntuaciones de similitud coseno para cada chunk.
        metadatos: lista de diccionarios con metadatos de cada chunk.
    """
    chunks     : list[str]
    similitudes: list[float]
    metadatos  : list[dict]

# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class VectorStore:
    """Gestiona la indexación y recuperación semántica con ChromaDB.

    Encapsula la creación de colecciones, indexación de embeddings
    y búsqueda por similitud coseno. Usa métrica coseno porque es
    la correcta para embeddings normalizados de BGE-M3.

    Métodos de búsqueda:
        search: recupera los top-k chunks sin filtrado — útil para
                evaluación y diagnóstico del pipeline.
        search_filtered: recupera y filtra por umbral mínimo de similitud
                        — usar en producción antes de construir el prompt
                        del LLM para evitar contexto irrelevante.

    Attributes:
        persist_path: ruta donde ChromaDB persiste los datos en disco.
        collection_name: nombre de la colección en ChromaDB.
        top_k: número máximo de chunks a recuperar por consulta.
        similarity_threshold: similitud mínima para incluir un chunk
                              en los resultados filtrados.
        _cliente: instancia de ChromaDB PersistentClient.
        _coleccion: colección activa de ChromaDB.
    """

    def __init__(
        self,
        persist_path        : Path,
        collection_name     : str   = DEFAULT_COLLECTION_NAME,
        top_k               : int   = DEFAULT_TOP_K,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        """Inicializa el VectorStore y conecta con ChromaDB.

        Args:
            persist_path: ruta donde ChromaDB persiste los datos.
            collection_name: nombre de la colección a usar o crear.
            top_k: número máximo de chunks a recuperar.
            similarity_threshold: similitud mínima para filtrar resultados.
        """
        self.persist_path         = persist_path
        self.collection_name      = collection_name
        self.top_k                = top_k
        self.similarity_threshold = similarity_threshold

        self._cliente   = chromadb.PersistentClient(path=str(persist_path))
        self._coleccion = self._inicializar_coleccion()

    # -----------------------------------------------------------------------
    # API pública
    # -----------------------------------------------------------------------

    def index(
        self,
        chunks    : list[str],
        embeddings: np.ndarray,
        fuente    : str = "documento",
    ) -> None:
        """Indexa chunks y sus embeddings en ChromaDB.

        Recrea la colección si ya existe para evitar duplicados
        en caso de re-indexación del mismo documento.

        Args:
            chunks: lista de textos a indexar.
            embeddings: array numpy de forma (n_chunks, 1024).
            fuente: identificador del documento de origen.
                    Usado como metadato para filtrado futuro
                    en pipelines multi-documento.

        Raises:
            ValueError: si el número de chunks y embeddings no coincide.
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"El número de chunks ({len(chunks)}) no coincide "
                f"con el número de embeddings ({embeddings.shape[0]})"
            )

        # Recrear colección para evitar duplicados en re-indexación
        self._cliente.delete_collection(name=self.collection_name)
        self._coleccion = self._inicializar_coleccion()

        self._coleccion.add(
            ids=[f"chunk_{i:04d}" for i in range(len(chunks))],
            embeddings=embeddings.tolist(),
            documents=chunks,
            metadatas=[
                {
                    "fuente"  : fuente,
                    "indice"  : i,
                    "longitud": len(chunks[i]),
                }
                for i in range(len(chunks))
            ],
        )

    def search(self, query_vector: list[float]) -> RetrievalResult:
        """Busca los chunks más similares al vector de consulta.

        Recupera los top-k chunks por similitud coseno sin aplicar
        el umbral mínimo. Usar search_filtered para recuperación
        con filtrado por umbral.

        Args:
            query_vector: vector de consulta de dimensión 1024,
                          output de DocumentEmbedder.embed_query().

        Returns:
            RetrievalResult con los top-k chunks y sus similitudes.

        Raises:
            ValueError: si la colección está vacía.
        """
        if self._coleccion.count() == 0:
            raise ValueError(
                "La colección está vacía. "
                "Ejecuta index() antes de realizar búsquedas."
            )

        resultados = self._coleccion.query(
            query_embeddings=[query_vector],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks     = resultados["documents"][0]
        distancias = resultados["distances"][0]
        metadatos  = resultados["metadatas"][0]

        # ChromaDB con métrica coseno devuelve distancias (1 - similitud)
        similitudes = [round(1 - d, 4) for d in distancias]

        return RetrievalResult(
            chunks=chunks,
            similitudes=similitudes,
            metadatos=metadatos,
        )

    def search_filtered(self, query_vector: list[float]) -> RetrievalResult:
        """Busca chunks similares y filtra por umbral mínimo de similitud.

        Aplica el umbral similarity_threshold para descartar chunks
        poco relevantes antes de construir el prompt del LLM.

        Args:
            query_vector: vector de consulta de dimensión 1024.

        Returns:
            RetrievalResult con solo los chunks que superan el umbral.
        """
        resultado = self.search(query_vector)

        indices_validos = [
            i for i, sim in enumerate(resultado.similitudes)
            if sim >= self.similarity_threshold
        ]

        return RetrievalResult(
            chunks=[resultado.chunks[i] for i in indices_validos],
            similitudes=[resultado.similitudes[i] for i in indices_validos],
            metadatos=[resultado.metadatos[i] for i in indices_validos],
        )

    def count(self) -> int:
        """Devuelve el número de documentos indexados en la colección."""
        return self._coleccion.count()

    def close(self) -> None:
        """Cierra la conexión con ChromaDB liberando los recursos del sistema.

        Necesario en Windows para evitar PermissionError al eliminar
        directorios temporales mientras ChromaDB mantiene archivos SQLite
        abiertos. En Linux/macOS el sistema operativo libera los recursos
        automáticamente al finalizar el proceso.

        Uso recomendado en tests con directorios temporales:
            store = VectorStore(persist_path=tmp_path)
            try:
                ...
            finally:
                store.close()

        O como fixture de pytest:
            yield store
            store.close()
        """
        self._cliente._system.stop()

    # -----------------------------------------------------------------------
    # Métodos privados
    # -----------------------------------------------------------------------

    def _inicializar_coleccion(self) -> chromadb.Collection:
        """Obtiene o crea la colección ChromaDB con métrica coseno.

        Returns:
            Colección ChromaDB lista para indexar o consultar.
        """
        try:
            return self._cliente.get_collection(name=self.collection_name)
        except Exception:
            return self._cliente.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
