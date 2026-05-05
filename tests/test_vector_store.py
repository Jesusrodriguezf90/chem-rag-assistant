"""
test_vector_store.py

Tests unitarios para src/retrieval/vector_store.py

Cubre:
    - Indexación de chunks y embeddings
    - Búsqueda semántica sin filtrado (search)
    - Búsqueda semántica con filtrado por umbral (search_filtered)
    - Validación de coherencia entre chunks y embeddings
    - Manejo de errores
    - Configuración personalizada

Nota sobre ChromaDB en Windows:
    ChromaDB mantiene los archivos SQLite abiertos mientras el cliente
    está activo. En Windows esto impide eliminar el directorio temporal
    al finalizar el test (PermissionError WinError 32).
    La solución profesional tiene dos partes:
      1. VectorStore.close() — método público que cierra el cliente
         correctamente. No se accede a internos de ChromaDB desde los tests.
      2. ignore_cleanup_errors=True en TemporaryDirectory — solución
         oficial de pytest para errores de limpieza en Windows.
    En Linux/macOS el sistema libera los recursos automáticamente
    y close() no es estrictamente necesario, pero se llama igualmente
    para garantizar comportamiento consistente entre plataformas.

Autor:   Jesús Rodríguez
Versión: 1.0.0
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.retrieval import (
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    RetrievalResult,
    VectorStore,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def directorio_temporal():
    """Crea un directorio temporal para ChromaDB en cada test.

    Usa ignore_cleanup_errors=True para evitar PermissionError en Windows
    cuando ChromaDB mantiene archivos SQLite abiertos al finalizar el test.
    Es la solución oficial recomendada por el repositorio de pytest
    para este tipo de errores en Windows.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def vector_store(directorio_temporal):
    """Instancia VectorStore con directorio temporal.

    Llama a close() en el teardown para liberar los archivos SQLite
    de ChromaDB antes de que el directorio temporal intente eliminarse.
    Usa la API pública close() — no accede a internos de ChromaDB.
    """
    store = VectorStore(persist_path=directorio_temporal)
    yield store
    store.close()


@pytest.fixture
def chunks_ejemplo():
    """Lista de chunks de ejemplo para tests.

    Contiene exactamente DEFAULT_TOP_K (4) chunks para que los tests
    de búsqueda puedan verificar que ChromaDB devuelve top_k resultados.
    ChromaDB no puede devolver más chunks de los que tiene indexados,
    por lo que el número de chunks debe ser >= top_k.
    """
    return [
        "## Introduction\n\nNHC complexes with Pd and Pt metals.",
        "## Results\n\nThe synthesis of PdL3 was achieved.",
        "## Conclusion\n\nAu(III) complexes show cytotoxicity.",
        "## Experimental\n\nAll reactions were performed under nitrogen.",
    ]


@pytest.fixture
def embeddings_ejemplo(chunks_ejemplo):
    """Embeddings normalizados de ejemplo — simulan BGE-M3."""
    n        = len(chunks_ejemplo)
    vectores = np.random.rand(n, 1024).astype(np.float32)
    normas   = np.linalg.norm(vectores, axis=1, keepdims=True)
    return vectores / normas


@pytest.fixture
def vector_store_indexado(vector_store, chunks_ejemplo, embeddings_ejemplo):
    """VectorStore con datos ya indexados."""
    vector_store.index(chunks_ejemplo, embeddings_ejemplo)
    return vector_store


@pytest.fixture
def query_vector():
    """Vector de consulta normalizado de ejemplo."""
    v    = np.random.rand(1024).astype(np.float32)
    norm = np.linalg.norm(v)
    return (v / norm).tolist()


# ---------------------------------------------------------------------------
# Tests de indexación
# ---------------------------------------------------------------------------

class TestIndex:
    """Tests para el método index."""

    def test_index_almacena_chunks(
        self, vector_store, chunks_ejemplo, embeddings_ejemplo
    ):
        vector_store.index(chunks_ejemplo, embeddings_ejemplo)
        assert vector_store.count() == len(chunks_ejemplo)

    def test_index_chunks_embeddings_incoherentes_lanza_error(
        self, vector_store, embeddings_ejemplo
    ):
        chunks_distintos = ["Solo un chunk"]
        with pytest.raises(ValueError):
            vector_store.index(chunks_distintos, embeddings_ejemplo)

    def test_index_reindexacion_no_duplica(
        self, vector_store, chunks_ejemplo, embeddings_ejemplo
    ):
        # Indexar dos veces debe resultar en el mismo número de documentos
        vector_store.index(chunks_ejemplo, embeddings_ejemplo)
        vector_store.index(chunks_ejemplo, embeddings_ejemplo)
        assert vector_store.count() == len(chunks_ejemplo)

    def test_index_guarda_metadatos(
        self, vector_store, chunks_ejemplo, embeddings_ejemplo
    ):
        vector_store.index(chunks_ejemplo, embeddings_ejemplo, fuente="PMC10967698")
        muestra = vector_store._coleccion.get(
            ids=["chunk_0000"],
            include=["metadatas"],
        )
        assert muestra["metadatas"][0]["fuente"] == "PMC10967698"


# ---------------------------------------------------------------------------
# Tests de búsqueda sin filtrado
# ---------------------------------------------------------------------------

class TestSearch:
    """Tests para el método search."""

    def test_search_devuelve_retrieval_result(
        self, vector_store_indexado, query_vector
    ):
        resultado = vector_store_indexado.search(query_vector)
        assert isinstance(resultado, RetrievalResult)

    def test_search_devuelve_top_k_chunks(
        self, vector_store_indexado, query_vector
    ):
        resultado = vector_store_indexado.search(query_vector)
        # chunks_ejemplo tiene exactamente DEFAULT_TOP_K (4) chunks
        assert len(resultado.chunks) == DEFAULT_TOP_K

    def test_search_similitudes_entre_0_y_1(
        self, vector_store_indexado, query_vector
    ):
        resultado = vector_store_indexado.search(query_vector)
        assert all(0.0 <= sim <= 1.0 for sim in resultado.similitudes)

    def test_search_similitudes_ordenadas_descendente(
        self, vector_store_indexado, query_vector
    ):
        resultado = vector_store_indexado.search(query_vector)
        assert resultado.similitudes == sorted(
            resultado.similitudes, reverse=True
        )

    def test_search_coleccion_vacia_lanza_error(
        self, vector_store, query_vector
    ):
        with pytest.raises(ValueError):
            vector_store.search(query_vector)


# ---------------------------------------------------------------------------
# Tests de búsqueda con filtrado
# ---------------------------------------------------------------------------

class TestSearchFiltered:
    """Tests para el método search_filtered."""

    def test_search_filtered_devuelve_retrieval_result(
        self, vector_store_indexado, query_vector
    ):
        resultado = vector_store_indexado.search_filtered(query_vector)
        assert isinstance(resultado, RetrievalResult)

    def test_search_filtered_todos_superan_umbral(
        self, vector_store_indexado, query_vector
    ):
        resultado = vector_store_indexado.search_filtered(query_vector)
        assert all(
            sim >= DEFAULT_SIMILARITY_THRESHOLD
            for sim in resultado.similitudes
        )

    def test_search_filtered_umbral_alto_devuelve_menos_chunks(
        self, directorio_temporal, chunks_ejemplo, embeddings_ejemplo, query_vector
    ):
        # Con umbral muy alto (0.99) deberían filtrarse casi todos los chunks
        store = VectorStore(
            persist_path=directorio_temporal,
            similarity_threshold=0.99,
        )
        store.index(chunks_ejemplo, embeddings_ejemplo)
        resultado = store.search_filtered(query_vector)
        assert len(resultado.chunks) <= len(chunks_ejemplo)
        store.close()

    def test_search_filtered_umbral_cero_devuelve_todos_los_indexados(
        self, directorio_temporal, chunks_ejemplo, embeddings_ejemplo, query_vector
    ):
        # Con umbral 0.0 todos los chunks deben pasar el filtro
        # ChromaDB devuelve como máximo el número de chunks indexados
        store = VectorStore(
            persist_path=directorio_temporal,
            similarity_threshold=0.0,
        )
        store.index(chunks_ejemplo, embeddings_ejemplo)
        resultado = store.search_filtered(query_vector)
        assert len(resultado.chunks) == len(chunks_ejemplo)
        store.close()


# ---------------------------------------------------------------------------
# Tests de configuración
# ---------------------------------------------------------------------------

class TestConfiguracion:
    """Tests para la configuración del VectorStore."""

    def test_config_por_defecto(self, vector_store):
        assert vector_store.top_k                == DEFAULT_TOP_K
        assert vector_store.similarity_threshold == DEFAULT_SIMILARITY_THRESHOLD
        assert vector_store.collection_name      == "chem_rag"

    def test_config_personalizada(self, directorio_temporal):
        store = VectorStore(
            persist_path=directorio_temporal,
            collection_name="mi_coleccion",
            top_k=6,
            similarity_threshold=0.70,
        )
        assert store.top_k                == 6
        assert store.similarity_threshold == 0.70
        assert store.collection_name      == "mi_coleccion"
        store.close()

    def test_count_coleccion_vacia(self, vector_store):
        assert vector_store.count() == 0
