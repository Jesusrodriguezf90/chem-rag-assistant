"""
test_embedder.py

Tests unitarios para src/embeddings/embedder.py

Cubre:
    - Generación de embeddings para chunks
    - Generación de embedding para consultas
    - Validación dimensional del output
    - Configuración personalizada via EmbedderConfig
    - Manejo de errores

Nota sobre el uso de mock:
    DocumentEmbedder carga BGE-M3 (~2.2 GB) automáticamente en __init__.
    Sin mock, cada fixture que instancia DocumentEmbedder descargaría
    el modelo completo antes de ejecutar cualquier test — inviable en
    un suite de tests automatizado.
    El mock intercepta BGEM3FlagModel y lo sustituye por un objeto que
    simula su comportamiento (vectores de dimensión 1024 normalizados)
    sin descargar nada. La validación del modelo real se realiza en
    los notebooks de Colab (02_embeddings.ipynb, 03_recuperacion.ipynb).

Autor:   Jesús Rodríguez
Versión: 1.0.0
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.embeddings import DocumentEmbedder, EmbedderConfig, EMBEDDING_DIMENSION

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_modelo():
    """Mock de BGEM3FlagModel para evitar cargar el modelo real en tests.

    Simula la respuesta de BGE-M3 con vectores aleatorios normalizados
    de dimensión 1024, idéntica a la del modelo real.
    """
    modelo = MagicMock()

    def encode_mock(textos, **kwargs):
        """Simula la respuesta de BGE-M3 con vectores aleatorios normalizados."""
        n = len(textos)
        vectores = np.random.rand(n, EMBEDDING_DIMENSION).astype(np.float32)
        # Normalizar para simular el comportamiento real de BGE-M3
        normas = np.linalg.norm(vectores, axis=1, keepdims=True)
        vectores = vectores / normas
        return {"dense_vecs": vectores}

    modelo.encode.side_effect = encode_mock
    return modelo

@pytest.fixture
def embedder(mock_modelo):
    """Instancia DocumentEmbedder con modelo mockeado.

    El patch intercepta BGEM3FlagModel en el momento en que
    DocumentEmbedder.__init__ llama a _cargar_modelo(), evitando
    la descarga del modelo real sin alterar la lógica del embedder.
    """
    with patch(
        "src.embeddings.embedder.BGEM3FlagModel",
        return_value=mock_modelo,
    ):
        yield DocumentEmbedder()

@pytest.fixture
def chunks_ejemplo():
    """Lista de chunks de ejemplo para tests."""
    return [
        "## Introduction\n\nN-heterocyclic carbenes (NHCs) form stable complexes.",
        "## Results\n\nThe synthesis of PdL3 was achieved using standard methods.",
        "## Conclusion\n\nThe Au(III) complexes show cytotoxicity in tumor cells.",
    ]

# ---------------------------------------------------------------------------
# Tests de generación de embeddings
# ---------------------------------------------------------------------------

class TestEmbed:
    """Tests para el método embed."""

    def test_embed_devuelve_numpy_array(self, embedder, chunks_ejemplo):
        resultado = embedder.embed(chunks_ejemplo)
        assert isinstance(resultado, np.ndarray)

    def test_embed_dimension_correcta(self, embedder, chunks_ejemplo):
        resultado = embedder.embed(chunks_ejemplo)
        assert resultado.shape == (len(chunks_ejemplo), EMBEDDING_DIMENSION)

    def test_embed_un_chunk(self, embedder):
        resultado = embedder.embed(["Texto de prueba para un único chunk."])
        assert resultado.shape == (1, EMBEDDING_DIMENSION)

    def test_embed_lista_vacia_lanza_error(self, embedder):
        with pytest.raises(ValueError):
            embedder.embed([])

    def test_embed_vectores_normalizados(self, embedder, chunks_ejemplo):
        resultado = embedder.embed(chunks_ejemplo)
        # Los vectores de BGE-M3 están normalizados — norma ~1.0
        normas = np.linalg.norm(resultado, axis=1)
        assert np.allclose(normas, 1.0, atol=1e-5)

# ---------------------------------------------------------------------------
# Tests de generación de embedding para consultas
# ---------------------------------------------------------------------------

class TestEmbedQuery:
    """Tests para el método embed_query."""

    def test_embed_query_devuelve_lista(self, embedder):
        resultado = embedder.embed_query("What metals are used in the study?")
        assert isinstance(resultado, list)

    def test_embed_query_dimension_correcta(self, embedder):
        resultado = embedder.embed_query("What metals are used in the study?")
        assert len(resultado) == EMBEDDING_DIMENSION

    def test_embed_query_vacia_lanza_error(self, embedder):
        with pytest.raises(ValueError):
            embedder.embed_query("")

    def test_embed_query_solo_espacios_lanza_error(self, embedder):
        with pytest.raises(ValueError):
            embedder.embed_query("   ")

    def test_embed_query_devuelve_floats(self, embedder):
        resultado = embedder.embed_query("What metals are used?")
        assert all(isinstance(v, float) for v in resultado)

# ---------------------------------------------------------------------------
# Tests de configuración
# ---------------------------------------------------------------------------

class TestConfiguracion:
    """Tests para EmbedderConfig y configuración del embedder."""

    def test_config_por_defecto(self, embedder):
        assert embedder.config.model_name == "BAAI/bge-m3"
        assert embedder.config.batch_size == 4
        assert embedder.config.max_length == 512
        assert embedder.config.use_fp16   is True
        assert embedder.config.device     == "cpu"

    def test_config_personalizada(self):
        config = EmbedderConfig(batch_size=8, max_length=256, device="cpu")
        with patch("src.embeddings.embedder.BGEM3FlagModel"):
            embedder = DocumentEmbedder(config=config)
        assert embedder.config.batch_size == 8
        assert embedder.config.max_length == 256

    def test_dimension_esperada_es_1024(self):
        assert EMBEDDING_DIMENSION == 1024

# ---------------------------------------------------------------------------
# Tests de consistencia entre embed y embed_query
# ---------------------------------------------------------------------------

class TestConsistencia:
    """Tests para verificar que embed y embed_query son comparables."""

    def test_embed_y_embed_query_misma_dimension(self, embedder, chunks_ejemplo):
        embeddings_chunks = embedder.embed(chunks_ejemplo)
        embedding_query   = embedder.embed_query("What metals are studied?")
        # Ambos deben tener la misma dimensión para que la búsqueda
        # por similitud en ChromaDB sea coherente — el modelo debe ser
        # idéntico en indexación y en consulta
        assert embeddings_chunks.shape[1] == len(embedding_query)
