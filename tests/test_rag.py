"""
test_rag.py

Tests unitarios para src/pipeline/rag.py

Cubre:
    - Ejecución del pipeline RAG completo
    - Manejo de contexto vacío (sin chunks que superen el umbral)
    - Validación de estructura del RAGResult
    - Manejo de errores (pregunta vacía, token ausente)
    - Configuración personalizada

Nota sobre el uso de mock:
    Se mockean tres dependencias externas para aislar la lógica del pipeline:
      1. DocumentEmbedder: evita cargar BGE-M3 (~2.2 GB)
      2. VectorStore: evita necesitar ChromaDB inicializado
      3. InferenceClient: evita llamadas reales a HF Inference API
         que consumirían créditos y requieren conexión a internet
    La validación del pipeline completo con dependencias reales
    se realiza en los notebooks de Colab (04_generacion.ipynb).

Autor:   Jesús Rodríguez
Versión: 1.0.0
"""

from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.rag import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_GENERATION,
    DEFAULT_TEMPERATURE,
    RAGPipeline,
    RAGResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_embedder():
    """Mock de DocumentEmbedder — evita cargar BGE-M3."""
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1] * 1024
    return embedder

@pytest.fixture
def mock_vector_store():
    """Mock de VectorStore con chunks y similitudes de ejemplo."""
    from src.retrieval import RetrievalResult

    store = MagicMock()
    store.search_filtered.return_value = RetrievalResult(
        chunks=[
            "## Introduction\n\nNHC complexes with Pd, Pt and Au metals.",
            "## Results\n\nThe synthesis of PdL3 was achieved via NHC coordination.",
        ],
        similitudes=[0.6227, 0.5969],
        metadatos=[
            {"fuente": "PMC10967698", "indice": 0, "longitud": 52},
            {"fuente": "PMC10967698", "indice": 1, "longitud": 57},
        ],
    )
    return store

@pytest.fixture
def mock_vector_store_vacio():
    """Mock de VectorStore que no devuelve chunks — simula umbral no superado."""
    from src.retrieval import RetrievalResult

    store = MagicMock()
    store.search_filtered.return_value = RetrievalResult(
        chunks=[],
        similitudes=[],
        metadatos=[],
    )
    return store

@pytest.fixture
def mock_respuesta_llm():
    """Mock de la respuesta de HF Inference API."""
    respuesta = MagicMock()
    respuesta.choices[0].message.content = (
        "According to the document, the NHC complexes studied involve "
        "palladium (Pd), platinum (Pt), and gold (Au) as the metal centers."
    )
    return respuesta

@pytest.fixture
def pipeline(mock_embedder, mock_vector_store, mock_respuesta_llm):
    """Instancia RAGPipeline con todas las dependencias mockeadas."""
    with patch("src.pipeline.rag.InferenceClient") as mock_cliente:
        mock_cliente.return_value.chat_completion.return_value = mock_respuesta_llm
        yield RAGPipeline(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            hf_token="token_test",
        )

# ---------------------------------------------------------------------------
# Tests del pipeline completo
# ---------------------------------------------------------------------------

class TestQuery:
    """Tests para el método query."""

    def test_query_devuelve_rag_result(self, pipeline):
        resultado = pipeline.query("What metals are used?")
        assert isinstance(resultado, RAGResult)

    def test_query_contiene_respuesta(self, pipeline):
        resultado = pipeline.query("What metals are used?")
        assert len(resultado.respuesta) > 0

    def test_query_registra_tiempo(self, pipeline):
        resultado = pipeline.query("What metals are used?")
        assert resultado.tiempo_total_s >= 0.0

    def test_query_registra_n_chunks(self, pipeline):
        resultado = pipeline.query("What metals are used?")
        assert resultado.n_chunks == 2

    def test_query_pregunta_vacia_lanza_error(self, pipeline):
        with pytest.raises(ValueError):
            pipeline.query("")

    def test_query_pregunta_solo_espacios_lanza_error(self, pipeline):
        with pytest.raises(ValueError):
            pipeline.query("   ")

    def test_query_preserva_pregunta_original(self, pipeline):
        pregunta  = "What metals are used in the study?"
        resultado = pipeline.query(pregunta)
        assert resultado.pregunta == pregunta

# ---------------------------------------------------------------------------
# Tests de contexto vacío
# ---------------------------------------------------------------------------

class TestContextoVacio:
    """Tests para el comportamiento cuando no hay chunks relevantes."""

    def test_contexto_vacio_devuelve_mensaje_informativo(
        self, mock_embedder, mock_vector_store_vacio, mock_respuesta_llm
    ):
        with patch("src.pipeline.rag.InferenceClient") as mock_cliente:
            mock_cliente.return_value.chat_completion.return_value = mock_respuesta_llm
            pipeline = RAGPipeline(
                vector_store=mock_vector_store_vacio,
                embedder=mock_embedder,
                hf_token="token_test",
            )
        resultado = pipeline.query("What metals are used?")
        assert "does not contain sufficient information" in resultado.respuesta

    def test_contexto_vacio_n_chunks_es_cero(
        self, mock_embedder, mock_vector_store_vacio, mock_respuesta_llm
    ):
        with patch("src.pipeline.rag.InferenceClient") as mock_cliente:
            mock_cliente.return_value.chat_completion.return_value = mock_respuesta_llm
            pipeline = RAGPipeline(
                vector_store=mock_vector_store_vacio,
                embedder=mock_embedder,
                hf_token="token_test",
            )
        resultado = pipeline.query("What metals are used?")
        assert resultado.n_chunks == 0

# ---------------------------------------------------------------------------
# Tests de configuración y manejo de errores
# ---------------------------------------------------------------------------

class TestConfiguracion:
    """Tests para la configuración del RAGPipeline."""

    def test_config_por_defecto(self, pipeline):
        assert pipeline.model_generation == DEFAULT_MODEL_GENERATION
        assert pipeline.max_tokens       == DEFAULT_MAX_TOKENS
        assert pipeline.temperature      == DEFAULT_TEMPERATURE

    def test_token_ausente_lanza_error(self, mock_embedder, mock_vector_store):
        with patch.dict("os.environ", {}, clear=True):
            # Eliminar HF_TOKEN del entorno para simular ausencia
            with patch("src.pipeline.rag.os.getenv", return_value=None):
                with pytest.raises(ValueError, match="HF_TOKEN"):
                    RAGPipeline(
                        vector_store=mock_vector_store,
                        embedder=mock_embedder,
                        hf_token=None,
                    )

    def test_token_como_parametro(self, mock_embedder, mock_vector_store):
        with patch("src.pipeline.rag.InferenceClient"):
            pipeline = RAGPipeline(
                vector_store=mock_vector_store,
                embedder=mock_embedder,
                hf_token="mi_token_directo",
            )
        assert pipeline.hf_token == "mi_token_directo"

# ---------------------------------------------------------------------------
# Tests de supresión del thinking mode
# ---------------------------------------------------------------------------

class TestThinkingMode:
    """Tests para la supresión del bloque <think> de Qwen3."""

    def test_elimina_bloque_think(
        self, mock_embedder, mock_vector_store
    ):
        # Simular respuesta de Qwen3 con bloque <think> activo
        respuesta_con_think = MagicMock()
        respuesta_con_think.choices[0].message.content = (
            "<think>Reasoning step...</think>\n"
            "The metals used are Pd, Pt and Au."
        )
        with patch("src.pipeline.rag.InferenceClient") as mock_cliente:
            mock_cliente.return_value.chat_completion.return_value = (
                respuesta_con_think
            )
            pipeline = RAGPipeline(
                vector_store=mock_vector_store,
                embedder=mock_embedder,
                hf_token="token_test",
            )
        resultado = pipeline.query("What metals are used?")
        assert "<think>" not in resultado.respuesta
        assert "The metals used are Pd, Pt and Au." in resultado.respuesta
