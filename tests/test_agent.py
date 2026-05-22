"""
test_agent.py

Tests unitarios para src/agent/graph.py

Cubre:
    - Ejecución del agente RAG completo (ruta paper_especifico)
    - Ruta general — respuesta sin recuperación de ChromaDB
    - Ruta fuera_dominio — respuesta directa sin contexto químico
    - Activación del Rewriter cuando los chunks no son relevantes
    - Límite de reintentos del Rewriter (máx. 2)
    - Validación de estructura del AgentResult
    - Manejo de errores (pregunta vacía, token ausente)
    - Supresión del bloque <think> de Qwen3

Nota sobre el uso de mock:
    Se mockean tres dependencias externas para aislar la lógica del agente:
      1. DocumentEmbedder: evita cargar BGE-M3 (~2.2 GB)
      2. VectorStore: evita necesitar ChromaDB inicializado
      3. InferenceClient: evita llamadas reales a HF Inference API
         que consumirían créditos y requieren conexión a internet
    Las respuestas secuenciales del LLM se inyectan via patch de
    _llamar_llm — función auxiliar interna que centraliza todas las
    llamadas al LLM, evitando el problema de timing con partial+compile.
    La validación del agente completo con dependencias reales
    se realiza en el notebook 07_agente.ipynb.

Autor:   Jesús Rodríguez
Versión: 1.0.0
"""

from unittest.mock import MagicMock, patch

import pytest

from src.agent.graph import (
    MAX_REINTENTOS,
    AgentGraph,
    AgentResult,
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
    """Mock de VectorStore con chunks de ejemplo."""
    from src.retrieval import RetrievalResult

    store = MagicMock()
    store.count.return_value = 77
    store.search.return_value = RetrievalResult(
        chunks=[
            "## Results\n\nThe yield of AuL9 was 47%.",
            "## Experimental\n\nAuL9 was synthesized using KAuCl4 and NaOAc.",
        ],
        similitudes=[0.7821, 0.6934],
        metadatos=[
            {"fuente": "PMC10967698", "indice": 29, "longitud": 38},
            {"fuente": "PMC10967698", "indice": 30, "longitud": 52},
        ],
    )
    return store


@pytest.fixture
def agente(mock_embedder, mock_vector_store):
    """Instancia AgentGraph con InferenceClient mockeado.

    El cliente se mockea en __init__ para evitar llamadas reales.
    Las respuestas del LLM se controlan en cada test via
    patch("src.agent.graph._llamar_llm").
    """
    with patch("src.agent.graph.InferenceClient"):
        yield AgentGraph(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            hf_token="token_test",
        )


# ---------------------------------------------------------------------------
# Función auxiliar para tests con secuencias de respuestas LLM
# ---------------------------------------------------------------------------

def _agente_con_secuencia_llm(
    mock_embedder,
    mock_vector_store,
    respuestas: list[str],
):
    """Instancia AgentGraph cuyo LLM devuelve respuestas en secuencia.

    Crea el agente con InferenceClient mockeado en __init__ para evitar
    llamadas reales. Las respuestas secuenciales se inyectan via patch
    de _llamar_llm — la función auxiliar que centraliza todas las llamadas
    al LLM en graph.py — evitando el problema de timing con partial+compile.

    Args:
        mock_embedder   : mock de DocumentEmbedder.
        mock_vector_store: mock de VectorStore.
        respuestas      : lista de strings — cada llamada a _llamar_llm
                          devuelve el siguiente elemento.

    Returns:
        Tupla (AgentGraph, respuestas) — las respuestas se usan como
        side_effect de _llamar_llm en el contexto del test.
    """
    with patch("src.agent.graph.InferenceClient"):
        ag = AgentGraph(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            hf_token="token_test",
        )
    return ag, respuestas


# ---------------------------------------------------------------------------
# Tests del agente completo — ruta paper_especifico
# ---------------------------------------------------------------------------

class TestRutaPaperEspecifico:
    """Tests para la ruta paper_especifico: Router → Retriever → Grader → Generator."""

    def test_query_devuelve_agent_result(self, agente):
        respuestas = ["paper_especifico", "yes", "The yield of AuL9 is 47%."]
        with patch("src.agent.graph._llamar_llm", side_effect=respuestas):
            resultado = agente.query("What is the yield of AuL9?")
        assert isinstance(resultado, AgentResult)

    def test_ruta_paper_especifico_llama_al_retriever(
        self, mock_embedder, mock_vector_store
    ):
        respuestas = [
            "paper_especifico",          # Router
            "yes",                       # Grader — chunks relevantes
            "The yield of AuL9 is 47%.", # Generator
        ]
        ag, resp = _agente_con_secuencia_llm(mock_embedder, mock_vector_store, respuestas)
        with patch("src.agent.graph._llamar_llm", side_effect=resp):
            ag.query("What is the yield of AuL9?")
        mock_vector_store.search.assert_called_once()

    def test_ruta_paper_especifico_registra_chunks(
        self, mock_embedder, mock_vector_store
    ):
        respuestas = ["paper_especifico", "yes", "The yield of AuL9 is 47%."]
        ag, resp = _agente_con_secuencia_llm(mock_embedder, mock_vector_store, respuestas)
        with patch("src.agent.graph._llamar_llm", side_effect=resp):
            resultado = ag.query("What is the yield of AuL9?")
        assert len(resultado.chunks_usados) == 2

    def test_ruta_paper_especifico_registra_ruta(
        self, mock_embedder, mock_vector_store
    ):
        respuestas = ["paper_especifico", "yes", "The yield of AuL9 is 47%."]
        ag, resp = _agente_con_secuencia_llm(mock_embedder, mock_vector_store, respuestas)
        with patch("src.agent.graph._llamar_llm", side_effect=resp):
            resultado = ag.query("What is the yield of AuL9?")
        assert resultado.ruta == "paper_especifico"


# ---------------------------------------------------------------------------
# Tests de rutas directas — general y fuera_dominio
# ---------------------------------------------------------------------------

class TestRutasDirectas:
    """Tests para rutas que no pasan por ChromaDB."""

    def test_ruta_general_no_llama_al_retriever(
        self, mock_embedder, mock_vector_store
    ):
        respuestas = [
            "general",                            # Router
            "An NHC is a stable carbene ligand.", # Generator
        ]
        ag, resp = _agente_con_secuencia_llm(mock_embedder, mock_vector_store, respuestas)
        with patch("src.agent.graph._llamar_llm", side_effect=resp):
            ag.query("What is an N-heterocyclic carbene?")
        mock_vector_store.search.assert_not_called()

    def test_ruta_fuera_dominio_no_llama_al_retriever(
        self, mock_embedder, mock_vector_store
    ):
        respuestas = [
            "fuera_dominio",                      # Router
            "The capital of Germany is Berlin.",  # Generator
        ]
        ag, resp = _agente_con_secuencia_llm(mock_embedder, mock_vector_store, respuestas)
        with patch("src.agent.graph._llamar_llm", side_effect=resp):
            ag.query("What is the capital of Germany?")
        mock_vector_store.search.assert_not_called()

    def test_ruta_general_devuelve_respuesta(
        self, mock_embedder, mock_vector_store
    ):
        respuestas = ["general", "An NHC is a stable carbene ligand."]
        ag, resp = _agente_con_secuencia_llm(mock_embedder, mock_vector_store, respuestas)
        with patch("src.agent.graph._llamar_llm", side_effect=resp):
            resultado = ag.query("What is an N-heterocyclic carbene?")
        assert len(resultado.respuesta) > 0
        assert resultado.ruta == "general"

    def test_ruta_fuera_dominio_cero_chunks(
        self, mock_embedder, mock_vector_store
    ):
        respuestas = ["fuera_dominio", "The capital of Germany is Berlin."]
        ag, resp = _agente_con_secuencia_llm(mock_embedder, mock_vector_store, respuestas)
        with patch("src.agent.graph._llamar_llm", side_effect=resp):
            resultado = ag.query("What is the capital of Germany?")
        assert len(resultado.chunks_usados) == 0


# ---------------------------------------------------------------------------
# Tests del Rewriter
# ---------------------------------------------------------------------------

class TestRewriter:
    """Tests para la activación y límite del Rewriter."""

    def test_rewriter_se_activa_cuando_chunks_no_relevantes(
        self, mock_embedder, mock_vector_store
    ):
        respuestas = [
            "paper_especifico",                      # Router
            "no",                                    # Grader — chunks no relevantes
            "NHC tetracarbene yield synthesis AuL9", # Rewriter
            "yes",                                   # Grader — segunda vez relevantes
            "The yield of AuL9 is 47%.",             # Generator
        ]
        ag, resp = _agente_con_secuencia_llm(mock_embedder, mock_vector_store, respuestas)
        with patch("src.agent.graph._llamar_llm", side_effect=resp):
            resultado = ag.query("What is the yield of AuL9?")
        assert resultado.intentos == 1

    def test_rewriter_no_supera_limite_max_reintentos(
        self, mock_embedder, mock_vector_store
    ):
        # Flujo con MAX_REINTENTOS=2:
        # Router → Retriever → Grader(no) → Rewriter → Retriever → Grader(no) → Generator
        # El segundo Grader(no) activa el Generator por límite de reintentos
        respuestas = [
            "paper_especifico",          # Router
            "no",                        # Grader intento 1 — no relevante, intentos=0 < 2 → Rewriter
            "query reescrita 1",         # Rewriter intento 1 — intentos=1
            "no",                        # Grader intento 2 — no relevante, intentos=1 < 2 → Rewriter
            "query reescrita 2",         # Rewriter intento 2 — intentos=2
            "no",                        # Grader intento 3 — intentos=2 >= 2 → Generator
            "The yield of AuL9 is 47%.", # Generator final
        ]
        ag, resp = _agente_con_secuencia_llm(mock_embedder, mock_vector_store, respuestas)
        with patch("src.agent.graph._llamar_llm", side_effect=resp):
            resultado = ag.query("What is the yield of AuL9?")
        assert resultado.intentos <= MAX_REINTENTOS

    def test_rewriter_incrementa_contador_intentos(
        self, mock_embedder, mock_vector_store
    ):
        respuestas = [
            "paper_especifico",
            "no",
            "NHC AuL9 yield synthesis",
            "yes",
            "The yield of AuL9 is 47%.",
        ]
        ag, resp = _agente_con_secuencia_llm(mock_embedder, mock_vector_store, respuestas)
        with patch("src.agent.graph._llamar_llm", side_effect=resp):
            resultado = ag.query("What is the yield?")
        assert resultado.intentos == 1


# ---------------------------------------------------------------------------
# Tests de estructura y validación
# ---------------------------------------------------------------------------

class TestEstructuraResultado:
    """Tests para la estructura del AgentResult."""

    def test_result_contiene_pregunta_original(
        self, mock_embedder, mock_vector_store
    ):
        respuestas = ["general", "An NHC is a stable carbene ligand."]
        ag, resp = _agente_con_secuencia_llm(mock_embedder, mock_vector_store, respuestas)
        pregunta = "What is an NHC?"
        with patch("src.agent.graph._llamar_llm", side_effect=resp):
            resultado = ag.query(pregunta)
        assert resultado.pregunta == pregunta

    def test_result_registra_tiempo(self, mock_embedder, mock_vector_store):
        respuestas = ["general", "An NHC is a stable carbene ligand."]
        ag, resp = _agente_con_secuencia_llm(mock_embedder, mock_vector_store, respuestas)
        with patch("src.agent.graph._llamar_llm", side_effect=resp):
            resultado = ag.query("What is an NHC?")
        assert resultado.tiempo_total_s >= 0.0

    def test_pregunta_vacia_lanza_error(self, agente):
        with pytest.raises(ValueError):
            agente.query("")

    def test_pregunta_solo_espacios_lanza_error(self, agente):
        with pytest.raises(ValueError):
            agente.query("   ")


# ---------------------------------------------------------------------------
# Tests de configuración y manejo de errores
# ---------------------------------------------------------------------------

class TestConfiguracion:
    """Tests para la configuración del AgentGraph."""

    def test_token_ausente_lanza_error(self, mock_embedder, mock_vector_store):
        with patch.dict("os.environ", {}, clear=True):
            with patch("src.agent.graph.os.getenv", return_value=None):
                with pytest.raises(ValueError, match="HF_TOKEN"):
                    AgentGraph(
                        vector_store=mock_vector_store,
                        embedder=mock_embedder,
                        hf_token=None,
                    )

    def test_token_como_parametro(self, mock_embedder, mock_vector_store):
        with patch("src.agent.graph.InferenceClient"):
            ag = AgentGraph(
                vector_store=mock_vector_store,
                embedder=mock_embedder,
                hf_token="mi_token_directo",
            )
        assert ag._cliente_llm is not None


# ---------------------------------------------------------------------------
# Tests de supresión del thinking mode
# ---------------------------------------------------------------------------

class TestThinkingMode:
    """Tests para la supresión del bloque <think> de Qwen3."""

    def test_elimina_bloque_think_en_respuesta(self, mock_embedder, mock_vector_store):
        """Verifica que _llamar_llm elimina el bloque <think> de Qwen3.

        Se testea _llamar_llm directamente porque mockear la función completa
        bypassea el re.sub interno.
        """
        from src.agent.graph import _llamar_llm

        mock_cliente = MagicMock()
        mock_cliente.chat.completions.create.return_value.choices[
            0
        ].message.content = (
            "<think>Reasoning step...</think>\nAn NHC is a stable carbene ligand."
        )

        resultado = _llamar_llm(
            mock_cliente,
            [{"role": "user", "content": "What is an NHC?"}],
        )

        assert "<think>" not in resultado
        assert "An NHC is a stable carbene ligand." in resultado