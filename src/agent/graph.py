"""
graph.py

Módulo del agente RAG con LangGraph — Adaptive RAG con self-correction.

Responsabilidad:
    Implementar el grafo de agente RAG que supera las limitaciones del
    pipeline RAG lineal de src/pipeline/rag.py. A diferencia del pipeline
    clásico (recuperar → generar, siempre), este agente decide de forma
    autónoma en cada consulta la ruta óptima:

      - paper_especifico: requiere información específica del paper
                          → ChromaDB → Grader → Generator
      - general         : respondible con conocimiento de química general
                          → Generator directamente
      - fuera_dominio   : no relacionada con química ni el paper
                          → Generator directamente

    Si los chunks recuperados no son relevantes, el Rewriter reformula
    la query y reintenta la recuperación (máx. 2 intentos).

    Patrón implementado:
        Adaptive RAG + Corrective RAG — estándar de la industria en 2026
        para sistemas RAG de producción con self-correction.

    Arquitectura del grafo:

        Query
          ↓
        Router  (LLM clasifica: paper_especifico / general / fuera_dominio)
          ↓                    ↓                        ↓
        Retriever          Generator               Generator
          ↓               (conocimiento            (respuesta
        Grader              general)                directa)
          ↓ no relevante  ↓ relevante
        Rewriter         Generator
          ↓
        Retriever (máx. 2 intentos)

    Relación con el pipeline RAG clásico:
        Este módulo reutiliza la misma instancia de VectorStore y
        DocumentEmbedder inicializadas en app/state.py — no crea
        dependencias nuevas. El Router es la diferencia fundamental:
        el pipeline clásico siempre recupera, el agente decide si
        recuperar según la naturaleza de la query.

Notas para migración a producción:
    - Añadir logging estructurado por nodo para trazabilidad
    - Añadir memoria de conversación con LangGraph checkpointer
    - Parametrizar MAX_REINTENTOS y MODELO_LLM desde config/

Autor:   Jesús Rodríguez
Versión: 1.0.0
"""

import os
import re
import time
from dataclasses import dataclass
from typing import Literal, TypedDict

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from src.embeddings import DocumentEmbedder
from src.retrieval import VectorStore

# ---------------------------------------------------------------------------
# Carga de variables de entorno
# ---------------------------------------------------------------------------

load_dotenv()

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DEFAULT_MODEL_LLM  = "Qwen/Qwen3-8B"
MAX_REINTENTOS     = 2       # Máximo de intentos del Rewriter antes de generar
TOP_K_RETRIEVER    = 4       # Coherente con DEFAULT_TOP_K de VectorStore

# ---------------------------------------------------------------------------
# Definición del estado del grafo
# ---------------------------------------------------------------------------

class EstadoAgente(TypedDict):
    """Estado compartido entre todos los nodos del grafo LangGraph.

    LangGraph pasa este estado automáticamente entre nodos — cada nodo
    devuelve solo los campos que modifica y el resto se mantiene intacto.

    Attributes:
        pregunta         : query original del usuario — no se modifica.
        query_reescrita  : query reformulada por el Rewriter si los
                           chunks no son relevantes.
        ruta             : decisión del Router (paper_especifico /
                           general / fuera_dominio).
        chunks           : documentos recuperados de ChromaDB.
        chunks_relevantes: resultado del Grader (True / False).
        intentos         : número de intentos de recuperación realizados.
        respuesta        : respuesta final generada.
        tiempo_total_s   : tiempo de ejecución end-to-end en segundos.
    """
    pregunta          : str
    query_reescrita   : str
    ruta              : str
    chunks            : list
    chunks_relevantes : bool
    intentos          : int
    respuesta         : str
    tiempo_total_s    : float


# ---------------------------------------------------------------------------
# Dataclass de resultado del agente
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """Resultado completo del agente RAG.

    Equivalente a RAGResult de src/pipeline/rag.py pero con los
    campos adicionales del agente: ruta e intentos de reescritura.

    Attributes:
        pregunta      : consulta original del usuario.
        respuesta     : respuesta generada por el LLM.
        ruta          : ruta seguida por el agente.
        chunks_usados : fragmentos del documento usados como contexto.
        intentos      : número de intentos de recuperación realizados.
        tiempo_total_s: tiempo de ejecución end-to-end en segundos.
    """
    pregunta      : str
    respuesta     : str
    ruta          : str
    chunks_usados : list[str]
    intentos      : int
    tiempo_total_s: float


# ---------------------------------------------------------------------------
# Función auxiliar LLM
# ---------------------------------------------------------------------------

def _llamar_llm(
    cliente   : InferenceClient,
    mensajes  : list[dict],
    max_tokens: int = 300,
    model     : str = DEFAULT_MODEL_LLM,
) -> str:
    """Llama a Qwen3-8B vía HF Inference API suprimiendo el thinking mode.

    Función auxiliar compartida por todos los nodos que necesitan LLM.
    Añade /no_think al último mensaje de usuario para reducir la latencia
    de ~8s a ~3s, coherente con src/pipeline/rag.py.

    Args:
        cliente   : instancia de InferenceClient.
        mensajes  : lista de dicts con role y content.
        max_tokens: máximo de tokens a generar.
        model     : modelo LLM a usar.

    Returns:
        Texto generado limpio sin bloque <think>.
    """
    mensajes_mod = mensajes.copy()
    for i in range(len(mensajes_mod) - 1, -1, -1):
        if mensajes_mod[i]["role"] == "user":
            mensajes_mod[i] = {
                **mensajes_mod[i],
                "content": mensajes_mod[i]["content"] + " /no_think",
            }
            break

    respuesta = cliente.chat.completions.create(
        model=model,
        messages=mensajes_mod,
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.8,
    )
    texto = respuesta.choices[0].message.content

    # Filtrado defensivo del bloque <think> — coherente con rag.py
    return re.sub(
        r"<think>.*?</think>\s*",
        "",
        texto,
        flags=re.DOTALL,
    ).strip()


# ---------------------------------------------------------------------------
# Nodos del grafo
# ---------------------------------------------------------------------------

def _nodo_router(
    estado : EstadoAgente,
    cliente: InferenceClient,
) -> EstadoAgente:
    """Nodo Router — clasifica la query para decidir la ruta del grafo.

    Usa el LLM para determinar si la pregunta requiere información
    específica del paper, puede responderse con conocimiento general
    de química, o está fuera del dominio.

    Args:
        estado : estado actual del grafo.
        cliente: instancia de InferenceClient.

    Returns:
        Estado actualizado con 'ruta' e 'intentos' asignados.
    """
    respuesta = _llamar_llm(
        cliente,
        [
            {
                "role": "system",
                "content": (
                    "You are a query classifier for a RAG system about a specific "
                    "scientific paper on organometallic chemistry (NHC complexes of "
                    "Pd, Pt and Au, PMC10967698).\n"
                    "Classify the query into exactly one of these three categories:\n"
                    "- paper_especifico: requires specific data from the paper "
                    "(yields, compounds, experimental results, specific values)\n"
                    "- general: can be answered with general chemistry knowledge "
                    "(concepts, definitions, mechanisms)\n"
                    "- fuera_dominio: unrelated to chemistry or the paper\n"
                    "Reply with ONLY one of: paper_especifico, general, fuera_dominio"
                ),
            },
            {
                "role": "user",
                "content": f"Query: {estado['pregunta']}",
            },
        ],
        max_tokens=10,
    )

    # Normalizar la respuesta — valor por defecto conservador
    ruta = "paper_especifico"
    for opcion in ["paper_especifico", "general", "fuera_dominio"]:
        if opcion in respuesta.lower():
            ruta = opcion
            break

    return {**estado, "ruta": ruta, "intentos": 0}


def _nodo_retriever(
    estado     : EstadoAgente,
    embedder   : DocumentEmbedder,
    vector_store: VectorStore,
) -> EstadoAgente:
    """Nodo Retriever — recupera los chunks más relevantes de ChromaDB.

    Usa la query reescrita si existe (tras un intento fallido del Grader),
    o la query original en el primer intento. Reutiliza la misma instancia
    de VectorStore y DocumentEmbedder del pipeline RAG clásico.

    Args:
        estado      : estado actual del grafo.
        embedder    : instancia de DocumentEmbedder.
        vector_store: instancia de VectorStore.

    Returns:
        Estado actualizado con 'chunks' asignados.
    """
    query        = estado.get("query_reescrita") or estado["pregunta"]
    query_vector = embedder.embed_query(query)
    resultado    = vector_store.search(query_vector)

    return {**estado, "chunks": resultado.chunks}


def _nodo_grader(
    estado : EstadoAgente,
    cliente: InferenceClient,
) -> EstadoAgente:
    """Nodo Grader — evalúa si los chunks recuperados son relevantes.

    Es el paso clave del Corrective RAG — si los chunks no contienen
    información útil activa el Rewriter en lugar de generar una
    respuesta potencialmente incorrecta.

    Args:
        estado : estado actual del grafo.
        cliente: instancia de InferenceClient.

    Returns:
        Estado actualizado con 'chunks_relevantes' (True/False).
    """
    chunks_texto = "\n---\n".join(estado["chunks"])

    respuesta = _llamar_llm(
        cliente,
        [
            {
                "role": "system",
                "content": (
                    "You are a relevance grader. Given a question and retrieved "
                    "document chunks, determine if the chunks contain enough "
                    "information to answer the question accurately.\n"
                    "Reply with ONLY: yes or no"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {estado['pregunta']}\n\n"
                    f"Chunks:\n{chunks_texto[:1500]}"
                ),
            },
        ],
        max_tokens=5,
    )

    relevantes = "yes" in respuesta.lower()
    return {**estado, "chunks_relevantes": relevantes}


def _nodo_rewriter(
    estado : EstadoAgente,
    cliente: InferenceClient,
) -> EstadoAgente:
    """Nodo Rewriter — reformula la query para mejorar la recuperación.

    Se activa cuando el Grader determina que los chunks no son relevantes.
    Añade terminología técnica del dominio NHC para mejorar la similitud
    semántica con los chunks indexados.

    Args:
        estado : estado actual del grafo.
        cliente: instancia de InferenceClient.

    Returns:
        Estado actualizado con 'query_reescrita' e 'intentos' incrementado.
    """
    respuesta = _llamar_llm(
        cliente,
        [
            {
                "role": "system",
                "content": (
                    "You are a query rewriter for a RAG system about organometallic "
                    "chemistry (NHC complexes, Pd/Pt/Au tetracarbenes, apoptosis).\n"
                    "Rewrite the query to improve retrieval from a scientific paper.\n"
                    "Add relevant technical terms (NHC, carbene, ligand, complex, "
                    "synthesis, characterization, apoptosis) when appropriate.\n"
                    "Return ONLY the rewritten query, nothing else."
                ),
            },
            {
                "role": "user",
                "content": f"Original query: {estado['pregunta']}",
            },
        ],
        max_tokens=80,
    )

    intentos = estado.get("intentos", 0) + 1
    return {**estado, "query_reescrita": respuesta, "intentos": intentos}


def _nodo_generator(
    estado : EstadoAgente,
    cliente: InferenceClient,
) -> EstadoAgente:
    """Nodo Generator — genera la respuesta final adaptando el prompt.

    Se invoca en tres escenarios con prompts distintos:
      1. Chunks relevantes disponibles → respuesta fundamentada en el paper
      2. Ruta 'general'               → respuesta con conocimiento de química
      3. Ruta 'fuera_dominio'         → respuesta directa sin contexto químico

    Usa el mismo SYSTEM_PROMPT estructurado de src/pipeline/rag.py
    cuando hay contexto del paper para mantener coherencia de estilo.

    Args:
        estado : estado actual del grafo.
        cliente: instancia de InferenceClient.

    Returns:
        Estado actualizado con 'respuesta' y 'tiempo_total_s' asignados.
    """
    from src.pipeline.rag import SYSTEM_PROMPT  # reutilizar prompt de producción

    ruta   = estado.get("ruta", "paper_especifico")
    chunks = estado.get("chunks", [])

    if ruta == "fuera_dominio":
        system_msg   = "You are a helpful assistant. Answer the question directly."
        user_content = estado["pregunta"]
    elif ruta == "general" or not chunks:
        system_msg = (
            "You are a specialized scientific assistant in organometallic "
            "chemistry and medicinal inorganic chemistry. "
            "Answer accurately using your knowledge of NHC metal complexes."
        )
        user_content = estado["pregunta"]
    else:
        # Respuesta fundamentada en chunks — mismo formato que rag.py
        system_msg        = SYSTEM_PROMPT
        contexto_formateado = "\n\n".join([
            f"[Fragmento {i + 1}]\n{chunk}"
            for i, chunk in enumerate(chunks)
        ])
        user_content = (
            f"Context from the scientific document:\n\n"
            f"{contexto_formateado}\n\n"
            f"Question: {estado['pregunta']}"
        )

    respuesta = _llamar_llm(
        cliente,
        [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_content},
        ],
        max_tokens=512,
    )

    return {**estado, "respuesta": respuesta}


# ---------------------------------------------------------------------------
# Funciones de enrutamiento condicional
# ---------------------------------------------------------------------------

def _decidir_tras_router(
    estado: EstadoAgente,
) -> Literal["_nodo_retriever", "_nodo_generator"]:
    """Edge condicional tras el Router.

    Enruta a Retriever si la query es específica del paper.
    Enruta directamente al Generator en cualquier otro caso.
    """
    if estado["ruta"] == "paper_especifico":
        return "_nodo_retriever"
    return "_nodo_generator"


def _decidir_tras_grader(
    estado: EstadoAgente,
) -> Literal["_nodo_generator", "_nodo_rewriter"]:
    """Edge condicional tras el Grader.

    Enruta al Generator si los chunks son relevantes.
    Enruta al Rewriter si no son relevantes y quedan intentos.
    Enruta al Generator si se alcanzó el límite de reintentos —
    evita bucles infinitos generando con el contexto disponible.
    """
    if estado["chunks_relevantes"]:
        return "_nodo_generator"
    if estado.get("intentos", 0) < MAX_REINTENTOS:
        return "_nodo_rewriter"
    return "_nodo_generator"


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class AgentGraph:
    """Agente RAG con LangGraph — Adaptive RAG con self-correction.

    Encapsula la construcción, compilación y ejecución del grafo LangGraph.
    Reutiliza las mismas dependencias del pipeline RAG clásico (VectorStore,
    DocumentEmbedder) sin crear instancias adicionales.

    El patrón de inicialización es coherente con RAGPipeline en rag.py:
    recibe las dependencias ya inicializadas en lugar de crearlas internamente,
    lo que facilita el testing y evita duplicar la carga de modelos pesados.

    Attributes:
        vector_store: instancia de VectorStore con el índice cargado.
        embedder    : instancia de DocumentEmbedder para vectorizar queries.
        _cliente_llm: instancia de InferenceClient para llamadas al LLM.
        _grafo      : grafo LangGraph compilado y listo para ejecutar.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder    : DocumentEmbedder,
        hf_token    : str | None = None,
        model_llm   : str        = DEFAULT_MODEL_LLM,
    ) -> None:
        """Inicializa el agente y compila el grafo LangGraph.

        Args:
            vector_store: instancia de VectorStore ya indexada.
            embedder    : instancia de DocumentEmbedder para vectorizar queries.
                          Debe ser el mismo modelo usado en la indexación.
            hf_token    : token HF para la Inference API. Si es None,
                          se lee de la variable de entorno HF_TOKEN.
            model_llm   : modelo LLM a usar para todos los nodos.

        Raises:
            ValueError: si no se encuentra el token HF_TOKEN.
        """
        self.vector_store = vector_store
        self.embedder     = embedder

        token = hf_token or os.getenv("HF_TOKEN")
        if not token:
            raise ValueError(
                "HF_TOKEN no encontrado. "
                "Configúralo en el archivo .env o pásalo como parámetro."
            )

        self._cliente_llm = InferenceClient(
            provider="auto",
            api_key=token,
        )
        self._model_llm = model_llm
        self._grafo     = self._compilar_grafo()

    # -----------------------------------------------------------------------
    # API pública
    # -----------------------------------------------------------------------

    def query(self, pregunta: str) -> AgentResult:
        """Ejecuta el agente RAG completo para una pregunta dada.

        Args:
            pregunta: consulta del usuario en cualquier idioma.

        Returns:
            AgentResult con la respuesta, ruta seguida y métricas.

        Raises:
            ValueError: si la pregunta está vacía.
        """
        if not pregunta.strip():
            raise ValueError("La pregunta no puede estar vacía.")

        inicio = time.time()

        estado_inicial: EstadoAgente = {
            "pregunta"         : pregunta,
            "query_reescrita"  : "",
            "ruta"             : "",
            "chunks"           : [],
            "chunks_relevantes": False,
            "intentos"         : 0,
            "respuesta"        : "",
            "tiempo_total_s"   : 0.0,
        }

        estado_final = self._grafo.invoke(estado_inicial)

        return AgentResult(
            pregunta      =estado_final["pregunta"],
            respuesta     =estado_final["respuesta"],
            ruta          =estado_final["ruta"],
            chunks_usados =estado_final["chunks"],
            intentos      =estado_final["intentos"],
            tiempo_total_s=round(time.time() - inicio, 2),
        )

    # -----------------------------------------------------------------------
    # Métodos privados
    # -----------------------------------------------------------------------

    def _compilar_grafo(self):
        """Construye y compila el grafo LangGraph.

        Los nodos son funciones parciales que capturan las dependencias
        (cliente LLM, embedder, vector_store) sin exponerlas en la firma
        del grafo — LangGraph solo pasa el estado entre nodos.

        Returns:
            Grafo LangGraph compilado y listo para invocar.
        """
        from functools import partial
        from langgraph.graph import StateGraph, END

        grafo = StateGraph(EstadoAgente)

        # Nodos con dependencias inyectadas via partial
        grafo.add_node(
            "_nodo_router",
            partial(_nodo_router, cliente=self._cliente_llm),
        )
        grafo.add_node(
            "_nodo_retriever",
            partial(
                _nodo_retriever,
                embedder=self.embedder,
                vector_store=self.vector_store,
            ),
        )
        grafo.add_node(
            "_nodo_grader",
            partial(_nodo_grader, cliente=self._cliente_llm),
        )
        grafo.add_node(
            "_nodo_rewriter",
            partial(_nodo_rewriter, cliente=self._cliente_llm),
        )
        grafo.add_node(
            "_nodo_generator",
            partial(_nodo_generator, cliente=self._cliente_llm),
        )

        # Punto de entrada
        grafo.set_entry_point("_nodo_router")

        # Edges fijos
        grafo.add_edge("_nodo_retriever", "_nodo_grader")
        grafo.add_edge("_nodo_rewriter",  "_nodo_retriever")
        grafo.add_edge("_nodo_generator", END)

        # Edges condicionales
        grafo.add_conditional_edges(
            "_nodo_router",
            _decidir_tras_router,
            {
                "_nodo_retriever": "_nodo_retriever",
                "_nodo_generator": "_nodo_generator",
            },
        )
        grafo.add_conditional_edges(
            "_nodo_grader",
            _decidir_tras_grader,
            {
                "_nodo_generator": "_nodo_generator",
                "_nodo_rewriter" : "_nodo_rewriter",
            },
        )

        return grafo.compile()
