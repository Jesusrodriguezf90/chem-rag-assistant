"""
rag.py

Módulo de orquestación del pipeline RAG completo.

Responsabilidad:
    Integrar las cuatro etapas del pipeline RAG en una única función
    de alto nivel: recuperación semántica con VectorStore + generación
    de respuestas fundamentadas con Qwen3 vía HF Inference API.

    Aplica:
      - Filtrado por umbral mínimo de similitud antes de construir el prompt
      - Prompt con role prompting, structured prompting (XML) y one-shot
      - Supresión del thinking mode de Qwen3 con /no_think
      - Respuesta en el idioma de la pregunta (detección automática)
      - Registro del tiempo de ejecución end-to-end

    Este módulo es la última etapa del pipeline RAG y su output
    es la entrada de app/routers/query.py (FastAPI).

Notas para migración a producción:
    - Añadir try/except con logging para trazabilidad de cada llamada
    - Añadir caché de respuestas para preguntas frecuentes
    - Parametrizar el system prompt desde config/

Autor:   Jesús Rodríguez
Versión: 1.0.0
"""

import os
import re
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from src.embeddings import DocumentEmbedder
from src.retrieval import VectorStore

# ---------------------------------------------------------------------------
# Carga de variables de entorno
# ---------------------------------------------------------------------------

load_dotenv()

# ---------------------------------------------------------------------------
# Constantes por defecto
# ---------------------------------------------------------------------------

DEFAULT_MODEL_GENERATION = "Qwen/Qwen3-8B"
DEFAULT_MAX_TOKENS       = 512
DEFAULT_TEMPERATURE      = 0.1  # Baja para respuestas deterministas en QA científico

# System prompt con tres técnicas de prompt engineering:
#   - Role prompting: define el dominio de expertise del modelo
#   - Structured prompting con etiquetas XML: separa instrucciones y ejemplo
#   - One-shot prompting: un ejemplo ancla el formato y estilo de citación
# Define además: comportamiento ante información insuficiente,
# instrucciones de citación, idioma dinámico y supresión de thinking mode
# mediante /no_think al final del mensaje de usuario
SYSTEM_PROMPT = """\
You are a specialized scientific assistant in organometallic chemistry \
and medicinal inorganic chemistry. Your role is to answer questions \
accurately and concisely based exclusively on the provided document context.

<guidelines>
- Answer ONLY based on the information present in the provided context.
- If the context does not contain enough information to answer the \
question, explicitly state: "The provided document does not contain \
sufficient information to answer this question."
- Do NOT speculate or introduce information not present in the context.
- First cite the relevant fragment that supports your answer, \
then provide your conclusion.
- Use precise scientific terminology consistent with the document.
- Respond in the same language as the user's question.
- Keep your answer focused and concise (maximum 3-4 paragraphs).
</guidelines>

<example>
Question: What ligand type is used in the complexes?
Context fragment: "N-heterocyclic carbenes (NHCs) are used as ligands \
in the synthesis of Pd(II) and Pt(II) complexes."
Answer: According to the document, the complexes use \
N-heterocyclic carbenes (NHCs) as ligands, specifically in \
the synthesis of Pd(II) and Pt(II) complexes.
</example>
"""

# ---------------------------------------------------------------------------
# Dataclass de resultado del pipeline
# ---------------------------------------------------------------------------

@dataclass
class RAGResult:
    """Resultado completo del pipeline RAG.

    Attributes:
        pregunta: consulta original del usuario.
        respuesta: respuesta generada por el LLM.
        chunks_usados: fragmentos del documento usados como contexto.
        similitudes: puntuaciones de similitud de cada chunk usado.
        n_chunks: número de chunks que superaron el umbral de similitud.
        tiempo_total_s: tiempo de ejecución end-to-end en segundos.
    """
    pregunta      : str
    respuesta     : str
    chunks_usados : list[str]
    similitudes   : list[float]
    n_chunks      : int
    tiempo_total_s: float

# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class RAGPipeline:
    """Orquesta el pipeline RAG completo: recuperación + generación.

    Integra VectorStore para la recuperación semántica y Qwen3
    vía HF Inference API para la generación de respuestas.

    Attributes:
        vector_store: instancia de VectorStore con el índice cargado.
        embedder: instancia de DocumentEmbedder para vectorizar consultas.
        hf_token: token de autenticación para HF Inference API.
        model_generation: nombre del modelo de generación en HuggingFace.
        max_tokens: límite de tokens en la respuesta generada.
        temperature: temperatura de muestreo del LLM.
        _cliente_llm: instancia de InferenceClient de HuggingFace.
    """

    def __init__(
        self,
        vector_store    : VectorStore,
        embedder        : DocumentEmbedder,
        hf_token        : str | None = None,
        model_generation: str        = DEFAULT_MODEL_GENERATION,
        max_tokens      : int        = DEFAULT_MAX_TOKENS,
        temperature     : float      = DEFAULT_TEMPERATURE,
    ) -> None:
        """Inicializa el pipeline RAG.

        Args:
            vector_store: instancia de VectorStore ya indexada.
            embedder: instancia de DocumentEmbedder para vectorizar consultas.
                      Debe ser el mismo modelo usado en la indexación.
            hf_token: token HF para la Inference API. Si es None,
                      se lee de la variable de entorno HF_TOKEN.
            model_generation: modelo LLM a usar para la generación.
            max_tokens: límite de tokens en la respuesta.
            temperature: temperatura del LLM (0.0-1.0).

        Raises:
            ValueError: si no se encuentra el token HF_TOKEN.
        """
        self.vector_store     = vector_store
        self.embedder         = embedder
        self.model_generation = model_generation
        self.max_tokens       = max_tokens
        self.temperature      = temperature

        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise ValueError(
                "HF_TOKEN no encontrado. "
                "Configúralo en el archivo .env o pásalo como parámetro."
            )

        self._cliente_llm = InferenceClient(
            model=self.model_generation,
            token=self.hf_token,
        )

    # -----------------------------------------------------------------------
    # API pública
    # -----------------------------------------------------------------------

    def query(self, pregunta: str) -> RAGResult:
        """Ejecuta el pipeline RAG completo para una pregunta dada.

        Args:
            pregunta: consulta del usuario en cualquier idioma.

        Returns:
            RAGResult con la respuesta, chunks usados y métricas.

        Raises:
            ValueError: si la pregunta está vacía.
        """
        if not pregunta.strip():
            raise ValueError("La pregunta no puede estar vacía.")

        inicio = time.time()

        # Fase 1: Recuperación y filtrado por umbral de similitud
        chunks_filtrados, similitudes = self._recuperar(pregunta)

        # Fase 2: Generación de respuesta fundamentada
        respuesta = self._generar(pregunta, chunks_filtrados, similitudes)

        return RAGResult(
            pregunta      =pregunta,
            respuesta     =respuesta,
            chunks_usados =chunks_filtrados,
            similitudes   =similitudes,
            n_chunks      =len(chunks_filtrados),
            tiempo_total_s=round(time.time() - inicio, 2),
        )

    # -----------------------------------------------------------------------
    # Métodos privados
    # -----------------------------------------------------------------------

    def _recuperar(
        self, pregunta: str
    ) -> tuple[list[str], list[float]]:
        """Recupera chunks relevantes usando el vector store.

        Genera el embedding de la consulta con el mismo modelo usado
        en la indexación para garantizar coherencia del espacio semántico.

        Args:
            pregunta: consulta del usuario.

        Returns:
            Tupla (chunks_filtrados, similitudes).
        """
        query_vector = self.embedder.embed_query(pregunta)
        resultado    = self.vector_store.search_filtered(query_vector)
        return resultado.chunks, resultado.similitudes

    def _generar(
        self,
        pregunta       : str,
        chunks_contexto: list[str],
        similitudes    : list[float],
    ) -> str:
        """Genera una respuesta fundamentada en el contexto recuperado.

        Args:
            pregunta: consulta del usuario.
            chunks_contexto: chunks recuperados y filtrados.
            similitudes: puntuaciones de similitud de cada chunk.

        Returns:
            Respuesta generada en el idioma de la pregunta,
            sin el bloque <think> de Qwen3.
        """
        # Si ningún chunk supera el umbral informar al usuario directamente
        # sin consumir tokens de la API de generación
        if not chunks_contexto:
            return (
                "The provided document does not contain sufficient "
                "information to answer this question with the required "
                "confidence level."
            )

        # Construir el contexto numerando cada chunk con su similitud
        contexto_formateado = "\n\n".join([
            f"[Fragmento {i + 1} | similitud: {sim:.4f}]\n{chunk}"
            for i, (chunk, sim) in enumerate(zip(chunks_contexto, similitudes))
        ])

        # /no_think suprime el modo de razonamiento interno de Qwen3
        # evitando el bloque <think>...</think> en la respuesta,
        # reduciendo la latencia de ~8s a ~3s por consulta
        mensaje_usuario = (
            f"Context from the scientific document:\n\n"
            f"{contexto_formateado}\n\n"
            f"Question: {pregunta}\n\n"
            f"/no_think"
        )

        respuesta = self._cliente_llm.chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": mensaje_usuario},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        contenido = respuesta.choices[0].message.content

        # Filtrado defensivo del bloque <think>...</think>
        # como salvaguarda en caso de que /no_think no sea respetado
        contenido_limpio = re.sub(
            r"<think>.*?</think>\s*",
            "",
            contenido,
            flags=re.DOTALL,
        ).strip()

        return contenido_limpio
