"""
chunker.py

Módulo de división y procesamiento de texto en chunks para el pipeline RAG.

Responsabilidad:
    Recibir el texto limpio en Markdown devuelto por loader.py y dividirlo
    en chunks semánticos usando la estrategia Contextual Chunk Headers,
    seleccionada como óptima en el notebook 02b_chunking_eval.ipynb
    (puntuación 0.5729 vs 0.5617 del baseline RecursiveCharacterTextSplitter).

    El proceso incluye:
      1. División con RecursiveCharacterTextSplitter
      2. Fusión de títulos de sección aislados con su contenido
      3. Anteposición del título de sección activo a cada chunk

    Este módulo es la segunda etapa del pipeline RAG y su output
    es la entrada de src/embeddings/embedder.py.

Autor:   Jesús Rodríguez
Versión: 1.0.0
"""

import re
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Constantes por defecto
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE    = 1000
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_MIN_LENGTH    = 100

# Separadores en orden de prioridad — respetan la estructura Markdown del paper
DEFAULT_SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", " "]

# ---------------------------------------------------------------------------
# Dataclass de configuración
# ---------------------------------------------------------------------------

@dataclass
class ChunkConfig:
    """Parámetros de configuración para el chunker.

    Attributes:
        chunk_size: número máximo de caracteres por chunk.
        chunk_overlap: solapamiento en caracteres entre chunks consecutivos.
        min_length: longitud mínima de un chunk. Los chunks por debajo
                    se fusionan con el siguiente para preservar contexto.
        separators: lista de separadores en orden de prioridad.
    """
    chunk_size   : int       = DEFAULT_CHUNK_SIZE
    chunk_overlap: int       = DEFAULT_CHUNK_OVERLAP
    min_length   : int       = DEFAULT_MIN_LENGTH
    separators   : list[str] = field(default_factory=lambda: DEFAULT_SEPARATORS.copy())

# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class DocumentChunker:
    """Divide texto Markdown en chunks contextuales para indexación RAG.

    Implementa la estrategia Contextual Chunk Headers: divide el texto
    con RecursiveCharacterTextSplitter, fusiona títulos de sección
    aislados con su contenido y antepone el título de sección activo
    a cada chunk para preservar el contexto estructural del documento.

    Attributes:
        config: parámetros de chunking.
        _splitter: instancia de RecursiveCharacterTextSplitter.
    """

    def __init__(self, config: ChunkConfig | None = None) -> None:
        """Inicializa el chunker con la configuración proporcionada.

        Args:
            config: parámetros de chunking. Si es None, usa los valores
                    por defecto definidos en ChunkConfig.
        """
        self.config   = config or ChunkConfig()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
        )

    # -----------------------------------------------------------------------
    # API pública
    # -----------------------------------------------------------------------

    def split(self, texto: str) -> list[str]:
        """Divide el texto en chunks contextuales.

        Ejecuta el pipeline completo: split → fusión → contexto.

        Args:
            texto: texto en Markdown limpio, output de DocumentLoader.

        Returns:
            Lista de chunks con contexto de sección anteponido,
            listos para ser vectorizados por DocumentEmbedder.

        Raises:
            ValueError: si el texto de entrada está vacío.
        """
        if not texto.strip():
            raise ValueError("El texto de entrada no puede estar vacío.")

        chunks_raw        = self._splitter.split_text(texto)
        chunks_fusionados = self._fusionar_chunks_cortos(chunks_raw)
        chunks_finales    = self._añadir_contexto_seccion(chunks_fusionados, texto)

        return chunks_finales

    # -----------------------------------------------------------------------
    # Métodos privados
    # -----------------------------------------------------------------------

    def _fusionar_chunks_cortos(self, chunks: list[str]) -> list[str]:
        """Fusiona chunks por debajo de la longitud mínima con el siguiente.

        Los chunks cortos suelen ser títulos de sección (## ...) que el
        splitter separó del contenido. Fusionarlos preserva el contexto
        semántico del encabezado junto a su contenido.

        Args:
            chunks: lista de chunks generados por el splitter.

        Returns:
            Lista de chunks con títulos fusionados a su contenido.
        """
        # Filtrar chunks vacíos antes de procesar para evitar
        # contaminación del índice vectorial
        chunks_validos = [c for c in chunks if c.strip()]

        chunks_fusionados = []
        i = 0
        while i < len(chunks_validos):
            if (
                len(chunks_validos[i]) < self.config.min_length
                and i + 1 < len(chunks_validos)
            ):
                chunks_fusionados.append(
                    chunks_validos[i] + "\n" + chunks_validos[i + 1]
                )
                i += 2
            else:
                chunks_fusionados.append(chunks_validos[i])
                i += 1

        return chunks_fusionados

    def _extraer_seccion_activa(self, chunk: str, documento: str) -> str:
        """Extrae el título de sección (##) más reciente antes del chunk.

        Args:
            chunk: texto del chunk.
            documento: documento completo del que proviene el chunk.

        Returns:
            Título de sección más reciente o cadena vacía si no se encuentra.
        """
        pos = documento.find(chunk[:100])
        if pos == -1:
            return ""
        fragmento_previo = documento[:pos]
        secciones = re.findall(r"^##[^#].*$", fragmento_previo, re.MULTILINE)
        return secciones[-1].strip() if secciones else ""

    def _añadir_contexto_seccion(
        self,
        chunks: list[str],
        documento: str,
    ) -> list[str]:
        """Antepone el título de sección activo a cada chunk.

        Implementación de Contextual Chunk Headers: antepone el título ##
        más reciente a cada chunk para preservar el contexto estructural
        del documento sin coste de API adicional.

        Args:
            chunks: lista de chunks fusionados.
            documento: documento completo para extraer el contexto de sección.

        Returns:
            Lista de chunks con contexto de sección anteponido.
        """
        chunks_contextuales = []
        for chunk in chunks:
            seccion = self._extraer_seccion_activa(chunk, documento)
            if seccion and not chunk.startswith(seccion):
                chunks_contextuales.append(f"{seccion}\n\n{chunk}")
            else:
                chunks_contextuales.append(chunk)
        return chunks_contextuales
