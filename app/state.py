"""
state.py

Gestión del estado compartido de la aplicación FastAPI.

Responsabilidad:
    Centralizar la inicialización y acceso a las dependencias compartidas
    entre los distintos routers: DocumentEmbedder, VectorStore y HF_TOKEN.

    El patrón de estado global es apropiado aquí porque estas dependencias
    son costosas de inicializar (BGE-M3 ~2.2 GB, ChromaDB en disco) y deben
    compartirse entre todos los requests sin reinicializarse en cada llamada.

    En producción con múltiples workers usar un servicio de caché externo
    como Redis para compartir estado entre procesos.

    Por qué existe este módulo:
        FastAPI instancia los routers de forma independiente. Sin un estado
        centralizado, cada router crearía su propia instancia de DocumentEmbedder
        (descargando BGE-M3 dos veces) y su propio VectorStore (con índices
        desincronizados). state.py resuelve esto con el patrón Singleton:
        una única instancia de cada dependencia inicializada en el startup
        del servidor y compartida entre todos los routers y requests.

Autor:   Jesús Rodríguez
Versión: 1.0.0
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from src.embeddings import DocumentEmbedder, EmbedderConfig
from src.retrieval import VectorStore

load_dotenv()

# ---------------------------------------------------------------------------
# Variables de estado global
# ---------------------------------------------------------------------------

# Instancias compartidas entre todos los requests
# Se inicializan en el evento startup de FastAPI (ver main.py)
_embedder    : DocumentEmbedder | None = None
_vector_store: VectorStore | None      = None
_hf_token    : str | None              = None

# ---------------------------------------------------------------------------
# Inicialización
# ---------------------------------------------------------------------------

def inicializar_estado(persist_path: Path) -> None:
    """Inicializa las dependencias compartidas de la aplicación.

    Se llama una única vez en el evento lifespan de FastAPI al arrancar
    el servidor. Carga BGE-M3 en memoria y conecta con ChromaDB.

    Args:
        persist_path: ruta donde ChromaDB persiste los datos en disco.
    """
    global _embedder, _vector_store, _hf_token

    # Cargar modelo de embeddings — descarga ~2.2 GB en primera ejecución
    _embedder = DocumentEmbedder(
        config=EmbedderConfig(device="cpu")
    )

    # Conectar con ChromaDB
    _vector_store = VectorStore(persist_path=persist_path)

    # Cargar token de HF Inference API
    _hf_token = os.getenv("HF_TOKEN")

# ---------------------------------------------------------------------------
# Getters
# ---------------------------------------------------------------------------

def get_embedder() -> DocumentEmbedder:
    """Devuelve la instancia compartida de DocumentEmbedder.

    Raises:
        RuntimeError: si el estado no ha sido inicializado.
    """
    if _embedder is None:
        raise RuntimeError(
            "DocumentEmbedder no inicializado. "
            "Verifica que el servidor arrancó correctamente."
        )
    return _embedder

def get_vector_store() -> VectorStore:
    """Devuelve la instancia compartida de VectorStore.

    Raises:
        RuntimeError: si el estado no ha sido inicializado.
    """
    if _vector_store is None:
        raise RuntimeError(
            "VectorStore no inicializado. "
            "Verifica que el servidor arrancó correctamente."
        )
    return _vector_store

def get_hf_token() -> str:
    """Devuelve el token de HF Inference API.

    Raises:
        RuntimeError: si HF_TOKEN no está configurado.
    """
    if not _hf_token:
        raise RuntimeError(
            "HF_TOKEN no encontrado. "
            "Configúralo en el archivo .env."
        )
    return _hf_token
