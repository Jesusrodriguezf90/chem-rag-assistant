"""
test_chunker.py

Tests unitarios para src/ingestion/chunker.py

Cubre:
    - División básica de texto en chunks
    - Fusión de chunks cortos con el siguiente
    - Filtrado de chunks vacíos
    - Anteposición de contexto de sección
    - Configuración personalizada via ChunkConfig
    - Manejo de errores

Autor:   Jesús Rodríguez
Versión: 1.0.0
"""

import pytest

from src.ingestion.chunker import ChunkConfig, DocumentChunker

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def chunker_defecto():
    """Instancia DocumentChunker con configuración por defecto."""
    return DocumentChunker()

@pytest.fixture
def chunker_chunks_pequenos():
    """Instancia DocumentChunker con chunk_size pequeño para tests."""
    config = ChunkConfig(
        chunk_size=200,
        chunk_overlap=20,
        min_length=50,
    )
    return DocumentChunker(config=config)

@pytest.fixture
def documento_simple():
    """Documento Markdown simple para tests básicos."""
    return (
        "## Introduction\n\n"
        "N-heterocyclic carbenes (NHCs) are widely used in catalysis. "
        "They form stable complexes with transition metals such as Pd, Pt and Au. "
        "These complexes have applications in medicinal chemistry.\n\n"
        "## Results\n\n"
        "The synthesis of PdL3 was achieved using standard NHC coordination. "
        "Characterization was performed by NMR and ESI-MS. "
        "Biological evaluation showed apoptosis induction in tumor cells."
    )

@pytest.fixture
def documento_con_titulos_aislados():
    """Documento donde los títulos de sección quedan aislados tras el split."""
    return (
        "## Introduction\n\n" + "A" * 900 + "\n\n"
        "## Results\n\n" + "B" * 900 + "\n\n"
        "## Conclusion\n\n" + "C" * 900
    )

# ---------------------------------------------------------------------------
# Tests de división básica
# ---------------------------------------------------------------------------

class TestDivisionBasica:
    """Tests para el método split."""

    def test_split_devuelve_lista(self, chunker_defecto, documento_simple):
        resultado = chunker_defecto.split(documento_simple)
        assert isinstance(resultado, list)

    def test_split_devuelve_chunks_no_vacios(self, chunker_defecto, documento_simple):
        resultado = chunker_defecto.split(documento_simple)
        assert all(chunk.strip() for chunk in resultado)

    def test_split_texto_vacio_lanza_error(self, chunker_defecto):
        with pytest.raises(ValueError):
            chunker_defecto.split("")

    def test_split_texto_solo_espacios_lanza_error(self, chunker_defecto):
        with pytest.raises(ValueError):
            chunker_defecto.split("   \n\n   ")

    def test_split_preserva_contenido(self, chunker_defecto, documento_simple):
        resultado = chunker_defecto.split(documento_simple)
        texto_reconstruido = " ".join(resultado)
        # Verificar que términos clave del documento están presentes
        assert "NHC" in texto_reconstruido
        assert "Pd" in texto_reconstruido

# ---------------------------------------------------------------------------
# Tests de fusión de chunks cortos
# ---------------------------------------------------------------------------

class TestFusionChunksCortos:
    """Tests para el método _fusionar_chunks_cortos."""

    def test_fusiona_chunk_corto_con_siguiente(self, chunker_defecto):
        chunks = ["## Introduction", "Contenido de la introducción con suficiente texto."]
        resultado = chunker_defecto._fusionar_chunks_cortos(chunks)
        assert len(resultado) == 1
        assert "## Introduction" in resultado[0]
        assert "Contenido" in resultado[0]

    def test_no_fusiona_chunks_suficientemente_largos(self, chunker_defecto):
        chunk_largo = "A" * 200
        chunks = [chunk_largo, chunk_largo]
        resultado = chunker_defecto._fusionar_chunks_cortos(chunks)
        assert len(resultado) == 2

    def test_filtra_chunks_vacios(self, chunker_defecto):
        chunks = ["Contenido válido con suficiente longitud.", "", "Otro contenido válido."]
        resultado = chunker_defecto._fusionar_chunks_cortos(chunks)
        assert all(chunk.strip() for chunk in resultado)

    def test_chunk_corto_al_final_no_fusiona(self, chunker_defecto):
        chunks = ["Contenido largo suficiente para no fusionar.", "## Fin"]
        resultado = chunker_defecto._fusionar_chunks_cortos(chunks)
        # El chunk corto al final no tiene siguiente — se mantiene como está
        # El número de chunks puede ser 1 o 2 dependiendo de si el splitter
        # los agrupa, pero el contenido de ambos debe estar presente
        texto_resultado = " ".join(resultado)
        assert "Contenido largo" in texto_resultado
        assert "## Fin" in texto_resultado

# ---------------------------------------------------------------------------
# Tests de contexto de sección
# ---------------------------------------------------------------------------

class TestContextoSeccion:
    """Tests para los métodos de contexto de sección."""

    def test_extrae_seccion_correcta(self, chunker_defecto, documento_simple):
        chunk = "The synthesis of PdL3 was achieved"
        seccion = chunker_defecto._extraer_seccion_activa(chunk, documento_simple)
        assert "Results" in seccion

    def test_seccion_no_encontrada_devuelve_vacio(self, chunker_defecto):
        chunk = "Texto que no existe en el documento"
        seccion = chunker_defecto._extraer_seccion_activa(chunk, "Documento diferente")
        assert seccion == ""

    def test_antepone_seccion_si_no_esta_presente(self, chunker_defecto, documento_simple):
        chunks = ["The synthesis of PdL3 was achieved using standard NHC coordination."]
        resultado = chunker_defecto._añadir_contexto_seccion(chunks, documento_simple)
        assert resultado[0].startswith("##")

    def test_no_duplica_seccion_si_ya_esta_presente(self, chunker_defecto, documento_simple):
        chunks = ["## Results\n\nContenido de resultados."]
        resultado = chunker_defecto._añadir_contexto_seccion(chunks, documento_simple)
        # Verificar que no hay doble encabezado
        assert resultado[0].count("## Results") == 1

# ---------------------------------------------------------------------------
# Tests de configuración
# ---------------------------------------------------------------------------

class TestConfiguracion:
    """Tests para ChunkConfig y configuración del chunker."""

    def test_config_por_defecto(self, chunker_defecto):
        assert chunker_defecto.config.chunk_size    == 1000
        assert chunker_defecto.config.chunk_overlap == 100
        assert chunker_defecto.config.min_length    == 100

    def test_config_personalizada(self):
        config = ChunkConfig(chunk_size=500, chunk_overlap=50, min_length=75)
        chunker = DocumentChunker(config=config)
        assert chunker.config.chunk_size    == 500
        assert chunker.config.chunk_overlap == 50
        assert chunker.config.min_length    == 75

    def test_chunks_mas_pequenos_con_config_reducida(
        self, chunker_chunks_pequenos, documento_simple
    ):
        chunker_grande = DocumentChunker()
        resultado_grande   = chunker_grande.split(documento_simple)
        resultado_pequeno  = chunker_chunks_pequenos.split(documento_simple)
        # Chunk size más pequeño debe producir más chunks o igual
        assert len(resultado_pequeno) >= len(resultado_grande)

    def test_chunks_respetan_limite_tokens_bge_m3(self, chunker_defecto, documento_simple):
        """Verifica que ningún chunk supera el límite de 8192 tokens de BGE-M3.
        Aproximación: 1 token ≈ 4 caracteres en inglés."""
        LIMITE_TOKENS_BGE_M3 = 8192
        CHARS_POR_TOKEN      = 4
        limite_chars         = LIMITE_TOKENS_BGE_M3 * CHARS_POR_TOKEN

        resultado = chunker_defecto.split(documento_simple)
        assert all(len(chunk) <= limite_chars for chunk in resultado)
