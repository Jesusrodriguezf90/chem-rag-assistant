"""
test_loader.py

Tests unitarios para src/ingestion/loader.py

Cubre:
    - Limpieza universal de artefactos de Docling
    - Limpieza específica por documento desde YAML
    - Manejo de errores (archivo no encontrado, PDF vacío)
    - Carga de configuración YAML

Autor:   Jesús Rodríguez
Versión: 1.0.0
"""

import re
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.ingestion.loader import DEFAULT_CONFIG_PATH, DocumentLoader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config_yaml_temporal():
    """Crea un archivo YAML de configuración temporal para los tests."""
    reglas = {
        "documento_test": {
            "remove_patterns": ["PATRON_TEST", "Cite this:.*"],
            "remove_affiliations": True,
            "fix_ocr_errors": True,
            "remove_references_section": True,
        }
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        delete=False,
        encoding="utf-8",
    ) as f:
        yaml.dump(reglas, f)
        return Path(f.name)


@pytest.fixture
def loader_con_config(config_yaml_temporal):
    """Instancia DocumentLoader con configuración temporal."""
    return DocumentLoader(config_path=config_yaml_temporal)


@pytest.fixture
def loader_sin_config():
    """Instancia DocumentLoader sin archivo de configuración."""
    return DocumentLoader(config_path=Path("ruta_inexistente.yaml"))


# ---------------------------------------------------------------------------
# Tests de limpieza universal
# ---------------------------------------------------------------------------

class TestLimpiezaUniversal:
    """Tests para el método _limpiar_universal."""

    def test_elimina_placeholders_imagen(self, loader_sin_config):
        texto = "Texto antes\n<!-- image -->\nTexto después"
        resultado = loader_sin_config._limpiar_universal(texto)
        assert "<!-- image -->" not in resultado
        assert "Texto antes" in resultado
        assert "Texto después" in resultado

    def test_elimina_unicode_problematico(self, loader_sin_config):
        texto = "compuesto\ue104uoromethanesulfonate"
        resultado = loader_sin_config._limpiar_universal(texto)
        assert "\ue104" not in resultado

    def test_elimina_encabezados_vacios(self, loader_sin_config):
        texto = "## Sección válida\n\nContenido\n\n##\n\nMás contenido"
        resultado = loader_sin_config._limpiar_universal(texto)
        assert re.search(r"^##\s*$", resultado, re.MULTILINE) is None

    def test_colapsa_saltos_de_linea(self, loader_sin_config):
        texto = "Párrafo 1\n\n\n\n\nPárrafo 2"
        resultado = loader_sin_config._limpiar_universal(texto)
        assert "\n\n\n" not in resultado
        assert "Párrafo 1" in resultado
        assert "Párrafo 2" in resultado

    def test_texto_sin_artefactos_no_cambia(self, loader_sin_config):
        texto = "## Introducción\n\nTexto limpio sin artefactos."
        resultado = loader_sin_config._limpiar_universal(texto)
        assert "Introducción" in resultado
        assert "Texto limpio sin artefactos." in resultado


# ---------------------------------------------------------------------------
# Tests de limpieza específica
# ---------------------------------------------------------------------------

class TestLimpiezaEspecifica:
    """Tests para el método _limpiar_especifico."""

    def test_elimina_patrones_especificos(self, loader_con_config):
        texto = "Texto con PATRON_TEST en medio del contenido"
        resultado = loader_con_config._limpiar_especifico(texto, "documento_test")
        assert "PATRON_TEST" not in resultado

    def test_corrige_errores_ocr(self, loader_con_config):
        texto = "NHCs, rst described in 1991"
        resultado = loader_con_config._limpiar_especifico(texto, "documento_test")
        assert " rst " not in resultado
        assert "first" in resultado

    def test_elimina_seccion_referencias(self, loader_con_config):
        texto = "## Conclusión\n\nTexto.\n\n## Notes and references\n\n1. Autor et al."
        resultado = loader_con_config._limpiar_especifico(texto, "documento_test")
        assert "Notes and references" not in resultado
        assert "Conclusión" in resultado

    def test_documento_sin_reglas_no_cambia(self, loader_con_config):
        texto = "## Introducción\n\nContenido original sin cambios."
        resultado = loader_con_config._limpiar_especifico(texto, "documento_sin_reglas")
        assert resultado == texto

    def test_elimina_afiliaciones(self, loader_con_config):
        texto = "Introducción\n\na Technical University of Munich, Garching, Germany\n\nContenido"
        resultado = loader_con_config._limpiar_especifico(texto, "documento_test")
        assert "Technical University of Munich" not in resultado


# ---------------------------------------------------------------------------
# Tests de carga de configuración
# ---------------------------------------------------------------------------

class TestCargaConfig:
    """Tests para el método _cargar_config."""

    def test_carga_yaml_valido(self, config_yaml_temporal):
        loader = DocumentLoader(config_path=config_yaml_temporal)
        assert "documento_test" in loader._config

    def test_yaml_inexistente_devuelve_dict_vacio(self):
        loader = DocumentLoader(config_path=Path("no_existe.yaml"))
        assert loader._config == {}

    def test_config_por_defecto_existe(self):
        assert DEFAULT_CONFIG_PATH.name == "cleaning_rules.yaml"


# ---------------------------------------------------------------------------
# Tests de manejo de errores
# ---------------------------------------------------------------------------

class TestManejoErrores:
    """Tests para el manejo de errores en el método load."""

    def test_pdf_no_encontrado_lanza_error(self, loader_sin_config):
        with pytest.raises(FileNotFoundError):
            loader_sin_config.load(Path("pdf_inexistente.pdf"))

    def test_pdf_vacio_lanza_error(self, loader_sin_config):
        # Simular que Docling devuelve texto vacío
        mock_resultado = MagicMock()
        mock_resultado.document.export_to_markdown.return_value = "   "

        with (
            tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as pdf_temp,
            patch.object(loader_sin_config.converter, "convert", return_value=mock_resultado),
        ):
            with pytest.raises(ValueError):
                loader_sin_config.load(Path(pdf_temp.name))
