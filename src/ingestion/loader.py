"""
loader.py

Módulo de carga y limpieza de documentos PDF científicos.

Responsabilidad:
    Recibir la ruta a un PDF y devolver el texto limpio en Markdown,
    aplicando limpieza universal (artefactos de Docling) y limpieza
    específica por documento definida en config/cleaning_rules.yaml.

    Este módulo es la primera etapa del pipeline RAG y su output
    es la entrada de src/ingestion/chunker.py.

Autor:   Jesús Rodríguez
Versión: 1.0.0
"""

import re
from pathlib import Path

import yaml
from docling.document_converter import DocumentConverter

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Ruta por defecto al archivo de reglas de limpieza
# Se puede sobreescribir pasando config_path al instanciar DocumentLoader
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "cleaning_rules.yaml"

# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class DocumentLoader:
    """Carga y limpia documentos PDF científicos usando Docling.

    Aplica dos fases de limpieza sobre el texto extraído:
      1. Limpieza universal: elimina artefactos comunes a cualquier PDF
         extraído con Docling (placeholders de imágenes, Unicode problemático,
         saltos de línea múltiples).
      2. Limpieza específica: aplica reglas opcionales definidas en
         config/cleaning_rules.yaml para un documento concreto.

    Attributes:
        config_path: ruta al archivo YAML de reglas de limpieza.
        converter: instancia de DocumentConverter de Docling.
        _config: diccionario con las reglas cargadas desde el YAML.
    """

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
        """Inicializa el loader con la ruta al archivo de configuración.

        Args:
            config_path: ruta al archivo cleaning_rules.yaml.
                         Por defecto usa config/cleaning_rules.yaml
                         relativo a la raíz del proyecto.
        """
        self.config_path = config_path
        self.converter   = DocumentConverter()
        self._config     = self._cargar_config()

    # -----------------------------------------------------------------------
    # API pública
    # -----------------------------------------------------------------------

    def load(self, pdf_path: Path) -> str:
        """Carga un PDF y devuelve el texto limpio en Markdown.

        Args:
            pdf_path: ruta al archivo PDF de entrada.

        Returns:
            Texto del documento en formato Markdown, limpio y listo
            para ser procesado por el chunker.

        Raises:
            FileNotFoundError: si el PDF no existe en la ruta indicada.
            ValueError: si el PDF no pudo ser procesado por Docling.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF no encontrado: {pdf_path}")

        # Extracción del texto mediante Docling
        resultado = self.converter.convert(str(pdf_path))
        texto_raw = resultado.document.export_to_markdown()

        if not texto_raw.strip():
            raise ValueError(f"Docling no pudo extraer texto de: {pdf_path}")

        # Aplicar limpieza universal y específica
        texto_limpio = self._limpiar_universal(texto_raw)
        texto_limpio = self._limpiar_especifico(texto_limpio, pdf_path.stem)

        return texto_limpio

    # -----------------------------------------------------------------------
    # Métodos privados
    # -----------------------------------------------------------------------

    def _cargar_config(self) -> dict:
        """Carga el archivo YAML de reglas de limpieza.

        Returns:
            Diccionario con las reglas. Vacío si el archivo no existe.
        """
        if not self.config_path.exists():
            return {}
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _limpiar_universal(self, texto: str) -> str:
        """Aplica limpieza universal a cualquier PDF extraído con Docling.

        Elimina artefactos independientes del documento:
          - Placeholders de imágenes generados por Docling
          - Caracteres Unicode problemáticos (ligaduras tipográficas no reconocidas)
          - Encabezados Markdown vacíos o residuales
          - Saltos de línea múltiples consecutivos

        Args:
            texto: texto extraído por Docling en formato Markdown.

        Returns:
            Texto con artefactos universales eliminados.
        """
        # Eliminar placeholders de imágenes generados por Docling
        texto = re.sub(r"<!--\s*image\s*-->", "", texto)

        # Eliminar caracteres Unicode problemáticos
        # (ligaduras tipográficas no reconocidas como fl, fi, ff)
        texto = re.sub(r"[\ue000-\uf8ff]", "", texto)

        # Eliminar encabezados Markdown vacíos o residuales
        texto = re.sub(r"^##\s*$", "", texto, flags=re.MULTILINE)

        # Colapsar múltiples saltos de línea consecutivos en uno solo
        texto = re.sub(r"\n{3,}", "\n\n", texto)

        return texto.strip()

    def _limpiar_especifico(self, texto: str, nombre_doc: str) -> str:
        """Aplica limpieza específica definida en cleaning_rules.yaml.

        Si no existe entrada para el documento, devuelve el texto sin cambios.

        Args:
            texto: texto tras la limpieza universal.
            nombre_doc: nombre del PDF sin extensión (clave en el YAML).

        Returns:
            Texto con limpieza específica aplicada.
        """
        reglas = self._config.get(nombre_doc, {})
        if not reglas:
            return texto

        # Eliminar patrones específicos del documento
        for patron in reglas.get("remove_patterns", []):
            texto = re.sub(patron, "", texto, flags=re.IGNORECASE)

        # Eliminar bloques de afiliaciones institucionales
        if reglas.get("remove_affiliations"):
            texto = re.sub(
                r"\n[a-z]\s+[A-Z][^\n]{20,}\n",
                "\n",
                texto,
            )

        # Corregir errores OCR básicos conocidos en papers científicos
        if reglas.get("fix_ocr_errors"):
            ocr_fixes = {
                r" rst ":       " first ",
                r"O  en ":      "Often ",
                r"signi  cant": "significant",
                r"di  erent":   "different",
                r"a  ect":      "affect",
                r"e  ect":      "effect",
            }
            for patron, reemplazo in ocr_fixes.items():
                texto = re.sub(patron, reemplazo, texto, flags=re.IGNORECASE)

        # Eliminar sección de referencias al final del documento
        if reglas.get("remove_references_section"):
            texto = re.sub(
                r"## Notes and references.*$",
                "",
                texto,
                flags=re.DOTALL | re.IGNORECASE,
            )

        # Eliminar encabezados Markdown vacíos generados por las eliminaciones
        texto = re.sub(r"^##\s*$", "", texto, flags=re.MULTILINE)

        # Colapsar saltos de línea generados por las eliminaciones
        texto = re.sub(r"\n{3,}", "\n\n", texto)

        return texto.strip()
