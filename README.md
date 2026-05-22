---
title: Chem RAG Assistant
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: API RAG para consulta de literatura química
---

# Chem RAG Assistant

API REST inteligente para consulta de literatura científica de química mediante Generación Aumentada por Recuperación (RAG).

**Pipeline RAG completo con fine-tuning QLoRA — implementado, evaluado, containerizado y desplegado en Hugging Face Spaces.**


---

## Tabla de Contenidos

- [Descripción](#descripción)
- [Funcionalidades](#funcionalidades)
- [Arquitectura](#arquitectura)
- [Stack Tecnológico](#stack-tecnológico)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Instalación y Puesta en Marcha](#instalación-y-puesta-en-marcha)
- [Endpoints de la API](#endpoints-de-la-api)
- [Despliegue con Docker](#despliegue-con-docker)
- [Hoja de Ruta](#hoja-de-ruta)
- [Licencia](#licencia)

---

## Descripción

**Chem RAG Assistant** es una API REST construida con FastAPI que permite consultar documentos científicos de química en lenguaje natural. El sistema procesa PDFs técnicos — artículos sobre compuestos organometálicos, protocolos de síntesis, informes de química ambiental — los indexa semánticamente y genera respuestas fundamentadas en el contenido del documento mediante un modelo de lenguaje grande (LLM).

El pipeline completo está containerizado con Docker y diseñado para desplegarse en **Hugging Face Spaces** usando el SDK Docker, sin coste de infraestructura.

---

## Funcionalidades

> Las funcionalidades marcadas con 🔲 están especificadas y pendientes de implementación.

- ✅ Extracción estructurada de texto y contenido científico mediante Docling
- ✅ Limpieza y preprocesamiento configurable por documento (universal + específica por YAML)
- ✅ Indexación semántica con embeddings BGE-M3
- ✅ Búsqueda por similitud sobre base de datos vectorial ChromaDB
- ✅ Evaluación de recuperación con preguntas de complejidad progresiva y ground truth
- ✅ Generación de respuestas en lenguaje natural mediante Qwen3 (HF Inference API)
- ✅ Evaluación comparativa de 4 estrategias de chunking con métricas cuantitativas
- ✅ Implementación de Contextual Chunk Headers como estrategia óptima de chunking
- ✅ Pipeline RAG orquestado como módulos Python reutilizables (src/)
- ✅ Ingesta de PDFs científicos vía endpoint REST (`POST /upload`)
- ✅ Documentación interactiva automática con Swagger UI (`/docs`) y ReDoc (`/redoc`)
- ✅ Containerización completa con Docker
- ✅ Despliegue en Hugging Face Spaces (Docker SDK)
- ✅ Pipeline CI/CD con GitHub Actions — validación automática en cada push
- ✅ Construcción del dataset de fine-tuning QA sobre química organometálica (notebook 05_dataset)
- ✅ Fine-tuning QLoRA de LLM especializado en química — Qwen2.5-1.5B publicado en HF Hub (notebook 06_finetune)
- ✅ Evaluación comparativa base vs fine-tuned vs RAG con análisis crítico de resultados (notebook 06b_evaluacion)
- ✅ Agente RAG con LangGraph — Adaptive RAG con Router, Grader y Rewriter (notebook 07_agente)

---

## Arquitectura

El sistema sigue una arquitectura de pipeline RAG en tres fases: ingesta, recuperación y generación. Todas las capas están encapsuladas dentro de un único contenedor Docker.

```
┌─────────────────────────────────────────────────────────────┐
│                      Cliente / Usuario                      │
│              (Swagger UI · curl · Frontend externo)         │
└───────────────────────────┬─────────────────────────────────┘
                            │  HTTP REST
                            ▼
┌─────────────────────────────────────────────────────────────┐
│               Contenedor Docker (HF Spaces)                 │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              FastAPI + Uvicorn                      │   │
│   │         /upload · /query · /health                  │   │
│   └───────────────────────┬─────────────────────────────┘   │
│                           │                                 │
│   ┌───────────────────────▼─────────────────────────────┐   │
│   │                  Pipeline RAG                       │   │
│   │                                                     │   │
│   │  ┌───────────┐  ┌───────────┐  ┌─────────────────┐  │   │
│   │  │  Ingesta  │─▶│ Embeddings│─▶│  Base Vectorial│  │   │
│   │  │ (Docling) │  │  (BGE-M3) │  │   (ChromaDB)    │  │   │
│   │  └───────────┘  └───────────┘  └────────┬────────┘  │   │
│   │                                         │           │   │
│   │  ┌───────────────────────────────────────▼────────┐ │   │
│   │  │        Recuperación + Generación               │ │   │
│   │  │   (LangChain + Qwen3 vía HF Inference API)     │ │   │
│   │  └────────────────────────────────────────────────┘ │   │
│   └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Stack Tecnológico

| Componente | Herramienta | Propósito |
|---|---|---|
| API REST | [FastAPI](https://fastapi.tiangolo.com/) | Servidor web asíncrono con documentación automática |
| Servidor ASGI | [Uvicorn](https://www.uvicorn.org/) | Servidor de producción para FastAPI |
| Extracción de PDF | [Docling (IBM)](https://github.com/DS4SD/docling) | Parseo con reconocimiento de estructura en PDFs científicos |
| Embeddings | [BGE-M3](https://huggingface.co/BAAI/bge-m3) | Representación vectorial semántica |
| Base Vectorial | [ChromaDB](https://www.trychroma.com/) | Búsqueda por similitud embebida en el contenedor |
| Orquestación RAG | [LangChain](https://www.langchain.com/) | Gestión del pipeline de recuperación y generación |
| LLM | [Qwen3](https://huggingface.co/Qwen) vía HF Inference API | Generación de respuestas fundamentadas |
| Contenedor | [Docker](https://www.docker.com/) | Empaquetado y despliegue reproducible |
| Despliegue | [Hugging Face Spaces (Docker SDK)](https://huggingface.co/docs/hub/spaces-sdks-docker) | Hosting gratuito en la nube |
| Desarrollo | [Google Colab](https://colab.research.google.com/) | Prototipado y experimentación con notebooks |
| Fine-tuning | [PEFT/LoRA](https://github.com/huggingface/peft) | Adaptadores QLoRA eficientes en memoria |
| Fine-tuning | [TRL](https://github.com/huggingface/trl) | SFTTrainer para supervisión del entrenamiento |
| Fine-tuning | [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) | Quantización 4-bit NF4 |
| Desarrollo | [Kaggle Notebooks](https://www.kaggle.com/code) | Alternativa gratuita a Colab con GPU T4 |
| Agente | [LangGraph](https://github.com/langchain-ai/langgraph) | Orquestación del grafo de agentes con estado persistente |
| Agente | [LangChain Core](https://github.com/langchain-ai/langchain) | Contratos base de mensajes y prompts |

---

## Estructura del Proyecto

```
chem-rag-assistant/
│
├── .github/
   └── workflows/
│       └── ci.yml              # Pipeline CI/CD — tests, Docker build y deploy a HF Spaces
│
├── config/
│   └── cleaning_rules.yaml     # Reglas de limpieza específicas por documento
│
├── data/
│   ├── raw/                    # PDFs de entrada (excluidos de Git)
│   ├── processed/              # Documento extraído y limpio (excluido de Git)
│   ├── embeddings/             # Vectores y chunks generados (excluidos de Git)
│   ├── chroma_db/              # Base de datos vectorial ChromaDB (excluida de Git)
│   ├── eval/                   # Resultados de evaluación comparativa (excluidos de Git)
│   ├── chroma_eval/            # Colecciones ChromaDB temporales (excluidas de Git)
│   └── finetune/               # Dataset y artefactos de fine-tuning (excluidos de Git)
│
├── notebooks/
│   ├── 00_setup.ipynb          # Configuración inicial del entorno (ejecutar una vez)
│   ├── 01_ingesta.ipynb        # Prototipo: carga y troceado de PDFs
│   ├── 02_embeddings.ipynb     # Prototipo: generación e indexación de embeddings
│   ├── 02b_chunking_eval.ipynb # Evaluación comparativa de estrategias de chunking
│   ├── 03_recuperacion.ipynb   # Prototipo: pipeline de recuperación
│   ├── 04_generacion.ipynb     # Prototipo: pipeline RAG completo con Qwen3
│   ├── 05_dataset.ipynb        # Construcción del dataset de fine-tuning QA
│   ├── 06_finetune.ipynb       # Fine-tuning QLoRA — Qwen2.5-1.5B en química organometálica
│   ├── 06b_evaluacion.ipynb    # Evaluación comparativa base vs fine-tuned vs RAG (Colab/Kaggle)
│   └── 07_agente.ipynb         # Agente RAG con LangGraph — Router, Grader, Rewriter, Generator
│
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py           # Cargador de PDFs basado en Docling
│   │   └── chunker.py          # Lógica de troceado de texto
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedder.py         # Wrapper de embeddings BGE-M3
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── vector_store.py     # Interfaz con ChromaDB
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── rag.py              # Orquestación RAG con LangChain
│   └── agent/
│       ├── __init__.py
│       └── graph.py            # Agente RAG con LangGraph — Adaptive RAG + Corrective RAG
│
├── app/
│   ├── main.py                 # Punto de entrada FastAPI
│   ├── state.py                # Estado compartido de la aplicación (Singleton)
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── ingestion.py        # Endpoint POST /upload
│   │   ├── query.py            # Endpoint POST /query
│   │   └── agent.py            # Endpoint POST /agent/query
│   └── schemas/
│       ├── __init__.py
│       └── models.py           # Modelos Pydantic (request/response)
│
├── tests/
│   ├── test_loader.py          # Tests unitarios de DocumentLoader
│   ├── test_chunker.py         # Tests unitarios de DocumentChunker
│   ├── test_embedder.py        # Tests unitarios de DocumentEmbedder
│   ├── test_vector_store.py    # Tests unitarios de VectorStore
│   ├── test_rag.py             # Tests unitarios de RAGPipeline
│   └── test_agent.py           # Tests unitarios de AgentGraph
│
├── .gitattributes              # Configuración Git LFS para archivos binarios (HF Spaces)
├── Dockerfile                  # Definición del contenedor
├── .dockerignore               # Archivos excluidos del contenedor
├── .env.example                # Plantilla de variables de entorno
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Instalación y Puesta en Marcha

### En Google Colab (prototipado)

1. Abre [Google Colab](https://colab.research.google.com/)
2. Monta tu Google Drive y clona este repositorio
3. Ejecuta los notebooks en orden: `00_setup` → `01_ingesta` → `02_embeddings` → `03_recuperacion` → `04_generacion`

### Requisitos previos

- Python 3.10+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado y en ejecución
- Cuenta en [Hugging Face](https://huggingface.co/join) con token de API generado
- Cuenta en [GitHub](https://github.com/) para control de versiones

### Variables de entorno

Copia `.env.example` a `.env` y añade tu token de Hugging Face:

```env
HF_TOKEN=tu_token_de_huggingface_aquí
```

> El archivo `.env` está excluido de Git por defecto. No lo subas al repositorio bajo ninguna circunstancia.

### Ejecución local sin Docker

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/chem-rag-assistant.git
cd chem-rag-assistant

# 2. Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Arrancar el servidor
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

La documentación interactiva estará disponible en `http://localhost:8000/docs`.

---

## Endpoints de la API

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/health` | Estado del servicio |
| `POST` | `/upload` | Subir e indexar un documento PDF |
| `POST` | `/query` | Realizar una consulta en lenguaje natural |
| `POST` | `/agent/query` | Consulta con agente RAG adaptativo (Router + Grader + Rewriter) |

### Ejemplo de uso previsto

```bash
# Subir un documento
curl -X POST "http://localhost:8000/upload" \
  -F "file=@paper_quimica.pdf"

# Realizar una consulta
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Qué compuestos de Pd(II) se sintetizaron?"}'
```

---
## Datos de prueba

> El documento utilizado para el desarrollo y validación del pipeline RAG es un artículo científico de acceso abierto sobre complejos tetra-NHC de metales de transición con actividad antitumoral, seleccionado por su relevancia directa con la temática del proyecto y su alta densidad técnica (fórmulas químicas, tablas de caracterización, nomenclatura especializada).

**Referencia:**

Wolfgang R E Büchele, Tim P Schlachta, Andreas L Gebendorfer, Jenny Pamperin, Leon F Richter, Michael J Sauer, Aram Prokop, Fritz E Kühn (2024). Synthesis, characterization, and biomedical evaluation of ethylene-bridged tetra-NHC Pd(ii), Pt(ii) and Au(iii) complexes, with apoptosis-inducing properties in cisplatin-resistant neuroblastoma cells. *Frontiers in Chemistry*.
PMC: [PMC10967698](https://pmc.ncbi.nlm.nih.gov/articles/PMC10967698/)

**Condiciones de uso:**

El documento se utiliza exclusivamente con fines de investigación y
desarrollo. No se distribuye ni se incluye en el repositorio — está
excluido del control de versiones mediante `.gitignore`.

## Despliegue con Docker

> ✅ **Implementado, validado localmente y desplegado en Hugging Face Spaces.**

El proyecto está diseñado para desplegarse como un Docker Space en Hugging Face. El flujo de despliegue objetivo es el siguiente:

### Build y ejecución local

```bash
# Construir la imagen
docker build -t chem-rag-assistant .

# Ejecutar el contenedor
docker run -p 7860:7860 --env-file .env chem-rag-assistant:latest
```

### Despliegue en Hugging Face Spaces

El Space utiliza el SDK Docker de HF Spaces. Para replicar el despliegue:

1. Crea un nuevo Space en Hugging Face seleccionando **Docker** como SDK
2. Conecta tu fork del repositorio de GitHub al Space
3. Añade `HF_TOKEN` como secreto desde la configuración del Space (`Settings → Repository secrets`)
4. Cualquier push a la rama `main` desencadena una nueva build automáticamente

La API quedará accesible en:
`https://jesusrodriguezf90-chem-rag-assistant.hf.space/docs`

---

## Hoja de Ruta

| Estado | Componente |
|---|---|
| ✅ | Estructura del proyecto y documentación |
| ✅ | Configuración inicial del entorno (notebook 00_setup) |
| ✅ | Prototipo de ingesta con Docling y limpieza configurable por documento (notebook) |
| ✅ | Pipeline de embeddings con BGE-M3, fusión de chunks y Contextual Chunk Headers (notebook) |
| ✅ | Integración con ChromaDB y evaluación de recuperación con ground truth (notebook) |
| ✅ | Pipeline RAG completo con Qwen3 — generación de respuestas (notebook) |
| ✅ | Evaluación comparativa de estrategias de chunking (notebook 02b) |
| ✅ | Migración del pipeline RAG a módulos Python en src/ con tests unitarios |
| ✅ | Implementación de la API REST con FastAPI |
| ✅ | Modelos Pydantic y validación de esquemas |
| ✅ | Containerización con Docker |
| ✅ | Despliegue en Hugging Face Spaces (Docker SDK) |
| ✅ | Pipeline CI/CD con GitHub Actions — tests, Docker build y deploy automático |
| ✅ | Construcción del dataset de fine-tuning QA (notebook 05_dataset) |
| ✅ | Fine-tuning QLoRA del LLM especializado en química (notebook 06_finetune) |
| ✅ | Evaluación comparativa base vs fine-tuned vs RAG (notebook 06b_evaluacion) |
| ✅ | Agente LangGraph sobre pipeline RAG — Adaptive RAG con self-correction |
| 🔲 | Extracción específica de fórmulas y compuestos químicos |
| 🔲 | Soporte multi-documento |
---

## Licencia

Este proyecto está distribuido bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.

---

<p align="center">
  Construido para la química y la inteligencia artificial
</p>