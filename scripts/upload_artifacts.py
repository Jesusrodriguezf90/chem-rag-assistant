"""
upload_artifacts.py

Sube los artefactos del pipeline RAG a HuggingFace Hub.

Uso:
    python scripts/upload_artifacts.py

Requisitos:
    - HF_TOKEN configurado en .env
    - Archivos en data/embeddings/:
        - PMC10967698_chunks.json
        - PMC10967698_embeddings.npy

Este script se ejecuta una única vez para publicar los artefactos
necesarios para reconstruir ChromaDB en el Space Gradio sin
necesidad de acceso a Google Drive.
"""

from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi
import os

load_dotenv()

HF_TOKEN     = os.getenv('HF_TOKEN')
REPO_ID      = 'Jesusrodriguezf90/chem-rag-artifacts'
DIR_EMBED    = Path('data/embeddings')

assert HF_TOKEN, 'HF_TOKEN no encontrado en .env'
assert (DIR_EMBED / 'PMC10967698_chunks.json').exists(), 'chunks.json no encontrado'
assert (DIR_EMBED / 'PMC10967698_embeddings.npy').exists(), 'embeddings.npy no encontrado'

api = HfApi(token=HF_TOKEN)

# Crear dataset si no existe
api.create_repo(
    repo_id=REPO_ID,
    repo_type='dataset',
    exist_ok=True,
    private=False,
)
print(f'Repositorio: https://huggingface.co/datasets/{REPO_ID}')

# Subir artefactos
for archivo in ['PMC10967698_chunks.json', 'PMC10967698_embeddings.npy']:
    api.upload_file(
        path_or_fileobj=str(DIR_EMBED / archivo),
        path_in_repo=archivo,
        repo_id=REPO_ID,
        repo_type='dataset',
        commit_message=f'feat: {archivo} — embeddings BGE-M3 del paper PMC10967698',
    )
    print(f'Subido: {archivo}')

print('\nArtefactos publicados correctamente.')
print(f'URL: https://huggingface.co/datasets/{REPO_ID}')