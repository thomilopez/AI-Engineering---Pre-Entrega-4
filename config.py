"""Configuración central del proyecto.

Carga PINECONE_API_KEY, OPENAI_API_KEY e INDEX_NAME desde .env
y expone constantes compartidas (dimensión, métrica, namespace, chunking).
Evita el Schema Drift: el esquema de metadatos vive en un solo lugar.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
INDEX_NAME = os.getenv("INDEX_NAME", "rag-scalable-index")

# Debe coincidir con text-embedding-3-small (1536 dims, métrica coseno).
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
METRIC = "cosine"
CLOUD = "aws"
REGION = "us-east-1"

# Namespace lógico: separa dev/staging/prod o tenants sin otro índice.
NAMESPACE = "docs-v1"

# Chunking: 600 tokens aprox (rango 500-800 pedido en clase).
CHUNK_SIZE = 600
CHUNK_OVERLAP = 60

# Esquema estricto de metadatos (una sola fuente de verdad).
METADATA_FIELDS = ("doc_id", "source", "text", "category", "author", "chunk_index")


def validate_env(require_openai: bool = True) -> None:
    """Falla rápido con mensaje claro si faltan variables de entorno."""
    missing = []
    if not PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")
    if require_openai and not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not INDEX_NAME:
        missing.append("INDEX_NAME")
    if missing:
        raise EnvironmentError(
            f"Faltan variables de entorno: {', '.join(missing)}. "
            "Copiá .env.example a .env y completalas."
        )
