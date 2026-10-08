"""Inicialización idempotente del índice Pinecone Serverless.

Uso:
    python setup_pinecone.py

Verifica si el índice existe; si no, lo crea con dimensión 1536
(text-embedding-3-small) y métrica coseno en ServerlessSpec aws/us-east-1.
"""
from pinecone import Pinecone, ServerlessSpec

from config import (
    PINECONE_API_KEY,
    INDEX_NAME,
    EMBEDDING_DIMENSION,
    METRIC,
    CLOUD,
    REGION,
    validate_env,
)


def ensure_index_exists() -> str:
    """Crea el índice solo si no existe. Retorna el nombre del índice."""
    validate_env(require_openai=False)
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing = [i["name"] for i in pc.list_indexes()]
    if INDEX_NAME in existing:
        print(f"El índice '{INDEX_NAME}' ya existe. No se crea nada.")
    else:
        print(f"Creando índice '{INDEX_NAME}' (dim={EMBEDDING_DIMENSION}, metric={METRIC})...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,  # 1536 para text-embedding-3-small
            metric=METRIC,                  # coseno: estándar para embeddings OpenAI
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )
        print("Índice creado.")

    # Diagnóstico útil: cuántos vectores hay por namespace.
    stats = pc.Index(INDEX_NAME).describe_index_stats()
    print(f"Stats del índice: {stats}")
    return INDEX_NAME


if __name__ == "__main__":
    ensure_index_exists()
