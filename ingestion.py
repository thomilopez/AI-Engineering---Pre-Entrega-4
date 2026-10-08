"""Pipeline de ingesta: loader -> limpieza -> chunking -> upsert en Pinecone.

Uso:
    python ingestion.py [--reindex]

- Lee .txt/.md de /data con DirectoryLoader (TextLoader).
- Fragmenta con RecursiveCharacterTextSplitter medido en TOKENS (tiktoken),
  chunk_size=600, chunk_overlap=60 (rango 500-800 de la consigna).
- Ingesta inteligente: guarda texto original + fuente/categoría/autor en la
  metadata de Pinecone (no como texto plano desconectado).
- Sube en batches de 100 (batch upsert) al namespace configurado.
- Por defecto NO reindexa si el namespace ya tiene vectores (ahorra tiempo/costo).
"""
import argparse
import re
import tiktoken
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

from config import (
    DATA_DIR,
    PINECONE_API_KEY,
    INDEX_NAME,
    NAMESPACE,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    validate_env,
)
from setup_pinecone import ensure_index_exists

BATCH_SIZE = 100
_tokenizer = tiktoken.get_encoding("cl100k_base")


def clean_text(text: str) -> str:
    """Normaliza ruido típico de extracciones (espacios, saltos, líneas vacías)."""
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def token_len(text: str) -> int:
    """Mide en tokens, no en caracteres (error #1: perder contexto global)."""
    return len(_tokenizer.encode(text))


def create_metadata(source: str, chunk_index: int, text: str) -> dict:
    """Esquema único de metadatos. Cambiarlo aquí cambia todo el pipeline."""
    stem = source.replace(".md", "").replace(".txt", "")
    return {
        "doc_id": stem,          # ID lógico del documento (usado por evaluate.py)
        "source": source,        # archivo de origen (trazabilidad)
        "text": text[:4000],     # texto original del chunk (evita otra DB)
        "category": "tech-docs", # categoría para hard filters
        "author": "ai-engineering",
        "chunk_index": chunk_index,
    }


def load_and_chunk():
    """Carga /data y devuelve chunks con metadata enriquecida."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"No existe {DATA_DIR}. Creala y agregá .txt/.md.")
    loader = DirectoryLoader(
        str(DATA_DIR), glob="**/*.{txt,md}",
        loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()
    if not docs:
        raise ValueError("No se encontraron .txt/.md en /data.")
    for d in docs:
        d.page_content = clean_text(d.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,       # ~600 tokens: ni pierde contexto ni diluye precisión
        chunk_overlap=CHUNK_OVERLAP, # preserva continuidad en el corte
        length_function=token_len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    # Metadata inteligente por chunk.
    for i, c in enumerate(chunks):
        source = c.metadata.get("source", "desconocido")
        c.metadata.update(create_metadata(source, i, c.page_content))
        c.id = f"{c.metadata['doc_id']}-chunk-{i}"  # ID determinista -> upsert idempotente
    return chunks


def ingest(force_reindex: bool = False) -> int:
    """Indexa en Pinecone. Retorna cantidad de chunks subidos (0 si ya existía)."""
    validate_env()
    ensure_index_exists()
    pc = Pinecone(api_key=PINECONE_API_KEY)
    stats = pc.Index(INDEX_NAME).describe_index_stats()
    ns_count = stats.get("namespaces", {}).get(NAMESPACE, {}).get("vector_count", 0)
    if ns_count > 0 and not force_reindex:
        print(f"Namespace '{NAMESPACE}' ya tiene {ns_count} vectores. Uso --reindex para forzar.")
        return 0

    chunks = load_and_chunk()
    print(f"Subiendo {len(chunks)} chunks al namespace '{NAMESPACE}' en batches de {BATCH_SIZE}...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)  # MISMO modelo que retrieval
    # Batch upsert manual (evita 1 request por vector).
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        PineconeVectorStore.from_documents(
            documents=batch,
            embedding=embeddings,
            index_name=INDEX_NAME,
            namespace=NAMESPACE,
        )
        print(f"  batch {start // BATCH_SIZE + 1}: {len(batch)} chunks")
    print("Ingesta completada.")
    return len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reindex", action="store_true", help="Reindexa aunque haya datos.")
    args = parser.parse_args()
    ingest(force_reindex=args.reindex)
