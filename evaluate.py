"""Evaluación del retriever con Golden Set: Recall@5 y Precision@5.

Uso:
    python evaluate.py

Golden Set: 5 preguntas con su documento_id_esperado conocido.
- Recall@5 = % de preguntas donde el esperado aparece en el top-5 (hit rate).
- Precision@5 (MVP) = promedio de (1/5 si acierta, 0/5 si no) = Recall@5 / 5.
"""
from retriever import RAGSystem

K = 5

# El doc_id es el stem del archivo en /data (ver create_metadata en ingestion.py).
GOLDEN_SET = [
    {
        "pregunta": "¿Qué es un embedding y en qué se diferencia de one-hot?",
        "documento_id_esperado": "doc-embeddings",
    },
    {
        "pregunta": "¿Qué chunk_size y overlap recomienda la clase para no perder contexto?",
        "documento_id_esperado": "doc-chunking",
    },
    {
        "pregunta": "¿Qué operaciones CRUD ofrece ChromaDB y qué diferencia get de query?",
        "documento_id_esperado": "doc-chromadb",
    },
    {
        "pregunta": "¿Qué es un namespace en Pinecone y para qué sirve el batch upsert?",
        "documento_id_esperado": "doc-pinecone",
    },
    {
        "pregunta": "¿Qué es RRF y por qué no se deben sumar scores de BM25 con coseno?",
        "documento_id_esperado": "doc-hybrid",
    },
]


def evaluate() -> dict:
    """Ejecuta el Golden Set e imprime Recall@5 y Precision@5."""
    rag = RAGSystem()
    hits = 0
    print(f"Evaluando {len(GOLDEN_SET)} consultas (top-{K})...\n")
    for item in GOLDEN_SET:
        retrieved_ids = rag.retrieve_ids(item["pregunta"], k=K)
        expected = item["documento_id_esperado"]
        hit = expected in retrieved_ids
        hits += int(hit)
        mark = "HIT " if hit else "MISS"
        print(f"[{mark}] Q: {item['pregunta']}")
        print(f"       esperado={expected} | top-{K}={retrieved_ids}")

    recall_at_5 = hits / len(GOLDEN_SET)
    # MVP: 1 útil de 5 si hay hit -> precisión por query = (1/5 | 0); promedio general.
    precision_at_5 = recall_at_5 / K

    print("\n===== RESULTADOS =====")
    print(f"Recall@{K}:    {recall_at_5:.2%}  ({hits}/{len(GOLDEN_SET)} aciertos)")
    print(f"Precision@{K}: {precision_at_5:.2%}  (promedio 1/{K} por hit, 0 por miss)")
    return {"recall@5": recall_at_5, "precision@5": precision_at_5, "hits": hits}


if __name__ == "__main__":
    evaluate()
