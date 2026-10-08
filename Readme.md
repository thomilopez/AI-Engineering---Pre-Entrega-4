# Sistema RAG Escalable en la Nube con Pinecone — Pre-Entrega 4

Búsqueda híbrida (BM25 léxico + Pinecone semántico, fusión RRF vía `EnsembleRetriever`)
con ingesta por batches, metadatos estrictos y evaluación Recall@5 / Precision@5.

## Estructura

```
.
├── config.py           # env (PINECONE_API_KEY, OPENAI_API_KEY, INDEX_NAME) + constantes
├── setup_pinecone.py   # crea el índice Serverless si no existe (1536, coseno)
├── ingestion.py        # loader -> chunking 600/60 por tokens -> batch upsert + metadata
├── retriever.py        # clase RAGSystem (EnsembleRetriever BM25 + Pinecone, top-5)
├── evaluate.py         # Golden Set de 5 preguntas -> Recall@5 y Precision@5
├── data/               # 5 docs técnicos de ejemplo (doc-embeddings, chunking, ...)
├── requirements.txt
├── .env.example
└── README.md
```

## 1. Clonar e instalar

```bash
git clone <tu-repo>.git
cd <tu-repo>
python -m venv venv
# Windows: venv\Scripts\activate | Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

## 2. Configurar `.env`

```bash
cp .env.example .env
```

Editar `.env`:

```
PINECONE_API_KEY=...
OPENAI_API_KEY=...
INDEX_NAME=rag-scalable-index
```

> Nunca subir `.env` a Git (ya está en `.gitignore`).

## 3. Replicar el índice Pinecone

```bash
python setup_pinecone.py
```

- Verifica si `INDEX_NAME` existe en Pinecone Serverless.
- Si no existe, lo crea: `dimension=1536` (compatible con `text-embedding-3-small`),
  `metric=cosine`, `ServerlessSpec(cloud="aws", region="us-east-1")`.
- Imprime `describe_index_stats()` para verificar.

## 4. Correr la ingesta

```bash
python ingestion.py
# Forzar reindexado:
python ingestion.py --reindex
```

- Carga `.txt/.md` de `/data` (`DirectoryLoader` + `TextLoader`).
- Chunking `RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)`
  medido en tokens con `tiktoken` (rango 500-800 de la consigna).
- Embeddings `text-embedding-3-small` — **mismo modelo en ingesta y retrieval**.
- Ingesta inteligente: cada vector guarda en metadata `doc_id, source, text,
  category, author, chunk_index` (el texto original viaja con el vector).
- Batch upsert de 100 vectores al `namespace="docs-v1"`.
- Idempotente: si el namespace ya tiene vectores, no reindexa (ahorra costo).

## 5. Ejecutar la evaluación

```bash
python evaluate.py
```

Salida esperada:

```
[HIT ] Q: ¿Qué es un embedding...?  esperado=doc-embeddings | top-5=[...]
...
===== RESULTADOS =====
Recall@5:    100.00%  (5/5 aciertos)
Precision@5: 20.00%  (promedio 1/5 por hit, 0 por miss)
```

- `Recall@5` = % de preguntas donde el `documento_id_esperado` está en el top-5.
- `Precision@5` (MVP) = promedio de 1/5 por hit: si acierta, 1 de 5 es útil.

## Decisiones de diseño

| Decisión | Por qué |
|---|---|
| `chunk 600/60` por tokens | Punto medio 500-800: contexto sin diluir precisión |
| Namespace `docs-v1` | Multi-tenancy / dev vs prod sin otro índice |
| Metadata estricta en `config.py` | Evita Schema Drift (`ingest_date` vs `date_created`) |
| `EnsembleRetriever` (0.4 BM25 + 0.6 vectorial) | RRF implícito: no suma scores heterogéneos |
| Mismo embedding en ingesta y query | Evita resultados aleatorios (error #1) |
| Batch upsert 100 | Equilibrio red/estabilidad (clase sugiere 100-200) |

## Errores evitados (rúbrica)

- Mismatch de dimensiones: índice 1536 = `text-embedding-3-small`.
- Namespace ignorado: todo va a `docs-v1`.
- Chunking extremo: 600 tokens, no 100 ni 2000.
- Suma de scores BM25+coseno: delegada a RRF del `EnsembleRetriever`.
