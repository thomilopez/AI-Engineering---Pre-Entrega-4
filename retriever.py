"""Recuperador híbrido: BM25 (léxico) + Pinecone (semántico) via EnsembleRetriever.

La fusión usa RRF internamente (por ranking, no suma directa de scores
heterogéneos). Retorna top-5.
"""
from langchain_classic.retrievers import BM25Retriever, EnsembleRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from config import INDEX_NAME, NAMESPACE, EMBEDDING_MODEL, validate_env
from ingestion import load_and_chunk


class RAGSystem:
    """Fachada POO del retrieval híbrido. Reutiliza el MISMO embedding que ingesta."""

    TOP_K = 5

    def __init__(self, weights: tuple[float, float] = (0.4, 0.6)) -> None:
        validate_env()
        # BM25 necesita los documentos en memoria: mismos chunks que se indexaron.
        try:
            self._chunks = load_and_chunk()
        except Exception as e:
            raise RuntimeError(f"No se pudo cargar /data para BM25: {e}") from e

        bm25 = BM25Retriever.from_documents(self._chunks)
        bm25.k = self.TOP_K

        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vectorstore = PineconeVectorStore(
            index_name=INDEX_NAME, embedding=embeddings, namespace=NAMESPACE
        )
        pinecone_retriever = vectorstore.as_retriever(
            search_kwargs={"k": self.TOP_K}
        )

        # EnsembleRetriever fusiona por RRF: no suma scores de escalas distintas.
        self.hybrid = EnsembleRetriever(
            retrievers=[bm25, pinecone_retriever],
            weights=list(weights),  # (léxico, semántico)
        )

    def retrieve(self, query: str, k: int = TOP_K):
        """Devuelve hasta k documentos híbridos para la consulta."""
        if not query or not query.strip():
            raise ValueError("La consulta no puede estar vacía.")
        docs = self.hybrid.invoke(query)
        return docs[:k]

    def retrieve_ids(self, query: str, k: int = TOP_K) -> list[str]:
        """Atajo para evaluación: solo los doc_id recuperados."""
        return [
            (d.metadata or {}).get("doc_id", d.metadata.get("source", "?"))
            for d in self.retrieve(query, k=k)
        ]
