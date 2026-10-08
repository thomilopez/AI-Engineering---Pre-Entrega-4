# Búsqueda híbrida: BM25 + Vectores con RRF

BM25 es búsqueda léxica por coincidencia exacta (frecuencia y rareza): excelente para
nombres propios, códigos de error y términos técnicos. Los embeddings capturan
semántica y sinónimos. La búsqueda híbrida combina ambos con EnsembleRetriever.

RRF (Reciprocal Rank Fusion) fusiona rankings por posición, no por score absoluto:
robusto porque BM25 y similitud coseno tienen escalas heterogéneas. Nunca sumar
scores directamente. El re-ranking con cross-encoder refina el top-k final con alta
precisión pero alto costo: solo sobre candidatos, nunca sobre millones.

Recall@k mide si el documento relevante aparece en el top-k; Precision@k qué
proporción del top-k es útil. F1 resume ambas.
