# Chunking y Overlap

Los modelos de embedding tienen ventana de contexto limitada: un documento entero
pierde matices. El chunking crea unidades atómicas y semánticamente coherentes.

RecursiveCharacterTextSplitter es el estándar: prueba párrafos (\n\n), saltos (\n),
espacios y caracteres. La práctica común de la clase es chunk_size de 500 a 800
tokens con overlap de 50 a 60 tokens: en este proyecto usamos 600 con 60 de overlap.

El overlap repite el final del Chunk 1 al inicio del Chunk 2 para que ningún concepto
clave se pierda en el corte. Medir en caracteres en vez de tokens es el error número 1:
siempre validar con tiktoken. Chunks muy pequeños pierden contexto global; muy grandes
diluyen la precisión del embedding.
