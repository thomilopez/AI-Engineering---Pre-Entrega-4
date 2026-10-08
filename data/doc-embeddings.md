# Embeddings vs One-Hot Encoding

Un embedding es una lista de números (un vector denso) que captura el significado
de un fragmento de texto. El modelo text-embedding-3-small de OpenAI genera
vectores de 1536 dimensiones.

Antes se usaba One-Hot Encoding: cada palabra era una posición en un vector gigante
de ceros. La distancia entre "perro" y "canino" era igual que entre "perro" y
"microprocesador": no había noción de significado.

Con embeddings densos y contextuales (BERT, Ada, Voyage), la similitud coseno entre
vectores refleja cercanía semántica. La regla de oro: con OpenAI o HuggingFace, usar
similitud coseno. Nunca mezclar modelos de embeddings entre indexación y consulta:
viven en espacios vectoriales distintos y los resultados serían aleatorios.
