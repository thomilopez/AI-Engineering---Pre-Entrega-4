# ChromaDB: CRUD en el espacio vectorial

ChromaDB es una base de datos vectorial local con persistencia en disco. Ideal para
prototipado rápido sin Docker.

Operaciones CRUD: upsert crea o actualiza según el ID; get() busca por IDs o
metadatos exactos (como SELECT); query() busca por similitud semántica y es el
corazón de RAG; update re-vectoriza; delete elimina vectores irrelevantes.

Los metadatos (autor, fecha, fuente) filtran antes de la búsqueda y permiten citar
fuentes. Usar IDs deterministas (hash o doc_id + chunk) permite updates eficientes.
Siempre inicializar con persist_directory para no perder datos al cerrar el script.
