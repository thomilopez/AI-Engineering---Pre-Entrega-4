# Pinecone Serverless y Namespaces

Pinecone es una base de datos vectorial nativa de la nube: arquitectura serverless en
aws/us-east-1, sin gestionar nodos, con separación de cómputo y almacenamiento.

Conceptos: Index (contenedor con dimensión y métrica), Namespace (partición lógica
para multi-tenancy o dev/staging/prod) y metadatos JSON para hard filters.

El flujo es ingesta (chunking, embedding, batch upsert de 100-200 vectores) y
recuperación (vectorizar consulta, ANN search, top-k con filtros). Guardar el texto
original en la metadata evita consultas extra a otra base de datos.

Ignorar namespaces mezcla datos de prueba con producción y degrada seguridad.
