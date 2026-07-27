from qdrant_client import (
    QdrantClient
)

def create_vector_store():
  return QdrantClient(path = "RAG_module/qdrant_db")