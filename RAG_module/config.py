from google.colab import userdata
import os

# external data
DOCUMENT_DIR = "RAG_module/data"

# chunking related
CHUNK_SIZE = 450
CHUNK_OVERLAP = 120

# embedding model
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
# EMBEDDING_MODEL = "gemini-embedding-001"
# EMBEDDING_DIM = 3072

# collection
COLLECTION_NAME = "demo"

# retrieval
TOP_K = 12

# reranking related
#RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_TOP_K = 8

# LLM
LLM_MODEL = "llama-3.1-8b-instant"
JUDGE_MODEL = "openai/gpt-oss-120b"
# LLM_MODEL = "gemini-2.0-flash-lite"
# JUDGE_MODEL = "gemini-2.0-flash-lite"