from RAG_module.embedding_service import create_query_embedding
from RAG_module.qdrant_repository import search_points, build_metadata_filter
from RAG_module.config import COLLECTION_NAME


def _make_chunk(text, metadata, score = None):
  return {
      "text": text,
      "file_name": metadata.get("file_name"),
      "page_label": metadata.get("page_label"),  # None if the file is .txt
      "score": score,
  }


def run_retrieval(qdrant_client, bm25_retriever, reranker, query, embedding_model, top_k, reranker_top_k, metadata_filters = None):
  query_vector = create_query_embedding(query, embedding_model)
  query_filter = build_metadata_filter(metadata_filters)

  dense_results = search_points(qdrant_client,
                                COLLECTION_NAME,
                                query_vector,
                                limit = top_k,
                                filters = query_filter).points
  bm25_results = bm25_retriever.retrieve(query, limit = top_k, metadata_filters = metadata_filters)

  dense_chunks = {p.payload["text"]: _make_chunk(p.payload["text"],
                                                 {k: v for k, v in p.payload.items() if k != "text"},
                                                 p.score) for p in dense_results}

  bm25_chunks = {node.text: _make_chunk(node.text, node.metadata, score) for node, score in bm25_results}

  # combine the results
  merged = {**dense_chunks, **bm25_chunks}  # the union in dictionary
  all_chunks = list(merged.values())

  # reranking
  reranked_chunks = reranker.rerank(query, all_chunks, reranker_top_k)

  return {
      "dense_chunks": list(dense_chunks.values()),
      "bm25_chunks": list(bm25_chunks.values()),
      "all_chunks": list(all_chunks),
      "reranked_chunks": reranked_chunks
  }