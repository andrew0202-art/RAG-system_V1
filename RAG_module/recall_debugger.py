from RAG_module.retrieval_pipeline import run_retrieval
from RAG_module.experiment_tracker import save_experiment
from RAG_module.chunker import create_nodes
from RAG_module.loader import load_documents
from RAG_module.bm25_retriever import BM25Retriever
from RAG_module.reranker import Reranker
from RAG_module.embedding_service import create_document_embedding
from RAG_module.qdrant_repository import create_points, create_collection, upsert_points
from RAG_module.evaluator import normalize
from RAG_module.metadata_loader import load_document_metadata
from RAG_module.config import COLLECTION_NAME, DOCUMENT_DIR


def recall_debug(qdrant_client,
                 dataset,
                 chunk_size,
                 chunk_overlap,
                 embedding_model,
                 embedding_dim,
                 reranker_model,
                 top_k,
                 reranker_top_k,
                 llm_model,
                 judge_model,
                 rebuild_kb = False,
                 notes = ""):

  # step 1: rebuild nodes with metadata
  from llama_index.core.node_parser import SentenceSplitter
  documents = load_documents()
  nodes = create_nodes(documents, chunk_size, chunk_overlap)

  metadata_lookup = load_document_metadata()
  for node in nodes:
    file_name = node.metadata.get("file_name")
    extracted = metadata_lookup.get(file_name, {})
    node.metadata.update(extracted)

  # step 2: rebuild BM25
  bm25_retriever = BM25Retriever(nodes)

  # step 3: rebuild KB if necessary
  if rebuild_kb:
    create_collection(qdrant_client, COLLECTION_NAME, embedding_dim)
    points = create_points(nodes, create_document_embedding, embedding_model)
    upsert_points(qdrant_client, COLLECTION_NAME, points)

  # step 4: perform the diagnosis for each question in the evaluation set
  reranker = Reranker(reranker_model)
  diagnosis_results = []

  for item in dataset:
    if item["source"] is None:
      continue

    source = item["source"]
    query = item["question"]
    metadata_filters = item.get("expected_filters")

    # layer 1: does the chunk exist?
    chunk_exists = any(normalize(source) in normalize(node.text) for node in nodes)

    # layer 2: are the keywords retrieved?
    result = run_retrieval(qdrant_client, bm25_retriever, reranker, query, embedding_model, top_k, reranker_top_k, metadata_filters = metadata_filters)
    dense_hit = any(normalize(source) in normalize(chunk["text"]) for chunk in result["dense_chunks"])
    bm25_hit = any(normalize(source) in normalize(chunk["text"]) for chunk in result["bm25_chunks"])

    # layer 3: are the keywords still captured after reranking?
    all_chunks = result["all_chunks"]
    rerank_hit = any(normalize(source) in normalize(chunk["text"]) for chunk in result["reranked_chunks"])

    diagnosis_results.append({
        "question": query,
        "source": source,
        "chunk_exists": chunk_exists,
        "dense_hit": dense_hit,
        "bm25_hit": bm25_hit,
        "rerank_hit": rerank_hit,
        "all_chunks": all_chunks
    })

  # step 5: print the diagnostic report
  print(f"\n{'='*60}")
  print(f"診斷參數：chunk_size = {chunk_size}, chunk_overlap = {chunk_overlap}, top_k = {top_k}, reranker_top_k = {reranker_top_k}")
  print(f"{'='*60}\n")

  recall_hits = 0
  for r in diagnosis_results:
    recall_hits += r["rerank_hit"]

    print(f"問題：{r['question']}")
    print(f"  source         : {r['source']}")
    print(f"  chunk exists   : {'✅' if r['chunk_exists'] else '❌'}")
    print(f"  dense hit      : {'✅' if r['dense_hit'] else '❌'}")
    print(f"  BM25 hit       : {'✅' if r['bm25_hit'] else '❌'}")
    print(f"  rerank hit     : {'✅' if r['rerank_hit'] else '❌'}")

    # diagnosis for implied actions
    if not r["chunk_exists"]:
      print(f"problem source: Chunking, adjust chunk_size or chunk_overlap")

    elif not r["dense_hit"] and not r["bm25_hit"]:
      print(f"problem source: Retrieval fails，consider to increase top_k or change embedding model")

    elif not r["dense_hit"]:
      print(f"problem source: keywords not in dense search but in BM25, consider to increase top_k")

    elif not r["bm25_hit"]:
      print(f"problem source: keywords not in BM25 but in dense search, consider to improve tokenization or increase top_k")

    elif not r["rerank_hit"]:
      dropped_by_reranker = [chunk for chunk in r["all_chunks"] if normalize(source) in normalize(chunk["text"])]
      print(f"problem source: Reranking, manually examine the contents for problematic chunks")
      for chunk in dropped_by_reranker:
        print(f"     ---")
        print(f"     ({chunk.get('file_name')} {chunk.get('page_label') or ''})")
        print(f"     {chunk["text"][:200]}")

      print(f"possible directions: adjust chunk_size/chunk_overlap, increase reranker_top_k, or change reranker model")

    else:
      print(f"✅ All pass")
    print()

  recall = recall_hits / len(diagnosis_results)
  print(f"{'=' * 60}")
  print(f"Recall@{reranker_top_k}: {recall:.3f}")
  print(f"{'=' * 60}\n")

  save_experiment(
      params = {"chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "top_k": top_k,
                "reranker_top_k": reranker_top_k,
                "embedding_model": embedding_model,
                "reranker_model": reranker_model,
                "llm_model": llm_model,
                "judge_model": judge_model},
      metrics = {"recall": round(recall, 3)},
      notes = notes
  )