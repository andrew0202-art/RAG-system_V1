from RAG_module.config import COLLECTION_NAME, EMBEDDING_DIM, DOCUMENT_DIR
from RAG_module.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, RERANKER_TOP_K, LLM_MODEL, EMBEDDING_MODEL, RERANKER_MODEL, JUDGE_MODEL
from RAG_module.loader import load_documents
from RAG_module.metadata_loader import load_document_metadata, CATEGORIES, DOC_TYPES
from RAG_module.kb_manager import is_kb_outdated, save_checksum
from RAG_module.chunker import create_nodes
from RAG_module.embedding_service import create_document_embedding, create_query_embedding
from RAG_module.vector_store import create_vector_store
from RAG_module.qdrant_repository import (
    create_collection,
    collection_exists,
    create_points,
    upsert_points
)
from RAG_module.retrieval_pipeline import run_retrieval
from RAG_module.bm25_retriever import BM25Retriever
from RAG_module.reranker import Reranker
from RAG_module.prompt_builder import build_prompt
from RAG_module.gemini_client import setup_gemini, ask_gemini
from RAG_module.citation_verifier import verify_citations
from RAG_module.faithfulness_checker import check_faithfulness


def main():
  # 0: initializations
  reranker = Reranker(RERANKER_MODEL)
  gemini_client = setup_gemini()
  qdrant_client = create_vector_store()

  # 1: load the documents
  documents = load_documents()

  # 2: chunking the Documents into TextNodes with metadata extraction
  nodes = create_nodes(documents, CHUNK_SIZE, CHUNK_OVERLAP)
  metadata_lookup = load_document_metadata()
  for node in nodes:
    file_name = node.metadata.get("file_name")
    extracted = metadata_lookup.get(file_name, {})
    node.metadata.update(extracted)

  bm25_retriever = BM25Retriever(nodes)

  outdated, current_checksum = is_kb_outdated(DOCUMENT_DIR, EMBEDDING_MODEL)
  if outdated or not collection_exists(qdrant_client, COLLECTION_NAME):
    ## start building the knowledge base ##
    print("building knowledge base...")

    # 3: transform TextNodes into Points to be stored in Qdrant
    points = create_points(nodes, create_document_embedding, EMBEDDING_MODEL)

    # 4: create collection
    create_collection(qdrant_client,
                      COLLECTION_NAME,
                      EMBEDDING_DIM)

    # 5: upsert the points into the collection created
    upsert_points(qdrant_client,
                  COLLECTION_NAME,
                  points)

    # save the current checksum
    save_checksum(current_checksum)

    print("knowledge base created.")
    ## end building the knowledge base ##

  else:
    print("Using existing knowledge base.")

  print("設定檢索篩選條件（每一項都可以直接按 Enter 跳過，不套用該項）")
  print()

  metadata_filters = {}

  print("主題分類：")
  for cat in CATEGORIES:
    print(f"- {cat}")

  category_input = input("輸入主題分類關鍵字（例如「氣候」）：").strip()
  if category_input:
    matches = [c for c in CATEGORIES if category_input in c]
    if len(matches) == 1:
      metadata_filters["category"] = matches[0]
      print(f"  已套用：category = {matches[0]}")
    elif len(matches) > 1:
      print(f"  關鍵字對應到多個分類：{matches}，這項不套用篩選")
    else:
      print(f"  找不到符合的分類，這項不套用篩選")

  print()
  print("文件性質：")
  for dt in DOC_TYPES:
    print(f"- {dt}")

  doctype_input = input("輸入文件性質關鍵字（例如「政策報告」）：").strip()
  if doctype_input:
    matches = [dt for dt in DOC_TYPES if doctype_input in dt]
    if len(matches) == 1:
      metadata_filters["doc_type"] = matches[0]
      print(f"  已套用：doc_type = {matches[0]}")
    elif len(matches) > 1:
      print(f"  關鍵字對應到多個文件性質：{matches}，這項不套用篩選")
    else:
      print(f"  找不到符合的文件性質，這項不套用篩選")

  if not metadata_filters:
    metadata_filters = None
  print()

  try:

    while True:
      query = input("\nQuestion: ")

      if query.lower() == "exit":
        break

      # 6: the retrieval step
      retrieval_result = run_retrieval(qdrant_client,
                                       bm25_retriever,
                                       reranker,
                                       query,
                                       EMBEDDING_MODEL,
                                       TOP_K,
                                       RERANKER_TOP_K,
                                       metadata_filters = metadata_filters)
      reranked_chunks = retrieval_result["reranked_chunks"]

      # 7: construct the augmented prompt
      prompt, citation_map = build_prompt(query,
                                          reranked_chunks)

      # 8: obtain the LLM's response with the augmented query as input
      answer = ask_gemini(gemini_client,
                          prompt,
                          LLM_MODEL)

      print()
      print(answer)
      print()

      # check if citations are valid, meaning that they exist in the citation map
      print("本次可引用的來源：")
      for tag, info in citation_map.items():
        loc = info["file_name"] or "未知來源"
        if info["page_label"]:
          loc += f" 第{info['page_label']}頁"
        print(f"  {tag}: {loc}")
      print()

      verification = verify_citations(answer, citation_map)
      if verification["has_hallucinated_citation"]:
        print(f"⚠️ 偵測到幻覺引用：{verification['invalid_tags']}（這些標籤不存在於本次 context 中）")
      else:
        print("✅ 引用標籤檢查通過")
      print()

      # check if valid citations truly support the LLM's answer
      faithfulness = check_faithfulness(query, answer, citation_map, JUDGE_MODEL)
      if faithfulness["supported"]:
        print("✅ 引用內容語意查核通過")
      else:
        print("⚠️ 這一輪檢索到的來源，合起來似乎無法支持這個回答")
      print()

  finally:

    # close the client
    qdrant_client.close()

if __name__ == "__main__":
  main()