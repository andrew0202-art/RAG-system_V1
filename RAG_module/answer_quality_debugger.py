from RAG_module.retrieval_pipeline import run_retrieval
from RAG_module.evaluator import normalize, judge_answer
from RAG_module.reranker import Reranker
from RAG_module.prompt_builder import build_prompt
from RAG_module.gemini_client import ask_gemini
from RAG_module.experiment_tracker import save_experiment
from RAG_module.loader import load_documents
from RAG_module.chunker import create_nodes
from RAG_module.metadata_loader import load_document_metadata
from RAG_module.bm25_retriever import BM25Retriever
from RAG_module.citation_verifier import verify_citations
from RAG_module.faithfulness_checker import check_faithfulness

def answer_quality_debug(qdrant_client,
                         gemini_client,
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
                         notes = ""):

  # step 1: initialize reranker
  from llama_index.core.node_parser import SentenceSplitter
  documents = load_documents()
  nodes = create_nodes(documents, chunk_size, chunk_overlap)

  metadata_lookup = load_document_metadata()
  for node in nodes:
    file_name = node.metadata.get("file_name")
    extracted = metadata_lookup.get(file_name, {})
    node.metadata.update(extracted)

  bm25_retriever = BM25Retriever(nodes)
  reranker = Reranker(reranker_model)

  # step 2: perform the diagnosis for each question in the evaluation set
  diagnosis_results = []

  for item in dataset:
    query = item["question"]
    expected = item["answer"]
    source = item["source"]
    metadata_filters = item.get("expected_filters")

    # run retrieval
    result = run_retrieval(qdrant_client, bm25_retriever, reranker, query, embedding_model, top_k, reranker_top_k, metadata_filters = metadata_filters)

    # recall 有沒有 hit（跟 recall_debugger 邏輯一致）
    if source is None:
      recall_hit = None  # 這題不適用 recall 判斷

    else:
      recall_hit = any(normalize(source) in normalize(chunk["text"]) for chunk in result["all_chunks"])

    # rerank
    reranked_chunks = result["reranked_chunks"]

    # LLM 回答
    prompt, citation_map = build_prompt(query, reranked_chunks)
    actual = ask_gemini(gemini_client, prompt, llm_model)

    # judge
    correct = judge_answer(query, expected, actual, judge_model)

    # citation verification and faithfulness checking
    citation_check = verify_citations(actual, citation_map)
    faithfulness_check = check_faithfulness(query, actual, citation_map, judge_model)

    diagnosis_results.append({
        "question": query,
        "expected": expected,
        "actual": actual,
        "correct": correct,
        "recall_hit": recall_hit,
        "has_hallucinated_citation": citation_check["has_hallucinated_citation"],
        "invalid_tags": citation_check["invalid_tags"],
        "faithfulness_supported": faithfulness_check["supported"],
    })

  # step 3: print the diagnostic report and save
  print(f"\n{'='*60}")
  print(f"Answer Quality Diagnosis | top_k = {top_k}, reranker_top_k = {reranker_top_k}")
  print(f"{'='*60}\n")

  correct_count = 0
  for r in diagnosis_results:
    correct_count += r["correct"]

    print(f"Question: {r['question']}")
    print(f"  expected     : {r['expected']}")
    print(f"  actual       : {r['actual'].strip()}")
    print(f"  correct      : {'✅' if r['correct'] else '❌'}")
    print(f"  citation OK  : {'❌ 幻覺引用 ' + str(r['invalid_tags']) if r['has_hallucinated_citation'] else '✅'}")
    print(f"  faithfulness : {'✅' if r['faithfulness_supported'] else '❌'}")

    if not r["correct"]:
      if r["recall_hit"] is False:
        print(f"problem source: Retrieval, recall does not hit, come back to Phase 1")

      elif r["recall_hit"] is None:
        print(f"the expected answer is 'I don't know', check if the actual answer is semantically equivalent to the expected one")

      else:
        print(f"Recall hits but the answer is wrong, see below")
        print(f"if the actual answer is in fact correct: Judge, consider more powerful judge models")
        print(f"if the actual answer is truly incorrect: Prompt, consider to change the prompt_builder")
    else:
      print(f"✅ All pass")

    print()

  score = correct_count / len(diagnosis_results)
  citation_validity = sum(not r["has_hallucinated_citation"] for r in diagnosis_results) / len(diagnosis_results)
  faithfulness_score = sum(r["faithfulness_supported"] for r in diagnosis_results) / len(diagnosis_results)

  print(f"{'=' * 60}")
  print(f"Answer Quality: {score:.3f}")
  print(f"{'=' * 60}\n")

  # record the results
  save_experiment(
      params = {"chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "top_k": top_k,
                "reranker_top_k": reranker_top_k,
                "embedding_model": embedding_model,
                "reranker_model": reranker_model,
                "llm_model": llm_model,
                "judge_model": judge_model},
      metrics = {"answer_quality": round(score, 3),
                "citation_validity": round(citation_validity, 3),
                "faithfulness": round(faithfulness_score, 3)},
      notes = notes
  )