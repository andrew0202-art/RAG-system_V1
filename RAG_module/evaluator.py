import json
import re
import time
import unicodedata
import RAG_module.api
from google import genai
from groq import Groq, RateLimitError
from RAG_module.citation_verifier import verify_citations
from RAG_module.faithfulness_checker import check_faithfulness


def normalize(text):
  text = unicodedata.normalize('NFKC', text)
  return re.sub(r'\s+', '', text).lower()


def load_eval_dataset(path = "RAG_module/eval_dataset.json"):
  with open(path, "r") as f:
    return json.load(f)


def eval_retrieval(dataset, retrieve_fn, configs):
  results = []

  for item in dataset:
    if item["source"] is None:
      continue

    retrieved_texts = retrieve_fn(item["question"], configs, item.get("expected_filters"))
    hit = any(normalize(item["source"]) in normalize(text) for text in retrieved_texts)

    results.append({
        "question": item["question"],
        "source": item["source"],
        "hit": hit
    })

  recall = sum(r["hit"] for r in results) / len(results)

  return recall, results


#_client = genai.Client()
_client = Groq()

def judge_answer(question, expected, actual, judge_model, max_retries = 5):
  prompt = f"""You are an evaluation assistant.
  Judge whether the actual answer correctly answers the question.
  The correctness is based on the information provided in the expected answer.
  Note that the actual answer might provide some redundant information, which
  should not affect whether the actual answer is correct or not. Also, if both
  the expected answer and actual answer are like "I don't know", then the actual
  answer correctly answers the question. Reply with only "yes" or "no".

  Question: {question}
  Expected answer: {expected}
  Actual answer: {actual}

  Is the actual answer correct?"""

  for attempt in range(max_retries):
    try:
      response = _client.chat.completions.create(
          model = judge_model,
          messages = [{"role": "user", "content": prompt}]
      )
      return response.choices[0].message.content.strip().lower() == "yes"

    except RateLimitError as e:
      if attempt == max_retries - 1:
        raise
      wait_seconds = 2 ** (attempt + 1)
      print(f"{attempt + 1}/{max_retries}: [Judge] encounters rate limit, wait for {wait_seconds} seconds to retry...")
      time.sleep(wait_seconds)

  # with google gemini
  #response = _client.models.generate_content(
  #    model = JUDGE_MODEL,
  #    contents = prompt
  #)

  # return response.text.strip().lower() == "yes"

def eval_answer(dataset, rag_fn, configs):
  results = []

  for item in dataset:
    actual, citation_map = rag_fn(item["question"], configs, item.get("expected_filters"))

    correct = judge_answer(item["question"], item["answer"], actual, configs["judge_model"])
    citation_check = verify_citations(actual, citation_map)
    faithfulness_check = check_faithfulness(item["question"], actual, citation_map, configs["judge_model"])

    results.append({
        "question": item["question"],
        "expected": item["answer"],
        "actual": actual,
        "correct": correct,
        "has_hallucinated_citation": citation_check["has_hallucinated_citation"],
        "invalid_tags": citation_check["invalid_tags"],
        "faithfulness_supported": faithfulness_check["supported"],
    })

  n = len(results)
  scores = {
      "answer_quality": sum(r["correct"] for r in results) / n,
      "citation_validity": sum(not r["has_hallucinated_citation"] for r in results) / n,
      "faithfulness": sum(r["faithfulness_supported"] for r in results) / n,
  }

  return scores, results