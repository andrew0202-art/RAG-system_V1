import re
import time
from groq import Groq, RateLimitError
import RAG_module.api

_client = Groq()


def _judge_faithfulness(question, answer_text, combined_source_text, judge_model, max_retries = 5):
  prompt = f"""You are a fact-checking assistant.
  You are given the ORIGINAL QUESTION, a set of SOURCE TEXT excerpts
  (separated by "---", each prefixed with a tag like [S1]), and an ANSWER
  that was generated based on these source texts.

  Determine whether the source texts, taken TOGETHER, support the answer
  as a whole. The answer must be about the same specific subject/entity/case
  as stated in the source texts, not merely about the same general topic.
  It is fine if no single excerpt alone is sufficient, as long as they
  jointly support the answer.
  Reply with only "yes" or "no".

  Original question: {question}

  Source text(s): {combined_source_text}

  Answer: {answer_text}

  Do the source texts, taken together, support this answer?"""

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
      print(f"{attempt + 1}/{max_retries}: [Faithfulness Checker] encounters rate limit, wait for {wait_seconds} seconds to retry...")
      time.sleep(wait_seconds)


def check_faithfulness(question, answer_text, citation_map, judge_model):
  if not citation_map:
    return {"supported": False, "reason": "no context retrieved to check against"}

  combined_source_text = "\n---\n".join(
      f"[{tag}] {info['text']}" for tag, info in citation_map.items()
  )

  supported = _judge_faithfulness(question, answer_text, combined_source_text, judge_model)

  return {"supported": supported}