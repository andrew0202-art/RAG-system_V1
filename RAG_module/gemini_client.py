import time
from google import genai
from groq import Groq, RateLimitError
import RAG_module.api


def setup_gemini():
  # use Google gemini
  # return genai.Client()
  return Groq()


def ask_gemini(client, prompt, llm_model, max_retries = 5):
  # 延遲避免 resource exhausted
  # time.sleep(1)

  # use Google gemini
  # response = (
  #    client.models.generate_content(model = LLM_MODEL,
  #                                   contents = prompt)
  #)
  # return response.text
  for attempt in range(max_retries):
    try:
      response = client.chat.completions.create(
          model = llm_model,
          messages = [{"role": "user", "content": prompt}]
      )
      return response.choices[0].message.content

    except RateLimitError as e:
      if attempt == max_retries - 1:
        raise

      wait_seconds = 2 ** (attempt + 1)
      print(f"{attempt + 1}/{max_retries}: [LLM] encounters rate limit, wait for {wait_seconds} seconds to retry...")
      time.sleep(wait_seconds)