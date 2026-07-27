import os
os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"
from sentence_transformers import SentenceTransformer
# from llama_index.core import Settings
# from google import genai
# import RAG_module.api


# _client = genai.Client()
_model = None


def _get_model(embedding_model):
  global _model
  if _model is None:
    print(f"載入本地 embedding 模型：{embedding_model}（第一次執行需要下載模型檔案，請稍候）")
    _model = SentenceTransformer(embedding_model)
  return _model


def create_document_embedding(text, embedding_model):
  model = _get_model(embedding_model)
  return model.encode(text, normalize_embeddings = True).tolist()

  #result = _client.models.embed_content(model = embedding_model,
  #                                      contents = text,
  #                                      config = {'task_type': "RETRIEVAL_DOCUMENT"})

  #return result.embeddings[0].values


def create_query_embedding(text, embedding_model):
  model = _get_model(embedding_model)
  return model.encode(text, normalize_embeddings = True).tolist()

  #result = _client.models.embed_content(model = embedding_model,
  #                                      contents = text,
  #                                      config = {"task_type": "RETRIEVAL_QUERY"})


  #return result.embeddings[0].values