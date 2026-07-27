from sentence_transformers import CrossEncoder

class Reranker:
  def __init__(self, model_name):
    self.model = CrossEncoder(model_name)

  def rerank(self, query, chunks, top_k):
    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = self.model.predict(pairs)
    ranked = sorted(zip(chunks, scores),
                    key = lambda x: x[1],
                    reverse = True)
    reranked = []
    for chunk, score in ranked[:top_k]:
      chunk = dict(chunk)
      chunk["rerank_score"] = float(score)
      reranked.append(chunk)
    return reranked