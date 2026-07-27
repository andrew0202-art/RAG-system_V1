import re

def extract_cited_tags(answer_text):
  return set(re.findall(r"\[S\d+\]", answer_text))


def verify_citations(answer_text, citation_map):
  # check whether the labels cited truly exist in the citation map
  cited_tags = extract_cited_tags(answer_text)
  valid_keys = {f"[{tag}]" for tag in citation_map.keys()}

  valid_tags = cited_tags & valid_keys
  invalid_tags = cited_tags - valid_keys

  return {
      "cited_tags": cited_tags,
      "valid_tags": valid_tags,
      "invalid_tags": invalid_tags,
      "has_hallucinated_citation": len(invalid_tags) > 0}