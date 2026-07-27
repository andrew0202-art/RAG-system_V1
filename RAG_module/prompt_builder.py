def _format_location(chunk):
  location = chunk.get("file_name") or "未知來源"
  if chunk.get("page_label"):
    location += f" 第{chunk['page_label']}頁"
  return location


def build_prompt(query, chunks):
  citation_map = {}
  context_blocks = []

  for i, chunk in enumerate(chunks, start = 1):
    tag = f"S{i}"
    citation_map[tag] = {
        "file_name": chunk.get("file_name"),
        "page_label": chunk.get("page_label"),
        "text": chunk.get("text")
    }
    context_blocks.append(f"[{tag}] ({_format_location(chunk)})\n{chunk.get('text', '')}")

  context = "\n\n".join(context_blocks)
  available_tags = ", ".join(citation_map.keys()) if citation_map else "none"

  prompt = f"""You are a RAG assistant that must cite sources.
  Answer the question only according to the context below.
  Each context block starts with a source tag like [S1], [S2].

  Rules:
  - Before answering, check whether the context actually identifies the
    specific country/institution/regulation named in the question. If the
    context does not explicitly establish that it is about that subject, say
    exactly: "I don't know." Do NOT assume unlabeled provisions belong to the
    country or institution asked about just because the surrounding structure
    or topic (e.g. "board composition", "capital requirements") matches. This
    matters especially for the comparative central bank law documents, which
    cover many countries with parallel article structures (e.g. Israel, South
    Africa, Indonesia, Turkey, Bahamas, USA) — a matching section heading does
    NOT mean it is about the country asked about.
  - When the context contains a figure (e.g. a rate, ratio, amount) that
    changed multiple times over a time series (e.g. reserve ratio or discount
    rate adjustment history), first check whether the question specifies a
    particular date or period:
      - If a specific date/period is named in the question, use the value
        that was in effect at that date/period, not the latest value.
      - If no date is specified (e.g. the question asks for the "current" or
        unqualified value), use the LAST/FINAL value in the time series
        across all context blocks, and briefly verify no later context block
        revises it further.
  - If the answer is not found in the context, say exactly: "I don't know."
  - For every factual claim in your answer, immediately add the source tag(s)
    it is based on, e.g. "...as stated in the report [S2]."
  - Only use tags that literally appear in the context below ({available_tags}).
    Never invent a tag or cite something not present in the context.
  - When there are more than one answer, use more context information to select
    and propose the best answer.

  Context: {context}
  Question: {query}
  Answer:"""

  return prompt, citation_map