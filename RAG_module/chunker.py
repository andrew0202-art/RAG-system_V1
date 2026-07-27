from llama_index.core.node_parser import SentenceSplitter

def create_chunker(chunk_size, chunk_overlap):
  return SentenceSplitter(chunk_size = chunk_size,
                          chunk_overlap = chunk_overlap)


def create_nodes(documents, chunk_size, chunk_overlap):
  chunker = create_chunker(chunk_size, chunk_overlap)
  nodes = chunker.get_nodes_from_documents(documents)

  return nodes