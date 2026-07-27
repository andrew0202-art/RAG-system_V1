from llama_index.core import (
    SimpleDirectoryReader
)

from RAG_module.config import DOCUMENT_DIR
from RAG_module.document_corrections import apply_ocr_corrections, _KNOWN_OCR_ERRORS

def load_documents():
  documents = SimpleDirectoryReader(DOCUMENT_DIR).load_data()
  documents = apply_ocr_corrections(documents)
  return documents