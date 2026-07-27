import os
import json
import hashlib

CHECKSUM_PATH = "RAG_module/checksum.json"

def compute_checksum(document_dir, embedding_model):
  hasher = hashlib.md5()
  for filename in sorted(os.listdir(document_dir)):
    filepath = os.path.join(document_dir, filename)

    if not os.path.isfile(filepath):
      continue

    with open(filepath, "rb") as f:
      hasher.update(f.read())

  hasher.update(embedding_model.encode("utf-8"))

  return hasher.hexdigest()

def load_checksum():
  if not os.path.exists(CHECKSUM_PATH):
    return None

  with open(CHECKSUM_PATH, "r") as f:
    return json.load(f).get("checksum")

def save_checksum(checksum):
  with open(CHECKSUM_PATH, "w") as f:
    json.dump({"checksum": checksum}, f)

def is_kb_outdated(document_dir, embedding_model):
  current = compute_checksum(document_dir, embedding_model)
  saved = load_checksum()
  return current != saved, current