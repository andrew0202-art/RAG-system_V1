
_KNOWN_OCR_ERRORS = {
    "貨幣政策工具.pdf": [
        ("店 、1,000", "500萬元、1,000"),
        ("國庫六 發 行條例", "國庫券發行條例"),
    ],
}


def apply_ocr_corrections(documents):
  # 先統計每個檔案、每筆修正，總共命中幾次
  hit_counts = {
      filename: {wrong: 0 for wrong, _ in corrections}
      for filename, corrections in _KNOWN_OCR_ERRORS.items()
  }

  for doc in documents:
    filename = doc.metadata.get("file_name")
    corrections = _KNOWN_OCR_ERRORS.get(filename)
    if not corrections:
      continue

    text = doc.get_content()
    changed = False
    for wrong, correct in corrections:
      if wrong in text:
        text = text.replace(wrong, correct)
        changed = True
        hit_counts[filename][wrong] += 1
        print(f"[ocr_correction] {filename}（page_label = "
              f"{doc.metadata.get('page_label')}）：「{wrong}」→「{correct}」")

    if changed:
      doc.set_content(text)


  # 檢查有沒有任何一筆修正，整份文件都完全沒命中
  made_corrections = True
  for filename, corrections in _KNOWN_OCR_ERRORS.items():
    for wrong, correct in corrections:
      if hit_counts.get(filename, {}).get(wrong, 0) == 0:
        made_corrections = False

  if len(_KNOWN_OCR_ERRORS) == 0:
    print(f"no corrections on the documents are needed.")

  else:
    if made_corrections:
      print(f"some known errors are corrected for the documents.")

    else:
      print(f"corrections are needed but none has been made.")

  return documents