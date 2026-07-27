import csv

# 主題分類（供 metadata filter 與 BM25 領域詞彙比對使用）
CATEGORIES = [
    "支付清算",
    "貨幣政策",
    "氣候金融",
    "央行比較法制",
    "其他",
]

# 文件性質
DOC_TYPES = [
    "本行政策報告",
    "國際準則中譯本",
    "比較法選輯",
    "制度說明手冊",
    "其他",
]

_METADATA_CSV_PATH = "RAG_module/document_metadata.csv"


def load_document_metadata(csv_path=_METADATA_CSV_PATH):
  """讀取文件層級 metadata，回傳以 filename 為 key 的 lookup dict。

  回傳格式：
  {
    "貨幣政策工具.pdf": {
        "title": "貨幣政策工具",
        "category": "貨幣政策",
        "publish_date": "",
        "issuing_unit": "中央銀行",
        "doc_type": "制度說明手冊",
    },
    ...
  }
  """
  metadata_lookup = {}
  with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
      filename = row.pop("filename")
      metadata_lookup[filename] = row
  return metadata_lookup


def get_metadata_for_file(filename, metadata_lookup):
  """取得單一檔案的 metadata；若找不到，回傳空值欄位並印出警告。

  這裡刻意不拋例外——文件蒐集階段可能會陸續加新檔案，
  找不到 metadata 時應該讓 pipeline 繼續跑，而不是整個中斷，
  但要讓使用者知道有檔案漏填 metadata。
  """
  if filename not in metadata_lookup:
    print(f"[metadata_loader] 警告：找不到 {filename} 的 metadata，"
          f"請確認 document_metadata.csv 是否有補上這筆資料。")
    return {
        "title": filename,
        "category": "其他",
        "publish_date": "",
        "issuing_unit": "",
        "doc_type": "其他",
    }
  return metadata_lookup[filename]