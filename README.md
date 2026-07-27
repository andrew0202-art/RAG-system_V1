# RAG System

一個自建的 Retrieval-Augmented Generation（檢索增強生成）系統，針對中文政策文件（支付清算、貨幣政策、氣候金融、央行比較法制等主題）進行問答與引用溯源。

## 專案結構

- `RAG_module/` — 系統原始碼（模組化設計）
  - `config.py`, `api.py` — 參數設定與外部 API 金鑰載入
  - `loader.py`, `document_corrections.py`, `metadata_loader.py` — 文件讀取與清理
  - `chunker.py` — 文件切塊
  - `embedding_service.py`, `vector_store.py`, `qdrant_repository.py` — 向量化與向量資料庫（Qdrant）
  - `bm25_retriever.py` — 關鍵字（稀疏）檢索
  - `retrieval_pipeline.py` — 整合向量檢索與關鍵字檢索
  - `reranker.py` — 檢索結果重排序
  - `prompt_builder.py`, `gemini_client.py` — 組裝 prompt 並呼叫 LLM
  - `citation_verifier.py`, `faithfulness_checker.py` — 引用正確性與答案忠實度檢查
  - `evaluator.py`, `experiment_tracker.py`, `recall_debugger.py`, `answer_quality_debugger.py` — 評估與調參工具
  - `main.py` — 互動式主程式
- `RAG System.ipynb` — 開發過程紀錄（Colab notebook，含建置流程、除錯與評估紀錄）

## 系統參數

- 外部文件：約 180 萬字，來自 7 份文件（745 頁）
- chunk_size = 550, chunk_overlap = 150
- embedding_model = `BAAI/bge-m3`（1024 維）
- top_k = 12, reranker_top_k = 8
- reranker_model = `cross-encoder/ms-marco-MiniLM-L-6-v2`
- llm_model = `llama-3.1-8b-instant`
- judge_model = `openai/gpt-oss-120b`

## 評估結果

以 Claude（Anthropic）協助擬定、並對照原始文件驗證的 40 題評估集為基準：

| 指標 | 數值 |
|---|---|
| Recall | 0.97 |
| Answer Accuracy | 0.92 |
| Citation Validity | 0.97 |
| Faithfulness | 0.82 |

Citation validity 指答案中所有引用標籤都確實對應到檢索結果中真實存在的段落，沒有捏造或指錯來源。Faithfulness 指以 LLM 作為裁判，判斷答案內容是否整體上都能被引用段落支持，而非模型自身先驗知識的產物。

主要待改進之處：當多個候選來源高度相似時，模型有時依賴表面線索（如段落位置、檔案分組）而非文字中明確指名的具體實體來消歧，導致引用錯誤。

## 安裝

```bash
pip install -r RAG_module/requirements.txt
```

需自行設定環境變數：`HF_TOKEN`、`GOOGLE_API_KEY`、`GROQ_API_KEY`。
