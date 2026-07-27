from rank_bm25 import BM25Okapi
import jieba
import RAG_module.api

_DOMAIN_TERMS = [
    # 支付清算類
    "跨行金融資訊系統",
    "票據交換結算系統",
    "信用卡結算系統",
    "外幣結算平台",
    "中央登錄債券系統",
    "債券等殖成交系統",
    "證券劃撥結算系統",
    "票保結算系統",
    "財金公司",
    "聯卡中心",
    "集保結算所",
    "櫃買中心",
    "即時總額清算",
    "定時淨額清算",
    "混合清算",
    "款對款同步收付",
    "款券同步交割",
    "系統性風險",
    "清算風險",
    "日間透支",
    # 貨幣政策工具類
    "準備金制度",
    "貼現窗口",
    "公開市場操作",
    "選擇性信用管理",
    "短期融通",
    "擔保放款之再融通",
    "金融機構轉存款",
    "選擇性信用融通",
    "選擇性信用管制",
    "附買回協定",
    "附賣回協定",
    "中央銀行定期存單",
    "中央銀行儲蓄券",
    "不動產信用管制",
    "消費者信用管制",
    # 氣候金融類
    "氣候變遷風險",
    "有形風險",
    "轉型風險",
    "總體審慎",
    "永續投資",
    "責任投資",
    "壓力測試",
    "淨零轉型",
    "綠色金融行動方案",
    "外匯存底管理",
    # 央行比較法制類
    "中央銀行法",
    "理事會",
    "貨幣政策委員會",
    "法定資本",
    "準備銀行法",
    # 國際組織/機構類
    "國際清算銀行",
    "金融穩定委員會",
    "巴塞爾銀行監理委員會",
    "綠色金融體系網絡",
]

for _term in _DOMAIN_TERMS:
  jieba.add_word(_term)


def _tokenize(text):
  return list(jieba.cut(text.lower()))


def _matches_filters(node, metadata_filters):
  return all(node.metadata.get(k) == v for k, v in metadata_filters.items())


class BM25Retriever:
  def __init__(self, nodes):
    self.nodes = nodes
    tokenized = [_tokenize(node.text) for node in nodes]
    #tokenized = [node.text.lower().split() for node in nodes]
    # 建一個能快速查關鍵字的索引
    self.bm25 = BM25Okapi(tokenized)

  def retrieve(self, query, limit = 3, metadata_filters = None):
    tokenized_query = _tokenize(query)
    # tokenized_query = query.lower().split()
    scores = self.bm25.get_scores(tokenized_query)

    if metadata_filters:
      for i, node in enumerate(self.nodes):
        if not _matches_filters(node, metadata_filters):
          scores[i] = -1

    top_indices = sorted(range(len(scores)),
                         key = lambda i: scores[i],
                         reverse = True)[:limit]

    return [(self.nodes[i], scores[i]) for i in top_indices]