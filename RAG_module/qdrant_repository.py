from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    VectorParams,
    Distance,
    Filter,
    FieldCondition,
    MatchValue
)

# 建立一個 qdrant collection
def create_collection(client,
                      collection_name,
                      vector_size):
  if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

  client.create_collection(collection_name = collection_name,
                           vectors_config = VectorParams(size = vector_size,
                                                         distance = Distance.COSINE))


# 檢查 collection是否已經存在
def collection_exists(client, collection_name):
  return client.collection_exists(collection_name)


def create_point(idx,
                 text,
                 embedding,
                 metadata):
  payload = {"text": text}
  payload.update(metadata)
  return PointStruct(id = idx,
                     vector = embedding,
                     payload = payload)

# create a list of points from a list of nodes
def create_points(nodes, create_embedding, embedding_model):
  points = []

  for idx, node in enumerate(nodes):
    embedding = create_embedding(node.text, embedding_model)
    point = create_point(idx,
                         node.text,
                         embedding,
                         node.metadata)
    points.append(point)

  return points

# client 是程式與資料庫溝通的媒介
# 將一批 Points 寫入指定的 Collection
def upsert_points(client, collection_name, points):
  # 將資料送入資料庫
  client.upsert(collection_name = collection_name,
                points = points)

def search_points(client,
                  collection_name,
                  query_vector,
                  limit = 3,
                  filters = None):
  return client.query_points(collection_name = collection_name,
                             query = query_vector,
                             limit = limit,
                             query_filter = filters,
                             with_payload = True)

def build_metadata_filter(metadata_filters):
  if not metadata_filters:
    return None

  conditions = [FieldCondition(key = k,
                               match = MatchValue(value = v)) for k, v in metadata_filters.items()]
  return Filter(must = conditions)