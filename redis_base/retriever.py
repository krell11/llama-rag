import redis
import numpy as np
from redis.commands.search.field import (
    NumericField,
    TextField,
    VectorField,
)
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from models.embedder import Embedder
from typing import List


class RedisClient:
    def __init__(self, HOST='localhost', PORT=6379, PASSWORD=None,
                 text_model: Embedder = None):
        try:
            self.client = redis.Redis(host=HOST, port=PORT, password=PASSWORD,
                                      decode_responses=True)
            self.text_embedding_model = text_model
        except redis.RedisError:
            pass

    def store_new_data(self, data: List[dict]):
        pipeline = self.client.pipeline()
        document_id = self.client.get("document_counter")
        if document_id is None:
            document_id = 0
        else:
            document_id = int(document_id)

        for document in data:
            document_id += 1
            redis_key = f"document:{document_id}"
            pipeline.json().set(redis_key, "$", document)

        self.client.set("document_counter", document_id)

        pipeline.execute()
        pipeline.reset()

    def create_vector_field(self):
        try:
            print(self.client.ft("idx:docs").info())
        except:
            print("Creating new index")
            vector_dim = len(self.text_embedding_model.get_embedding('vec'))
            schema = (
                NumericField("$.page", as_name="page"),
                TextField("$.content", as_name="content"),
                TextField("$.images", as_name="images"),
                VectorField(
                    "$.embeddings",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": vector_dim,
                        "DISTANCE_METRIC": "COSINE",
                    },
                    as_name="chunk_vector",
                )
            )
            definition = IndexDefinition(prefix=["document:"], index_type=IndexType.JSON)
            self.client.ft("idx:docs").create_index(
                fields=schema, definition=definition
            )

    def create_embeddings(self):
        keys = sorted(self.client.keys("document:*"))
        content = self.client.json().mget(keys, "$.content")
        content = [item for sublist in content for item in sublist]
        text_embeddings = self.text_embedding_model.get_embedding(content)
        return keys, text_embeddings

    def store_embeddings(self, keys: List[str], text_embeddings: List[np.array]):
        pipeline = self.client.pipeline()
        for i, key in enumerate(keys):
            pipeline.json().set(key, "$.embeddings", text_embeddings)
        pipeline.execute()

    def search_query(self, k: int, user_query: str):
        query = (
            Query(f'(*)=>[KNN {k} @chunk_vector $query_vector AS vector_score]')
            .sort_by('vector_score')
            .return_fields('vector_score', 'content')
            .dialect(2)
        )
        encoded_query = self.text_embedding_model.get_embedding(user_query)
        result_list = self.client.ft("idx:docs").search(query, {
            'query_vector': np.array(encoded_query, dtype=np.float32).tobytes()}).docs
        res = []
        for index in result_list:
            res.append(index.id)
        return res

    def get_context_from_ids(self, document_ids: list):
        context = ""
        for doc_id in document_ids:
            document_text = self.client.json().get(doc_id)['content']
            context += document_text
        return context
