import json
from typing import List
import os
from src.redis.redis_client import RedisClient
from models.embedder_model import Embedder


if __name__ == "__main__":
    embedding_model = Embedder('distiluse-base-multilingual-cased-v1')
    redis = RedisClient(model=embedding_model)
    redis.client.flushdb()
    data = []
    json_folder_path = "json_collection"
    json_files = [f for f in os.listdir(json_folder_path) if f.endswith('.json')]

    for json_file in json_files:
        with open(os.path.join(json_folder_path, json_file), 'r', encoding='utf-8') as file:
            json_data = json.load(file)
            json_data['images'] = []
            data.append(json_data)

    redis.client.flushdb()
    redis.store_new_data(data=data)
    redis.create_vector_field()
    key, value = redis.create_embeddings()
    redis.store_embeddings(keys=key, embeddings=value)