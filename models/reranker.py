from sentence_transformers import CrossEncoder
import numpy as np


class Reranker:
    def __init__(self, model_name: str = ''):
        self.model = CrossEncoder(model_name, device='cuda')

    def rerank(self, query: list[str], documents: list[str]) -> np.ndarray:
        for doc in documents:
            predicted_results = self.model.predict()
