from sentence_transformers import SentenceTransformer
import numpy as np
from torch import cuda


class Embedder:
    def __init__(self, model_name: str):

        device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

        self.model = SentenceTransformer(model_name,
                                         device=device)

    def get_embedding(self, text) -> []:
        return self.model.encode(text).astype(np.float32).tolist()
