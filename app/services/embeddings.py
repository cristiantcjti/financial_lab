from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
from qdrant_client import models
from config.settings import settings


class EmbeddingService:
    def __init__(self):
        self.dense_model = TextEmbedding(settings.dense_model)
        self.sparse_model = SparseTextEmbedding(settings.sparse_model)
        self.colbert_model = LateInteractionTextEmbedding(settings.colbert_model)

    def embed_query(self, query: str):
        dense = list(self.dense_model.passage_embed([query]))[0].tolist()
        sparse_raw = list(self.sparse_model.passage_embed([query]))[0]
        sparse = models.SparseVector(indices=sparse_raw.indices.tolist(), values=sparse_raw.values.tolist())
        colbert = list(self.colbert_model.passage_embed([query]))[0].tolist()
        return dense, sparse, colbert
    