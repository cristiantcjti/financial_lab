from typing import Any

from qdrant_client import QdrantClient, models

from schemas.search import SearchResponse, SearchResult
from services.embeddings import EmbeddingService


class SearchService:
    def __init__(self, qdrant_url: str, qdrant_api_key: str, collection_name: str):
        self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.collection_name = collection_name
        self.embedding_service = EmbeddingService()

    def _build_qdrant_filter(self, filters: dict[str, Any] | None) -> dict | None:
        if not filters:
            return

        must_conditions = []
        for key, value in filters.items():
            must_conditions.append(
                {"key": f"metadata.{key}", "match": {"value": value}}
            )
        return {"must": must_conditions}

    def search(self, query: str, limit: int = 3, filter: dict[str, Any] | None = None):
        query_dense, query_sparse, query_colbert = self.embedding_service.embed_query(
            query
        )

        query_filter = self._build_qdrant_filter(filters=filter)

        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    prefetch=[
                        models.Prefetch(query=query_dense, using="dense", limit=20),
                        models.Prefetch(query=query_sparse, using="sparse", limit=20),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=15,
                )
            ],
            query=query_colbert,
            using="colbert",
            limit=limit,
            query_filter=query_filter,
        )

        max_score = max(result.score for result in results.points)

        search_results = [
            SearchResult(
                score=result.score / max_score,
                text=result.payload["text"],
                metadata=result.payload.get("metadata", {}),
            )
            for result in results.points
        ]

        return SearchResponse(results=search_results)
