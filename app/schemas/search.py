from typing import List, Any
from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    limit: int = 3
    filter: dict[str, Any] | None = None


class SearchResult(BaseModel):
    score: float
    text: str
    metadata: dict


class SearchResponse(BaseModel):
    results: List[SearchResult]