
from fastapi import APIRouter

from schemas.search import SearchRequest, SearchResponse
from services.search import SearchService
from config.settings import settings

search_service = SearchService(
    qdrant_url=settings.qdrant_url,
    qdrant_api_key=settings.qdrant_api_key,
    collection_name=settings.collection_name
)

router = APIRouter(
    prefix="/search",
    tags=["Search"]
)

@router.post("", response_model=SearchResponse)
async def seach(request: SearchRequest):
    return search_service.search(query=request.query, limit=request.limit)