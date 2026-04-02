
from fastapi import APIRouter

from schemas.rag import RAGRequest, RAGResponse
from services.rag import RAGService
from services.search import SearchService
from config.settings import settings

router = APIRouter(
    prefix="/rag",
    tags=["Rag"]
)

search_service = SearchService(
    qdrant_url=settings.qdrant_url,
    qdrant_api_key=settings.qdrant_api_key,
    collection_name=settings.collection_name
)

rag_service = RAGService(search_service=search_service)

@router.post("", response_model=RAGResponse)
async def rag(request: RAGRequest):
    return rag_service.generate_answer(query=request.query, limit=request.limit)