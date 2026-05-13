from fastapi import APIRouter
from schemas.agent import AgentRequest, AgentResponse
from services.agent import AgentService

from routers.search import search_service

router = APIRouter(prefix="/agent", tags=["Agent"])

agent_service = AgentService(search_service=search_service)


@router.post("", response_model=AgentResponse)
async def agent(request: AgentRequest):
    return await agent_service.analyze(ticker=request.ticker, limit=request.limit)
