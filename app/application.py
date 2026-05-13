from fastapi import APIRouter, FastAPI
from fastapi.responses import ORJSONResponse

from routers import healthcheck, search, rag, agent


class ApplicationBuilder:
    def __init__(self) -> None:
        self.__app = FastAPI(
            title="Financial Project API",
            default_response_class=ORJSONResponse,
            docs_url="/docs",
            redoc_url="/redocs",
        )

        self.__routes: list[APIRouter] = [
            healthcheck.router,
            search.router,
            rag.router,
            agent.router
        ]
        
        self.__configure_routes()

    def __configure_routes(self) -> None:
        for router in self.__routes:
            self.__app.include_router(router)

    def get_app(self) -> FastAPI:
        return self.__app