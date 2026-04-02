from fastapi import APIRouter

router = APIRouter(
    prefix="/healthcheck",
    tags=["Healthcheck"],
)


@router.get("")
def get_healthcheck() -> dict[str, str]:
    return {"message": "Module is healthy!!"}
