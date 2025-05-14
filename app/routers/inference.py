from fastapi import APIRouter

router = APIRouter(
    prefix="/inference",
    tags=["inference"]
)

@router.post("/")
async def postInference():
    return {"message":"inference Started successfully"}