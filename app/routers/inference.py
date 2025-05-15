from fastapi import APIRouter
from typing import Annotated
from fastapi import Depends
from ..dependencies import run_inference
import os

router = APIRouter(
    prefix="/inference",
    tags=["inference"]
)

@router.post("/")
async def postInference(resultsUrl: Annotated[str, Depends(run_inference)]):
    print(os.getenv("ULTRALYTICS_API_KEY"))
    return {"resultsUrl":resultsUrl}