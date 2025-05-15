from fastapi import APIRouter
from typing import Annotated
from fastapi import Depends
from ..dependencies import save_to_supabase_storage
import os

router = APIRouter(
    prefix="/inference",
    tags=["inference"]
)

@router.post("/")
async def postInference(resultsUrl: Annotated[str, Depends(save_to_supabase_storage)]):

    return {"resultsUrl":resultsUrl}