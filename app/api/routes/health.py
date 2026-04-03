from fastapi import APIRouter
from app.services.sam_runtime import sam_runtime

router = APIRouter()

@router.get("/health")
async def health():
    return {
        "status": "ok",
        "sam_loaded": sam_runtime.is_ready(),
    }