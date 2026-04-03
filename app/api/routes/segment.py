from fastapi import APIRouter

from typing import Annotated
from app.services.segment_service import segment_uploaded_image
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from app.services.visualize_service import draw_boxes_on_image
from fastapi.responses import StreamingResponse

router = APIRouter()

@router.post("/image")
async def segment_image(
    file: Annotated[UploadFile, File()],
    prompt: Annotated[str, Form()],
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected an image upload")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        output = segment_uploaded_image(file_bytes, prompt)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    masks = output["masks"]
    boxes = output["boxes"]
    scores = output["scores"]

    return {
        "mask_count": len(masks),
        "boxes": boxes.tolist() if hasattr(boxes, "tolist") else boxes,
        "scores": scores.tolist() if hasattr(scores, "tolist") else scores,
    }

@router.post("/image/preview")
async def segment_image_preview(
    file: Annotated[UploadFile, File()],
    prompt: Annotated[str, Form()],
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected an image upload")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        output = segment_uploaded_image(file_bytes, prompt)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    boxes = output["boxes"]
    scores = output["scores"]

    image_stream = draw_boxes_on_image(
        file_bytes=file_bytes,
        boxes=boxes.tolist() if hasattr(boxes, "tolist") else boxes,
        scores=scores.tolist() if hasattr(scores, "tolist") else scores,
    )

    return StreamingResponse(image_stream, media_type="image/png")