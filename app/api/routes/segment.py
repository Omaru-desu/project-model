from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.gcs_service import download_image_from_gcs, GCSConfigurationError
from app.services.postprocess_service import build_review_candidates
from app.services.segment_service import segment_uploaded_image, segment_image_with_multiple_prompts
from app.services.visualize_service import draw_boxes_on_image
from app.services.vocab_service import build_prompt_list

router = APIRouter()


class FrameRequest(BaseModel):
    frame_id: str
    project_id: str
    upload_id: str
    frame_gcs_uri: str


class SegmentFramesRequest(BaseModel):
    frames: list[FrameRequest]
    label_ids: list[str] | None = None


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
        raw_outputs = segment_uploaded_image(file_bytes, prompt)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    first = raw_outputs[0]

    return {
        "mask_count": len(first["boxes"]),
        "boxes": first["boxes"],
        "scores": first["scores"],
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
        raw_outputs = segment_uploaded_image(file_bytes, prompt)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    first = raw_outputs[0]

    image_stream = draw_boxes_on_image(
        file_bytes=file_bytes,
        boxes=first["boxes"],
        scores=first["scores"],
    )

    return StreamingResponse(image_stream, media_type="image/png")


@router.post("/frames")
async def segment_frames(request: SegmentFramesRequest):
    try:
        prompts = build_prompt_list(label_ids=request.label_ids)
        results = []

        for frame in request.frames:
            image = download_image_from_gcs(frame.frame_gcs_uri)

            raw_outputs = segment_image_with_multiple_prompts(
                image=image,
                prompts=prompts,
            )

            detections = build_review_candidates(
                frame_id=frame.frame_id,
                image=image,
                raw_outputs=raw_outputs,
                frame_gcs_uri=frame.frame_gcs_uri,
            )

            results.append({
                "frame_id": frame.frame_id,
                "detections": detections,
            })

        return {"results": results}

    except GCSConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc