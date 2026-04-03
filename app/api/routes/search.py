from typing import Annotated

from app.services.segment_service import segment_uploaded_image
from app.services.embedding_service import embed_full_image, embed_crops_from_boxes

router = APIRouter()


@router.post("/image/search-ready")
async def segment_and_embed_image(
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
        raise HTTPException(status_code=500, detail=f"SAM failed: {str(exc)}") from exc

    boxes = output["boxes"]
    scores = output["scores"]

    boxes_list = boxes.tolist() if hasattr(boxes, "tolist") else boxes
    scores_list = scores.tolist() if hasattr(scores, "tolist") else scores

    try:
        frame_embedding = embed_full_image(file_bytes, embedding_type="cls")
        crop_embeddings = embed_crops_from_boxes(
            file_bytes=file_bytes,
            boxes=boxes_list,
            embedding_type="cls",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"DINOv3 failed: {str(exc)}") from exc

    # attach SAM score back onto crop results by original box order
    crop_results = []
    for item in crop_embeddings:
        idx = item["crop_index"]
        score = scores_list[idx] if idx < len(scores_list) else None
        crop_results.append({
            **item,
            "score": score,
        })

    return {
        "mask_count": len(output["masks"]),
        "box_count": len(boxes_list),
        "frame_embedding": frame_embedding,
        "crops": crop_results,
    }