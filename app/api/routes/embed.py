from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.gcs_service import download_image_from_gcs, GCSConfigurationError
from app.services.embedding_service import embed_pil_images

router = APIRouter()


class DetectionEmbedRequest(BaseModel):
    detection_id: str
    crop_gcs_uri: str


class FrameEmbedRequest(BaseModel):
    frame_id: str
    frame_gcs_uri: str
    detections: list[DetectionEmbedRequest]


class EmbedFramesRequest(BaseModel):
    frames: list[FrameEmbedRequest]


@router.post("/frames")
async def embed_frames(request: EmbedFramesRequest):
    try:
        results = []

        for frame in request.frames:
            frame_image = download_image_from_gcs(frame.frame_gcs_uri)

            crop_images = [
                download_image_from_gcs(det.crop_gcs_uri)
                for det in frame.detections
            ]

            all_images = [frame_image] + crop_images
            all_embeddings = embed_pil_images(all_images)

            frame_embedding = all_embeddings[0]
            crop_embeddings = all_embeddings[1:]

            results.append({
                "frame_id": frame.frame_id,
                "frame_embedding": frame_embedding,
                "detections": [
                    {
                        "detection_id": det.detection_id,
                        "crop_embedding": crop_embeddings[i],
                    }
                    for i, det in enumerate(frame.detections)
                ],
            })

        return {"results": results}

    except GCSConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
