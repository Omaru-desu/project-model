import io
from typing import Annotated

import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from app.services.eval_service import EvalMetrics, compute_image_metrics
from app.services.gcs_service import GCSConfigurationError, download_image_from_gcs
from app.services.segment_service import segment_image_with_multiple_prompts, segment_uploaded_image
from app.services.vocab_service import build_prompt_list

router = APIRouter()


class EvalFramesRequest(BaseModel):
    frame_gcs_uris: list[str]
    label_ids: list[str] | None = None


def _metrics_to_dict(metrics: EvalMetrics) -> dict:
    return {
        "prompt": metrics.prompt,
        "detection_count": metrics.detection_count,
        "mean_score": metrics.mean_score,
        "min_score": metrics.min_score,
        "max_score": metrics.max_score,
        "mean_mask_area_ratio": metrics.mean_mask_area_ratio,
        "score_distribution": metrics.score_distribution,
    }


@router.post("/image")
async def eval_image(
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
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        metrics = compute_image_metrics(raw_outputs, image, prompt=prompt)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _metrics_to_dict(metrics)


@router.post("/frames")
async def eval_frames(request: EvalFramesRequest):
    if not request.frame_gcs_uris:
        raise HTTPException(status_code=400, detail="No frames provided")

    try:
        prompts = build_prompt_list(label_ids=request.label_ids)
        frame_results = []
        all_scores = []
        all_detection_counts = []
        frames_with_no_detections = 0

        for gcs_uri in request.frame_gcs_uris:
            image = download_image_from_gcs(gcs_uri)
            raw_outputs = segment_image_with_multiple_prompts(image=image, prompts=prompts)

            combined_scores = [float(s) for o in raw_outputs for s in o["scores"]]
            combined_raw = [{
                "label_id": "combined",
                "display_label": "combined",
                "prompt": "combined",
                "masks": torch.cat([o["masks"] for o in raw_outputs], dim=0) if raw_outputs else torch.zeros(0),
                "boxes": [b for o in raw_outputs for b in o["boxes"]],
                "scores": combined_scores,
            }]

            prompt_label = ",".join(p[2] for p in prompts) if prompts else "all"
            metrics = compute_image_metrics(combined_raw, image, prompt=prompt_label)
            frame_result = _metrics_to_dict(metrics)
            frame_result["frame_gcs_uri"] = gcs_uri
            frame_results.append(frame_result)

            all_scores.extend(combined_scores)
            all_detection_counts.append(metrics.detection_count)
            if metrics.detection_count == 0:
                frames_with_no_detections += 1

        frame_count = len(frame_results)
        mean_detections = sum(all_detection_counts) / frame_count if frame_count else 0.0
        mean_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

        return {
            "frame_count": frame_count,
            "mean_detections_per_frame": round(mean_detections, 2),
            "mean_score_across_frames": round(mean_score, 4),
            "frames_with_no_detections": frames_with_no_detections,
            "frames": frame_results,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
