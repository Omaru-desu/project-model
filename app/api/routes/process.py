import asyncio
import base64
import json
from io import BytesIO

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from PIL import Image

from app.services.segment_service import segment_image_with_multiple_prompts
from app.services.embedding_service import embed_batch
from app.services.postprocess_service import build_review_candidates_from_memory
from app.services.vocab_service import build_prompt_list

router = APIRouter()

_gpu_semaphore = asyncio.Semaphore(1)


def _pil_to_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    buf = BytesIO()
    if fmt == "JPEG":
        image.convert("RGB").save(buf, format="JPEG", quality=90)
    else:
        image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _run_sam_sync(frame_id: str, image: Image.Image, prompts: list) -> list:
    raw_outputs = segment_image_with_multiple_prompts(image=image, prompts=prompts)
    return build_review_candidates_from_memory(
        frame_id=frame_id,
        image=image,
        raw_outputs=raw_outputs,
    )


@router.post("/frames")
async def process_frames(request: Request):
    try:
        form = await request.form()

        frames_metadata_raw = form.get("frames_metadata")
        if not frames_metadata_raw:
            raise HTTPException(status_code=400, detail="Missing frames_metadata field")
        frames_metadata = json.loads(frames_metadata_raw)

        label_ids_raw = form.get("label_ids")
        label_ids = json.loads(label_ids_raw) if label_ids_raw else None

        prompts = build_prompt_list(label_ids=label_ids)

        frame_inputs = []
        for frame_meta in frames_metadata:
            frame_id = frame_meta["frame_id"]

            file_field = form.get(frame_id)
            if file_field is None:
                raise HTTPException(status_code=400, detail=f"Missing frame file for frame_id: {frame_id}")

            frame_bytes = await file_field.read()
            image = Image.open(BytesIO(frame_bytes)).convert("RGB")
            frame_inputs.append((frame_id, image))

        async with _gpu_semaphore:
            frame_images = []
            frame_detections = []

            for frame_id, image in frame_inputs:
                detections = await run_in_threadpool(_run_sam_sync, frame_id, image, prompts)
                frame_images.append((frame_id, image))
                frame_detections.append(detections)

            all_images = []
            slices = []

            for (frame_id, image), detections in zip(frame_images, frame_detections):
                start = len(all_images)
                all_images.append(image)
                for det in detections:
                    all_images.append(det["crop_image"])
                slices.append((frame_id, start, len(all_images)))

            all_embeddings = await run_in_threadpool(embed_batch, all_images)

        results = []
        for (frame_id, start, end), detections in zip(slices, frame_detections):
            frame_embedding = all_embeddings[start]
            crop_embeddings = all_embeddings[start + 1 : end]

            detection_results = []
            for det, crop_emb in zip(detections, crop_embeddings):
                detection_results.append({
                    "detection_id": det["detection_id"],
                    "label_id": det["label_id"],
                    "display_label": det["display_label"],
                    "prompt": det["prompt"],
                    "bbox": det["bbox"],
                    "score": det["score"],
                    "blur_score": det["blur_score"],
                    "crop_embedding": crop_emb,
                    "crop_image": _pil_to_base64(det["crop_image"], "JPEG"),
                    "mask_image": _pil_to_base64(det["mask_image"], "PNG"),
                })

            results.append({
                "frame_id": frame_id,
                "frame_embedding": frame_embedding,
                "detections": detection_results,
            })

        return {"results": results}

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
