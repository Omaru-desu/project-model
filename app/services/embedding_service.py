import io
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.core.config import DEVICE, BATCH_SIZE
from app.services.dinov3_runtime import dinov3_runtime


def _extract_embedding(outputs, model, embedding_type: str = "cls"):
    if embedding_type == "pooler":
        if getattr(outputs, "pooler_output", None) is None:
            raise RuntimeError("DINOv3 output has no pooler_output")
        emb = outputs.pooler_output

    elif embedding_type == "cls":
        emb = outputs.last_hidden_state[:, 0, :]

    elif embedding_type == "mean_patches":
        nreg = getattr(model.config, "num_register_tokens", 0)
        patch_tokens = outputs.last_hidden_state[:, 1 + nreg :, :]
        emb = patch_tokens.mean(dim=1)

    else:
        raise ValueError("embedding_type must be one of: cls, pooler, mean_patches")

    emb = F.normalize(emb, p=2, dim=1)
    return emb


def _run_dinov3_on_pil(images: List[Image.Image], embedding_type: str = "cls") -> np.ndarray:
    if dinov3_runtime.processor is None or dinov3_runtime.model is None:
        raise RuntimeError("DINOv3 runtime is not loaded")

    inputs = dinov3_runtime.processor(images=images, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.inference_mode():
        if DEVICE == "cuda" and torch.cuda.is_available():
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = dinov3_runtime.model(**inputs)
        else:
            outputs = dinov3_runtime.model(**inputs)

    emb = _extract_embedding(outputs, dinov3_runtime.model, embedding_type)
    return emb.detach().cpu().numpy().astype(np.float32)


def _clamp_box(box, width: int, height: int):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), width))
    y1 = max(0, min(int(y1), height))
    x2 = max(0, min(int(x2), width))
    y2 = max(0, min(int(y2), height))
    return x1, y1, x2, y2


def embed_pil_images(images: List[Image.Image], embedding_type: str = "cls") -> List[List[float]]:
    """Run DINOv3 on a batch of PIL images and return normalised embeddings."""
    embs = _run_dinov3_on_pil(images, embedding_type=embedding_type)
    return [e.tolist() for e in embs]


def embed_batch(
    images: List[Image.Image],
    batch_size: int = BATCH_SIZE,
    embedding_type: str = "cls",
) -> List[List[float]]:
    """Process a flat list of PIL Images in chunks of batch_size through DINOv3."""
    results = []
    for i in range(0, len(images), batch_size):
        chunk = images[i : i + batch_size]
        embs = _run_dinov3_on_pil(chunk, embedding_type=embedding_type)
        results.extend(e.tolist() for e in embs)
    return results


def embed_full_image(file_bytes: bytes, embedding_type: str = "cls") -> List[float]:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    emb = _run_dinov3_on_pil([image], embedding_type=embedding_type)[0]
    return emb.tolist()


def embed_crops_from_boxes(
    file_bytes: bytes,
    boxes,
    embedding_type: str = "cls",
    min_crop_size: int = 16,
) -> List[Dict[str, Any]]:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    width, height = image.size

    crop_images = []
    crop_meta = []

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = _clamp_box(box, width, height)
        crop_w = x2 - x1
        crop_h = y2 - y1

        if crop_w < min_crop_size or crop_h < min_crop_size:
            continue

        crop = image.crop((x1, y1, x2, y2))
        crop_images.append(crop)
        crop_meta.append({
            "crop_index": idx,
            "bbox": [x1, y1, x2, y2],
            "width": crop_w,
            "height": crop_h,
        })

    if not crop_images:
        return []

    crop_embs = _run_dinov3_on_pil(crop_images, embedding_type=embedding_type)

    results = []
    for meta, emb in zip(crop_meta, crop_embs):
        results.append({
            **meta,
            "embedding": emb.tolist(),
        })

    return results
