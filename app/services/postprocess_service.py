from uuid import uuid4
import numpy as np
import cv2
from PIL import Image
import torch

from app.core.config import (
    MIN_SCORE,
    MIN_BOX_AREA,
    MAX_BOX_AREA_RATIO,
    IOU_MERGE_THRESHOLD,
    MIN_BLUR_SCORE,
)
from app.services.gcs_service import (
    upload_pil_image_to_gcs,
    upload_mask_to_gcs,
    build_detection_artifact_gcs_uris,
)


def box_area(box: list[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(box1: list[float], box2: list[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0

    union = box_area(box1) + box_area(box2) - inter
    if union <= 0:
        return 0.0

    return inter / union


def iom(box1: list[float], box2: list[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0

    min_area = min(box_area(box1), box_area(box2))
    if min_area <= 0:
        return 0.0

    return inter / min_area


def blur_score_from_bbox(image: Image.Image, bbox: list[float]) -> float:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    crop = image.crop((x1, y1, x2, y2))
    crop_np = np.array(crop)

    if crop_np.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop_np, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def mask_to_cpu_numpy(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    mask = np.array(mask)

    # remove singleton dimensions like [1, H, W] or [1, 1, H, W]
    mask = np.squeeze(mask)

    # after squeeze, mask should be 2D
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask after squeeze, got shape {mask.shape}")

    return mask


def normalize_raw_outputs(frame_id: str, raw_outputs: list[dict]) -> list[dict]:
    normalized = []

    for group in raw_outputs:
        for idx, (box, score) in enumerate(zip(group["boxes"], group["scores"])):
            normalized.append({
                "frame_id": frame_id,
                "label_id": group["label_id"],
                "display_label": group["display_label"],
                "prompt": group["prompt"],
                "bbox": box,
                "score": float(score),
                "mask": mask_to_cpu_numpy(group["masks"][idx]),
            })

    return normalized


def apply_filters(image: Image.Image, detections: list[dict]) -> list[dict]:
    width, height = image.size
    image_area = width * height
    filtered = []

    for det in detections:
        area = box_area(det["bbox"])

        if det["score"] < MIN_SCORE:
            continue

        if area < MIN_BOX_AREA:
            continue

        if area / image_area > MAX_BOX_AREA_RATIO:
            continue

        blur = blur_score_from_bbox(image, det["bbox"])
        if blur < MIN_BLUR_SCORE:
            continue

        det["blur_score"] = blur
        filtered.append(det)

    return filtered


def merge_overlapping_detections(detections: list[dict]) -> list[dict]:
    detections = sorted(detections, key=lambda d: d["score"], reverse=True)
    clusters = []

    for det in detections:
        added_to_cluster = False
        for cluster in clusters:
            primary = cluster[0]
            if iou(det["bbox"], primary["bbox"]) >= IOU_MERGE_THRESHOLD or iom(det["bbox"], primary["bbox"]) >= 0.80:
                cluster.append(det)
                added_to_cluster = True
                break
        
        if not added_to_cluster:
            clusters.append([det])

    kept = []
    for cluster in clusters:
        seen_labels = set()
        cluster_kept = []
        for det in cluster:
            if det["label_id"] not in seen_labels:
                cluster_kept.append(det)
                seen_labels.add(det["label_id"])
            if len(cluster_kept) == 3:
                break
        
        kept.extend(cluster_kept)

    return kept


def save_detection_artifacts(
    image: Image.Image,
    detections: list[dict],
    frame_gcs_uri: str,
) -> list[dict]:
    saved = []

    for det in detections:
        detection_id = uuid4().hex

        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        crop = image.crop((x1, y1, x2, y2))

        crop_gcs_uri, mask_gcs_uri = build_detection_artifact_gcs_uris(
            frame_gcs_uri=frame_gcs_uri,
            detection_id=detection_id,
        )

        upload_pil_image_to_gcs(crop, crop_gcs_uri)
        upload_mask_to_gcs(det["mask"], mask_gcs_uri)

        saved.append({
            "detection_id": detection_id,
            "frame_id": det["frame_id"],
            "label_id": det["label_id"],
            "display_label": det["display_label"],
            "prompt": det["prompt"],
            "bbox": det["bbox"],
            "score": det["score"],
            "blur_score": det["blur_score"],
            "crop_gcs_uri": crop_gcs_uri,
            "mask_gcs_uri": mask_gcs_uri,
        })

    return saved


def build_review_candidates(
    frame_id: str,
    image: Image.Image,
    raw_outputs: list[dict],
    frame_gcs_uri: str,
) -> list[dict]:
    detections = normalize_raw_outputs(frame_id, raw_outputs)
    detections = apply_filters(image, detections)
    detections = merge_overlapping_detections(detections)
    detections = save_detection_artifacts(
        image=image,
        detections=detections,
        frame_gcs_uri=frame_gcs_uri,
    )
    return detections


def build_review_candidates_from_memory(
    frame_id: str,
    image: Image.Image,
    raw_outputs: list[dict],
) -> list[dict]:
    """In-memory version: returns crop and mask as PIL Image objects instead of uploading to GCS."""
    detections = normalize_raw_outputs(frame_id, raw_outputs)
    detections = apply_filters(image, detections)
    detections = merge_overlapping_detections(detections)

    result = []
    for det in detections:
        detection_id = uuid4().hex

        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        crop_image = image.crop((x1, y1, x2, y2))

        mask_np = np.array(det["mask"])
        mask_np = np.squeeze(mask_np)
        mask_np = (mask_np > 0).astype(np.uint8) * 255
        mask_image = Image.fromarray(mask_np, mode="L")

        result.append({
            "detection_id": detection_id,
            "frame_id": det["frame_id"],
            "label_id": det["label_id"],
            "display_label": det["display_label"],
            "prompt": det["prompt"],
            "bbox": det["bbox"],
            "score": det["score"],
            "blur_score": det["blur_score"],
            "crop_image": crop_image,
            "mask_image": mask_image,
        })

    return result
