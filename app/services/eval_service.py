from dataclasses import dataclass
from PIL import Image
import torch


@dataclass
class EvalMetrics:
    prompt: str
    detection_count: int
    mean_score: float
    min_score: float
    max_score: float
    mean_mask_area_ratio: float
    score_distribution: dict[str, int]


def compute_image_metrics(
    raw_outputs: list[dict],
    image: Image.Image,
    prompt: str,
) -> EvalMetrics:
    """Compute per-image quality metrics from SAM3 segment output."""
    if not raw_outputs:
        return EvalMetrics(
            prompt=prompt,
            detection_count=0,
            mean_score=0.0,
            min_score=0.0,
            max_score=0.0,
            mean_mask_area_ratio=0.0,
            score_distribution={"low": 0, "mid": 0, "high": 0},
        )

    image_area = image.width * image.height

    first = raw_outputs[0]
    scores: list[float] = first["scores"] if first["scores"] else []
    masks = first["masks"]

    if not scores:
        return EvalMetrics(
            prompt=prompt,
            detection_count=0,
            mean_score=0.0,
            min_score=0.0,
            max_score=0.0,
            mean_mask_area_ratio=0.0,
            score_distribution={"low": 0, "mid": 0, "high": 0},
        )

    score_values = [float(s) for s in scores]
    detection_count = len(score_values)
    mean_score = sum(score_values) / detection_count
    min_score = min(score_values)
    max_score = max(score_values)

    score_distribution = {"low": 0, "mid": 0, "high": 0}
    for s in score_values:
        if s < 0.3:
            score_distribution["low"] += 1
        elif s < 0.6:
            score_distribution["mid"] += 1
        else:
            score_distribution["high"] += 1

    if isinstance(masks, torch.Tensor) and masks.numel() > 0:
        mask_areas = masks.float().sum(dim=(-2, -1))
        mean_mask_area_ratio = float((mask_areas / image_area).mean().item())
    else:
        mean_mask_area_ratio = 0.0

    return EvalMetrics(
        prompt=prompt,
        detection_count=detection_count,
        mean_score=round(mean_score, 4),
        min_score=round(min_score, 4),
        max_score=round(max_score, 4),
        mean_mask_area_ratio=round(mean_mask_area_ratio, 6),
        score_distribution=score_distribution,
    )
