from pydantic import BaseModel
from typing import List


class MaskResult(BaseModel):
    area: int
    bbox: List[float]
    predicted_iou: float | None = None
    stability_score: float | None = None
    crop_box: List[float] | None = None


class SegmentImageResponse(BaseModel):
    mask_count: int
    masks: List[MaskResult]