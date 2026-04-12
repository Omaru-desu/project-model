from pathlib import Path

SAM3_CHECKPOINT = Path("checkpoints/sam3.1/sam3.1_multiplex.pt")
DINO_MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"

DEVICE = "cuda"

MIN_SCORE = 0.45
MIN_BOX_AREA = 400
MAX_BOX_AREA_RATIO = 0.60
IOU_MERGE_THRESHOLD = 0.50
MIN_BLUR_SCORE = 30.0