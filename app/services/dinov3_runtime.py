from transformers import AutoImageProcessor, AutoModel
import torch
from app.core.config import DEVICE, DINO_MODEL_NAME

class DinoV3Runtime:
    def __init__(self):
        self.processor = None
        self.model = None

    def load(self):
        self.processor = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)
        self.model = AutoModel.from_pretrained(DINO_MODEL_NAME).to(DEVICE).eval()

dinov3_runtime = DinoV3Runtime()