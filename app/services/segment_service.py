import io
from PIL import Image
import torch

from app.services.sam_runtime import sam_runtime
from app.core.config import DEVICE


def segment_uploaded_image(file_bytes: bytes, prompt: str):
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    if sam_runtime.processor is None:
        raise RuntimeError("SAM 3 runtime is not loaded")

    if DEVICE == "cuda" and torch.cuda.is_available():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            state = sam_runtime.processor.set_image(image)
            output = sam_runtime.processor.set_text_prompt(
                state=state,
                prompt=prompt,
            )
    else:
        with torch.inference_mode():
            state = sam_runtime.processor.set_image(image)
            output = sam_runtime.processor.set_text_prompt(
                state=state,
                prompt=prompt,
            )

    return output