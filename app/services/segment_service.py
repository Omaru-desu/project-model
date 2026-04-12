import io
import torch
from PIL import Image

from app.core.config import DEVICE
from app.services.sam_runtime import sam_runtime


def to_python_or_cpu(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.tolist()
    return value


def to_cpu_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def load_image_from_bytes(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def segment_uploaded_image(file_bytes: bytes, prompt: str) -> list[dict]:
    image = load_image_from_bytes(file_bytes)
    return segment_image_with_multiple_prompts(image, [("single", prompt, prompt)])


def segment_image_with_multiple_prompts(
    image: Image.Image,
    prompts: list[tuple[str, str, str]],
) -> list[dict]:
    if sam_runtime.processor is None:
        raise RuntimeError("SAM 3 runtime is not loaded")

    results = []

    with torch.inference_mode():
        if DEVICE == "cuda" and torch.cuda.is_available():
            with torch.autocast("cuda", dtype=torch.float16):
                state = sam_runtime.processor.set_image(image)

                for label_id, display_label, prompt_term in prompts:
                    output = sam_runtime.processor.set_text_prompt(
                        state=state,
                        prompt=prompt_term,
                    )

                    results.append({
                        "label_id": label_id,
                        "display_label": display_label,
                        "prompt": prompt_term,
                        "masks": to_cpu_tensor(output["masks"]),
                        "boxes": to_python_or_cpu(output["boxes"]),
                        "scores": to_python_or_cpu(output["scores"]),
                    })
        else:
            state = sam_runtime.processor.set_image(image)

            for label_id, display_label, prompt_term in prompts:
                output = sam_runtime.processor.set_text_prompt(
                    state=state,
                    prompt=prompt_term,
                )

                results.append({
                    "label_id": label_id,
                    "display_label": display_label,
                    "prompt": prompt_term,
                    "masks": to_cpu_tensor(output["masks"]),
                    "boxes": to_python_or_cpu(output["boxes"]),
                    "scores": to_python_or_cpu(output["scores"]),
                })

    return results