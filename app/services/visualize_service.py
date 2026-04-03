from io import BytesIO
from PIL import Image, ImageDraw


def draw_boxes_on_image(file_bytes: bytes, boxes, scores=None) -> BytesIO:
    image = Image.open(BytesIO(file_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]

        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)

        if scores is not None and i < len(scores):
            label = f"{scores[i]:.2f}"
            draw.text((x1, max(0, y1 - 20)), label, fill="red")

    output = BytesIO()
    image.save(output, format="PNG")
    output.seek(0)
    return output