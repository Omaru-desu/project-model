from io import BytesIO
from functools import lru_cache

import numpy as np
from PIL import Image
from google.cloud import storage
from google.auth.exceptions import DefaultCredentialsError
import torch


class GCSConfigurationError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_storage_client() -> storage.Client:
    try:
        return storage.Client()
    except DefaultCredentialsError as e:
        raise GCSConfigurationError(
            "Google Cloud credentials not configured. "
            "For local development, run `gcloud auth application-default login` "
            "or set GOOGLE_APPLICATION_CREDENTIALS to a service account JSON key. "
            "For Cloud Run or GCE, attach a service account to the runtime and do not "
            "set GOOGLE_APPLICATION_CREDENTIALS."
        ) from e


def parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")

    path = gcs_uri[5:]
    bucket_name, blob_name = path.split("/", 1)
    return bucket_name, blob_name


def download_image_from_gcs(gcs_uri: str) -> Image.Image:
    client = get_storage_client()
    bucket_name, blob_name = parse_gcs_uri(gcs_uri)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    data = blob.download_as_bytes()
    return Image.open(BytesIO(data)).convert("RGB")


def upload_pil_image_to_gcs(image: Image.Image, gcs_uri: str) -> str:
    client = get_storage_client()
    bucket_name, blob_name = parse_gcs_uri(gcs_uri)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    blob.upload_from_string(buffer.getvalue(), content_type="image/png")
    return gcs_uri


def upload_mask_to_gcs(mask_array, gcs_uri: str) -> str:
    if isinstance(mask_array, torch.Tensor):
        mask_array = mask_array.detach().cpu().numpy()

    mask = np.array(mask_array)
    mask = np.squeeze(mask)

    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D before saving, got shape {mask.shape}")

    mask = (mask > 0).astype(np.uint8) * 255

    image = Image.fromarray(mask, mode="L")
    return upload_pil_image_to_gcs(image, gcs_uri)


def build_detection_artifact_gcs_uris(
    frame_gcs_uri: str,
    detection_id: str,
) -> tuple[str, str]:
    bucket_name, blob_name = parse_gcs_uri(frame_gcs_uri)

    if "/frames/" not in blob_name:
        raise ValueError(f"frame_gcs_uri does not contain '/frames/': {frame_gcs_uri}")

    prefix = blob_name.split("/frames/")[0]

    crop_gcs_uri = f"gs://{bucket_name}/{prefix}/detections/{detection_id}/crop.png"
    mask_gcs_uri = f"gs://{bucket_name}/{prefix}/detections/{detection_id}/mask.png"

    return crop_gcs_uri, mask_gcs_uri