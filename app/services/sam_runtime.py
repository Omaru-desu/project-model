import threading
from pathlib import Path

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from app.core.config import SAM3_CHECKPOINT, DEVICE


class SamRuntime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.model = None
        self.processor = None

    def load(self) -> None:
        with self._lock:
            if self.processor is not None:
                return

            checkpoint_path = str(Path(SAM3_CHECKPOINT).resolve())

            self.model = build_sam3_image_model(
                checkpoint_path=checkpoint_path,
                device=DEVICE,
                eval_mode=True,
                load_from_HF=False,
            )
            self.processor = Sam3Processor(self.model)

    def is_ready(self) -> bool:
        return self.processor is not None


sam_runtime = SamRuntime()