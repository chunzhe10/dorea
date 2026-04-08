"""YOLOv11n-seg inference for binary diver/water segmentation."""

from __future__ import annotations
from typing import List, Optional

import numpy as np


class YoloSegInference:
    """YOLOv11n-seg binary segmentation: diver (class_id=1) vs water (class_id=0).

    Uses ultralytics YOLOv11n-seg with COCO pretrained weights.
    COCO class 0 = "person" → mapped to class_id=1 (diver).
    All other pixels → class_id=0 (water).
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda", conf: float = 0.3):
        from ultralytics import YOLO
        self.device = device
        self.conf = conf
        # Load YOLOv11n-seg (nano variant for speed)
        if model_path is None:
            model_path = "yolo11n-seg.pt"  # auto-downloads from ultralytics hub
        self.model = YOLO(model_path)
        self.model.to(device)

    def infer(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Run segmentation on a single frame.

        Args:
            frame_rgb: (H, W, 3) uint8 RGB array

        Returns:
            class_mask: (H, W) uint8 array — 0=water, 1=diver
        """
        h, w = frame_rgb.shape[:2]
        # YOLO expects BGR
        import cv2
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        results = self.model(frame_bgr, conf=self.conf, verbose=False)

        mask = np.zeros((h, w), dtype=np.uint8)
        if results and results[0].masks is not None:
            for i, cls_id in enumerate(results[0].boxes.cls):
                if int(cls_id) == 0:  # COCO class 0 = "person"
                    seg_mask = results[0].masks.data[i].cpu().numpy()
                    # Resize mask to frame dimensions if needed
                    if seg_mask.shape != (h, w):
                        seg_mask = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask[seg_mask > 0.5] = 1
        return mask

    def infer_batch(self, frames_rgb: List[np.ndarray]) -> List[np.ndarray]:
        """Run segmentation on a batch of frames.

        Returns list of (H, W) uint8 class masks in same order as input.
        """
        return [self.infer(frame) for frame in frames_rgb]
