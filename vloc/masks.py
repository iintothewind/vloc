from __future__ import annotations

import math
from io import BytesIO

import numpy as np
from PIL import Image


def to_png(mask: np.ndarray, task: str, size: tuple[int, int] | None = None) -> bytes:
    """Encode a probability map as an ``L`` PNG.

    ``segmentation`` is binarized at 0.5 (after any resize); matting tasks keep soft alpha.
    """
    m = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    if size is not None and (m.shape[1], m.shape[0]) != tuple(size):
        resized = Image.fromarray(m, mode="F").resize(tuple(size), Image.Resampling.BILINEAR)
        m = np.clip(np.asarray(resized, dtype=np.float32), 0.0, 1.0)
    if task == "segmentation":
        u8 = ((m >= 0.5) * 255).astype(np.uint8)
    else:
        u8 = np.round(m * 255.0).astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(u8, mode="L").save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def outside_box_fraction(mask: np.ndarray, box_norm: list[float]) -> float | None:
    h, w = mask.shape[:2]
    x1 = min(max(math.floor(box_norm[0] * w), 0), w)
    y1 = min(max(math.floor(box_norm[1] * h), 0), h)
    x2 = min(max(math.ceil(box_norm[2] * w), 0), w)
    y2 = min(max(math.ceil(box_norm[3] * h), 0), h)
    total = float(mask.sum(dtype=np.float64))
    if total <= 0:
        return None
    inside = float(mask[y1:y2, x1:x2].sum(dtype=np.float64))
    return round(1.0 - inside / total, 4)


def mask_mean(mask: np.ndarray) -> float:
    return round(float(mask.mean(dtype=np.float64)), 4)
