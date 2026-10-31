from __future__ import annotations

import re
import unicodedata

from wevisone_vl.prompts import TASKS, build_prompt

from vloc.errors import ApiError

LABEL_TASKS = {"detection", "point", "count_box", "count_point"}
PHRASE_TASKS = {"referring", "segmentation", "matting", "ocr_grounding"}
COUNT_TASKS = {"count_box", "count_point"}
PIXEL_TASKS = {"segmentation", "matting", "matting_box"}

MODES = {
    "detection": "box",
    "referring": "box",
    "count_box": "box",
    "ocr": "box",
    "ocr_grounding": "box",
    "visual_prompt": "box",
    "point": "point",
    "count_point": "point",
    "segmentation": "pixel",
    "matting": "pixel",
    "matting_box": "pixel",
    "freeform": "text",
}

MAX_LABELS = 20
MAX_LABEL_LEN = 64
MAX_PHRASE_LEN = 200
MAX_FREEFORM_LEN = 2000
MAX_REFERENCE_BOXES = 8

_WS_RE = re.compile(r"\s+")


def _bad(message: str) -> ApiError:
    return ApiError("invalid_request", message)


def sanitize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("<", "").replace(">", "")
    return _WS_RE.sub(" ", text).strip()


def normalize_task(task: str) -> str:
    t = task.strip().lower()
    if t not in TASKS:
        raise _bad(f"unknown task '{t}'; choose from {', '.join(TASKS)}")
    return t


def _is_empty(query: str | list[str] | None) -> bool:
    if query is None:
        return True
    if isinstance(query, str):
        return not query.strip()
    return not any(str(q).strip() for q in query)


def labels(query: str | list[str] | None) -> list[str]:
    items = query.split("\n") if isinstance(query, str) else list(query or [])
    out: list[str] = []
    for item in items:
        label = sanitize(item)
        if label and label not in out:
            out.append(label)
    if not 1 <= len(out) <= MAX_LABELS:
        raise _bad(f"query must contain 1-{MAX_LABELS} labels")
    for label in out:
        if len(label) > MAX_LABEL_LEN:
            raise _bad(f"each label must be <= {MAX_LABEL_LEN} chars")
    return out


def text(query: str | list[str] | None, max_len: int, name: str = "query") -> str:
    joined = " ".join(query) if isinstance(query, list) else (query or "")
    value = sanitize(joined)
    if not 1 <= len(value) <= max_len:
        raise _bad(f"{name} must be 1-{max_len} chars")
    return value


def scale_box(
    box: list[float], scale: float, size: tuple[int, int], name: str
) -> list[float]:
    if len(box) != 4:
        raise _bad(f"{name} must have 4 values [x1, y1, x2, y2]")
    w, h = size
    x1, y1, x2, y2 = (v * scale for v in box)
    x1, x2 = (min(max(v, 0.0), float(w)) for v in (x1, x2))
    y1, y2 = (min(max(v, 0.0), float(h)) for v in (y1, y2))
    if not (x2 > x1 and y2 > y1):
        raise _bad(f"{name} has no positive area inside the image")
    return [x1, y1, x2, y2]


def build(
    task: str,
    query: str | list[str] | None,
    reference_boxes: list[list[float]] | None,
    processed_size: tuple[int, int],
    scale: float,
) -> str:
    """Validate the request query for ``task`` and return the exact checkpoint prompt.

    ``task`` must already be normalized; ``reference_boxes`` are original-image pixels.
    """
    if reference_boxes is not None and task != "visual_prompt":
        raise _bad("reference_boxes is only valid for task visual_prompt")
    if task in LABEL_TASKS:
        items = labels(query)
        if task in COUNT_TASKS and len(items) != 1:
            raise _bad(f"{task} takes exactly 1 label")
        return build_prompt(task, items)
    if task in PHRASE_TASKS:
        return build_prompt(task, text(query, MAX_PHRASE_LEN))
    if task == "freeform":
        return build_prompt(task, text(query, MAX_FREEFORM_LEN))
    if task == "ocr":
        if not _is_empty(query):
            raise _bad("ocr takes no query")
        return build_prompt(task)
    # visual_prompt
    if not _is_empty(query):
        raise _bad("visual_prompt takes no query; pass reference_boxes")
    if not reference_boxes or len(reference_boxes) > MAX_REFERENCE_BOXES:
        raise _bad(f"visual_prompt requires 1-{MAX_REFERENCE_BOXES} reference_boxes")
    scaled = [
        scale_box(b, scale, processed_size, f"reference_boxes[{i}]") for i, b in enumerate(reference_boxes)
    ]
    return build_prompt(task, reference_boxes=scaled, image_size=processed_size)
