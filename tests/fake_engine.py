from __future__ import annotations

import threading

import numpy as np
from PIL import Image

from vloc.engine import EngineOutput
from vloc.parse import parse_response

COORDS = "<13><274><398><949>,<378><221><619><950>,<580><486><997><996>,<636><59><835><542>"
DETECTION = f"<ref>dogs</ref><box>{COORDS}</box>"
SEGMENTATION = "<ref>dogs</ref><box>" + ",".join(f"{c}<OBJ>" for c in COORDS.split(",")) + "</box>"
COUNT_BOX = "<ref>dog</ref><box><13><274><398><949>,<378><221><619><950></box>\nTotal number: 3."
ABSENT = "<ref>dogs</ref><box><13><274><398><949></box><ref>people</ref><box>none</box>"
MALFORMED = "<ref>chairs</ref><point>none</box>"

DEFAULT_RESPONSES = {
    "Locate all the instances": DETECTION,
    "Locate the instances that match the following description and provide the alpha mattes": SEGMENTATION,
    "How many": COUNT_BOX,
}

BAND_PX = 10


def synthetic_mask(size: tuple[int, int], box_norm: list[float]) -> np.ndarray:
    """1.0 inside the box, 0.25 in a 10-px band just outside it."""
    w, h = size
    x1, y1, x2, y2 = (round(v * s) for v, s in zip(box_norm, (w, h, w, h)))
    m = np.zeros((h, w), dtype=np.float32)
    m[max(0, y1 - BAND_PX) : y2 + BAND_PX, max(0, x1 - BAND_PX) : x2 + BAND_PX] = 0.25
    m[y1:y2, x1:x2] = 1.0
    return m


class FakeEngine:
    name = "fake-2B"

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        block: threading.Event | None = None,
        entered: threading.Event | None = None,
    ):
        self.responses = DEFAULT_RESPONSES if responses is None else responses
        self.block = block
        self.entered = entered
        self.prompts: list[str] = []

    def _wait(self) -> None:
        if self.entered is not None:
            self.entered.set()
        if self.block is not None:
            self.block.wait(30)

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int) -> EngineOutput:
        self._wait()
        self.prompts.append(prompt)
        response = next((r for p, r in self.responses.items() if prompt.startswith(p)), "")
        masks = [
            synthetic_mask(image.size, inst.norm)
            for g in parse_response(response).groups
            for inst in g.instances
            if inst.has_obj
        ]
        return EngineOutput(response, masks, 0.01, len(response) // 4)

    def decode_box(self, image: Image.Image, prompt: str, response: str, box_norm: list[float]) -> EngineOutput:
        self._wait()
        return EngineOutput(response, [synthetic_mask(image.size, box_norm)], 0.01, 0)
