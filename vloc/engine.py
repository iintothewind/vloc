from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Protocol

import numpy as np
from PIL import Image

from vloc.config import Settings
from vloc.errors import ApiError

log = logging.getLogger("vloc")


@dataclass
class EngineOutput:
    response: str
    masks: list[np.ndarray]  # float32, processed-image size
    latency_s: float
    response_tokens: int


class Engine(Protocol):
    name: str

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int) -> EngineOutput: ...

    def decode_box(self, image: Image.Image, prompt: str, response: str, box_norm: list[float]) -> EngineOutput: ...


class WeVisOneEngine:
    """The only place that imports the model runtime (torch/transformers via wevisone_vl)."""

    def __init__(self, settings: Settings):
        from wevisone_vl import WeVisOneVL

        self.eng = WeVisOneVL(settings.model_path, min_pixels=settings.min_pixels, max_pixels=settings.max_pixels)
        self.name = Path(settings.model_path).name

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int) -> EngineOutput:
        r = self.eng.predict(image, prompt=prompt, max_new_tokens=max_new_tokens)
        tokens = len(self.eng.tokenizer(r.response)["input_ids"])
        return EngineOutput(r.response, list(r.masks), r.latency_s, tokens)

    def decode_box(self, image: Image.Image, prompt: str, response: str, box_norm: list[float]) -> EngineOutput:
        t0 = time.perf_counter()
        _, masks = self.eng.decode_masks(image, prompt, response, boxes=[box_norm])
        return EngineOutput(response, list(masks), time.perf_counter() - t0, 0)

    def warmup(self) -> None:
        from wevisone_vl.prompts import build_prompt

        img = Image.new("RGB", (1024, 768), (128, 128, 128))
        self.generate(img, build_prompt("detection", "object"), 64)
        self.decode_box(
            img,
            build_prompt("matting", "object"),
            "<ref>object</ref><box><100><100><900><900><OBJ></box>",
            [0.1, 0.1, 0.9, 0.9],
        )


class EngineRunner:
    """Single entry point for all GPU work: one job at a time, bounded wait queue, sync deadline."""

    def __init__(self, engine_factory: Callable[[], Engine], settings: Settings):
        self._factory = engine_factory
        self._settings = settings
        self.state: Literal["loading", "ready", "failed"] = "loading"
        self.error: str | None = None
        self.engine: Engine | None = None
        self._lock = asyncio.Lock()
        self._waiting = 0

    def start(self) -> None:
        threading.Thread(target=self._load, name="engine-load", daemon=True).start()

    def _load(self) -> None:
        try:
            engine = self._factory()
            warmup = getattr(engine, "warmup", None)
            if callable(warmup):
                t0 = time.perf_counter()
                warmup()
                log.info("engine warmup done in %.1fs", time.perf_counter() - t0)
            self.engine = engine
            self.state = "ready"
        except Exception as exc:
            log.exception("engine failed to load")
            self.error = str(exc) or type(exc).__name__
            self.state = "failed"

    async def run(self, fn: Callable[[Engine], EngineOutput]) -> tuple[EngineOutput, int]:
        entered = time.monotonic()
        deadline = entered + self._settings.sync_timeout_s
        if self.state != "ready" or self.engine is None:
            msg = "model is loading" if self.state == "loading" else f"model failed to load: {self.error}"
            raise ApiError("not_ready", msg)
        if self._waiting >= self._settings.max_queue:
            raise ApiError("busy", "server busy, retry later")
        self._waiting += 1
        try:
            try:
                await asyncio.wait_for(self._lock.acquire(), max(0.0, deadline - time.monotonic()))
            except TimeoutError:
                raise ApiError("timeout", f"request exceeded {self._settings.sync_timeout_s}s") from None
            queue_ms = round((time.monotonic() - entered) * 1000)
            engine = self.engine
            task = asyncio.create_task(asyncio.to_thread(fn, engine))
            # The lock stays held until the GPU work really finishes, even if this caller times out.
            task.add_done_callback(lambda _t: self._lock.release())
            try:
                out = await asyncio.wait_for(asyncio.shield(task), max(0.0, deadline - time.monotonic()))
            except TimeoutError:
                task.add_done_callback(_consume_exception)
                raise ApiError("timeout", f"request exceeded {self._settings.sync_timeout_s}s") from None
            return out, queue_ms
        finally:
            self._waiting -= 1


def _consume_exception(task: asyncio.Task) -> None:
    if not task.cancelled() and task.exception() is not None:
        log.error("engine job failed after its request timed out", exc_info=task.exception())
