from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from wevisone_vl.prompts import TASKS, build_prompt

from vloc import __version__, queries
from vloc.artifacts import ArtifactStore, media_type_for
from vloc.config import Settings
from vloc.engine import Engine, EngineRunner
from vloc.errors import install_handlers
from vloc.schemas import MattingBoxRequest, PerceiveRequest, PerceiveResponse
from vloc.service import matting_box, perceive

log = logging.getLogger("vloc")

SWEEP_INTERVAL_S = 3600

_EXAMPLES: dict[str, Any] = {
    "detection": ["dog", "person"],
    "point": ["dog"],
    "count_box": "dog",
    "count_point": "dog",
    "referring": "the dog on the left",
    "segmentation": "the dog on the left",
    "matting": "the dog on the left",
    "ocr_grounding": "OPEN",
    "freeform": "Describe this image in one sentence.",
}


def _task_catalog() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for t in TASKS:
        if t in queries.LABEL_TASKS:
            n = 1 if t in queries.COUNT_TASKS else queries.MAX_LABELS
            kind, limits = "labels", {"min_labels": 1, "max_labels": n, "max_label_chars": queries.MAX_LABEL_LEN}
        elif t in queries.PHRASE_TASKS:
            kind, limits = "phrase", {"min_chars": 1, "max_chars": queries.MAX_PHRASE_LEN}
        elif t == "freeform":
            kind, limits = "text", {"min_chars": 1, "max_chars": queries.MAX_FREEFORM_LEN}
        elif t == "ocr":
            kind, limits = "none", {}
        else:
            kind, limits = "reference_boxes", {"min_boxes": 1, "max_boxes": queries.MAX_REFERENCE_BOXES}
        if t == "visual_prompt":
            example = build_prompt(t, reference_boxes=[[10, 10, 100, 100]], image_size=(1000, 1000))
        else:
            example = build_prompt(t, _EXAMPLES.get(t, ""))
        entries.append(
            {"task": t, "mode": queries.MODES[t], "query": kind, "limits": limits, "prompt_example": example}
        )
    entries.append(
        {
            "task": "matting_box",
            "mode": "pixel",
            "query": "phrase",
            "limits": {"min_chars": 1, "max_chars": queries.MAX_PHRASE_LEN, "box": "pixel xyxy"},
            "prompt_example": build_prompt("matting", "object"),
        }
    )
    return entries


def create_app(settings: Settings, engine_factory: Callable[[], Engine]) -> FastAPI:
    Image.MAX_IMAGE_PIXELS = settings.max_image_pixels
    catalog = _task_catalog()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = ArtifactStore(Path(settings.artifact_dir), settings.artifact_ttl_s, settings.public_base_url)
        runner = EngineRunner(engine_factory, settings)
        app.state.store = store
        app.state.runner = runner
        runner.start()

        async def sweeper() -> None:
            while True:
                try:
                    removed = await asyncio.to_thread(store.sweep)
                    if removed:
                        log.info("artifact sweep removed %d request dirs", removed)
                except Exception:
                    log.exception("artifact sweep failed")
                await asyncio.sleep(SWEEP_INTERVAL_S)

        sweep_task = asyncio.create_task(sweeper())
        try:
            yield
        finally:
            sweep_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sweep_task

    app = FastAPI(title="vloc", version=__version__, lifespan=lifespan)
    install_handlers(app)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/v1/health")
    async def health(request: Request) -> JSONResponse:
        runner: EngineRunner = request.app.state.runner
        if runner.state == "ready" and runner.engine is not None:
            return JSONResponse(
                {
                    "status": "ready",
                    "model": runner.engine.name,
                    "min_pixels": settings.min_pixels,
                    "max_pixels": settings.max_pixels,
                    "version": __version__,
                }
            )
        if runner.state == "failed":
            return JSONResponse({"status": "failed", "error": runner.error}, status_code=503)
        return JSONResponse({"status": "loading"}, status_code=503)

    @app.get("/v1/tasks")
    async def tasks() -> dict[str, Any]:
        return {"tasks": catalog}

    @app.post("/v1/perceive", response_model=PerceiveResponse)
    async def perceive_route(req: PerceiveRequest, request: Request) -> PerceiveResponse:
        return await perceive(req, request.app.state.runner, request.app.state.store, settings)

    @app.post("/v1/matting_box", response_model=PerceiveResponse)
    async def matting_box_route(req: MattingBoxRequest, request: Request) -> PerceiveResponse:
        return await matting_box(req, request.app.state.runner, request.app.state.store, settings)

    @app.get("/v1/artifacts/{request_id}/{name}")
    async def artifact(request_id: str, name: str, request: Request) -> FileResponse:
        path = request.app.state.store.path(request_id, name)
        return FileResponse(path, media_type=media_type_for(name))

    return app


def build_default_app() -> FastAPI:
    from vloc.engine import WeVisOneEngine

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    s = Settings.from_env()
    return create_app(s, lambda: WeVisOneEngine(s))
