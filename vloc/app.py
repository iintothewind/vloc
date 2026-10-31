from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from pathlib import Path as FsPath
from typing import Annotated, Any, Callable

from fastapi import Body, FastAPI, Path, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from PIL import Image

from vloc import __version__, docs
from vloc.artifacts import ArtifactStore, media_type_for
from vloc.config import Settings
from vloc.engine import Engine, EngineRunner
from vloc.errors import install_handlers
from vloc.schemas import (
    HealthResponse,
    LivenessResponse,
    MattingBoxRequest,
    PerceiveRequest,
    PerceiveResponse,
    TasksResponse,
)
from vloc.service import matting_box, perceive

log = logging.getLogger("vloc")

SWEEP_INTERVAL_S = 3600


def create_app(settings: Settings, engine_factory: Callable[[], Engine]) -> FastAPI:
    Image.MAX_IMAGE_PIXELS = settings.max_image_pixels
    catalog = docs.task_catalog()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = ArtifactStore(FsPath(settings.artifact_dir), settings.artifact_ttl_s, settings.public_base_url)
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

    app = FastAPI(
        title="vloc perception API",
        version=__version__,
        summary="Detection, grounding, counting, OCR, segmentation and matting on one image, synchronously.",
        description=docs.api_description(settings),
        openapi_tags=docs.OPENAPI_TAGS,
        servers=[{"url": settings.public_base_url}] if settings.public_base_url else None,
        lifespan=lifespan,
    )
    install_handlers(app)

    @app.get(
        "/healthz",
        tags=["meta"],
        operation_id="liveness",
        summary="Liveness: is the HTTP process up",
        response_model=LivenessResponse,
    )
    async def healthz() -> dict[str, str]:
        """Always 200 while the process serves HTTP, even if the model is still loading or failed.
        Use for container liveness only; check `GET /v1/health` before sending work."""
        return {"status": "alive"}

    @app.get(
        "/skill.md",
        tags=["meta"],
        operation_id="getSkill",
        summary="Agent skill: compact usage guide in Markdown",
        response_class=PlainTextResponse,
        responses={200: {"content": {"text/markdown": {"schema": {"type": "string"}}}}},
    )
    async def skill(request: Request) -> PlainTextResponse:
        """Markdown skill for coding agents: when to use each task, code to send a local image, how to read
        results, recipes and known model quirks. The base URL is `VLOC_PUBLIC_BASE_URL` if set, else the URL
        this request reached. Save it as an agent skill (e.g. `skills/vloc/SKILL.md`)."""
        base = settings.public_base_url or str(request.base_url)
        return PlainTextResponse(docs.skill_markdown(settings, base), media_type="text/markdown; charset=utf-8")

    @app.get(
        "/v1/health",
        tags=["meta"],
        operation_id="getHealth",
        summary="Readiness: can the model accept requests",
        response_model=HealthResponse,
        responses={503: {"model": HealthResponse, "description": "Model is `loading` or `failed` (with `error`)."}},
    )
    async def health(request: Request) -> JSONResponse:
        """200 with `status: ready` once the model is loaded and warmed up. 503 with `status: loading` during
        startup (poll every few seconds) or `status: failed` with `error` if loading failed (needs a restart).
        Perception calls made before `ready` fail with 503 `not_ready`."""
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

    @app.get(
        "/v1/tasks",
        tags=["meta"],
        operation_id="listTasks",
        summary="List tasks with usage, limits and example requests",
        response_model=TasksResponse,
    )
    async def tasks() -> dict[str, Any]:
        """Machine-readable task guide: for each task, what it does, which input it needs (`query` kind and
        limits), where the result appears, the exact prompt the server builds, and a ready-to-send request body
        for `POST /v1/perceive` (the `matting_box` entry is for `POST /v1/matting_box`). Works while loading."""
        return {"tasks": catalog}

    @app.post(
        "/v1/perceive",
        tags=["perception"],
        operation_id="perceive",
        summary="Run a perception task on one image",
        response_model=PerceiveResponse,
        responses=docs.INFERENCE_RESPONSES,
    )
    async def perceive_route(
        req: Annotated[PerceiveRequest, Body(openapi_examples=docs.PERCEIVE_EXAMPLES)], request: Request
    ) -> PerceiveResponse:
        """Send an image plus a `task` (and the `query` that task needs); get back labeled boxes, points or
        masks in original-image pixels, optionally with rendered images.

        - Label tasks (`detection`, `point`, `count_box`, `count_point`): `query` is a list of labels.
        - Phrase tasks (`referring`, `segmentation`, `matting`, `ocr_grounding`): `query` is one phrase.
        - `freeform`: `query` is the instruction; the answer is `raw_response`.
        - `ocr`: no query. `visual_prompt`: no query, `reference_boxes` instead.

        Validation happens before the GPU is used, so 4xx errors are cheap. See the examples selector for one
        request per task and `GET /v1/tasks` for limits."""
        return await perceive(req, request.app.state.runner, request.app.state.store, settings)

    @app.post(
        "/v1/matting_box",
        tags=["perception"],
        operation_id="mattingBox",
        summary="Alpha matte for a known box",
        response_model=PerceiveResponse,
        responses=docs.INFERENCE_RESPONSES,
    )
    async def matting_box_route(
        req: Annotated[MattingBoxRequest, Body(openapi_examples=docs.MATTING_BOX_EXAMPLES)], request: Request
    ) -> PerceiveResponse:
        """Give a pixel box (for example a `box_px` from a detection call) and a short phrase; get one soft alpha
        matte for the object in that box. No text generation runs, so it is faster than `matting`. Response has
        `task: matting_box`, one group, one instance with `mask`; `raw_response` is the synthesized model input."""
        return await matting_box(req, request.app.state.runner, request.app.state.store, settings)

    @app.get(
        "/v1/artifacts/{request_id}/{name}",
        tags=["artifacts"],
        operation_id="getArtifact",
        summary="Download a rendered view or mask",
        response_class=FileResponse,
        responses=docs.ARTIFACT_RESPONSES,
    )
    async def artifact(
        request_id: Annotated[str, Path(description="`request_id` from the perception response (32 hex chars).")],
        name: Annotated[str, Path(description="Artifact file name, e.g. `overlay.jpg`, `cutout.png`, `mask_000.png`.")],
        request: Request,
    ) -> FileResponse:
        """Fetch a file listed as `url` in `renders[]` or `groups[].instances[].mask`. Prefer using that `url`
        verbatim. Files expire after the artifact TTL (404 `not_found`)."""
        path = request.app.state.store.path(request_id, name)
        return FileResponse(path, media_type=media_type_for(name))

    return app


def build_default_app() -> FastAPI:
    from vloc.engine import WeVisOneEngine

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    s = Settings.from_env()
    return create_app(s, lambda: WeVisOneEngine(s))
