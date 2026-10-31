from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from wevisone_vl.prompts import TASKS

RenderView = Literal["overlay", "alpha", "green_screen", "cutout"]
Mode = Literal["box", "point", "pixel", "text"]
QueryKind = Literal["labels", "phrase", "text", "none", "reference_boxes"]

BOX_PX = "Pixel box `[x1, y1, x2, y2]` (top-left origin, x right, y down) in the ORIGINAL image after EXIF rotation."


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ImageInput(_Strict):
    """The image to analyze. Set exactly one of `url` or `b64`."""

    url: str | None = Field(
        None,
        description="http(s) URL the server downloads (follows redirects). Rejected with 400 if URL fetch is disabled.",
        examples=["https://example.com/photo.jpg"],
    )
    b64: str | None = Field(
        None,
        description="Base64 image bytes, raw or as a data URL (`data:image/jpeg;base64,...`). "
        "Formats: JPEG, PNG, WEBP, BMP, GIF (first frame), MPO.",
    )

    @model_validator(mode="after")
    def _exactly_one(self) -> ImageInput:
        if (self.url is None) == (self.b64 is None):
            raise ValueError("exactly one of image.url or image.b64 is required")
        return self


class RenderStyle(_Strict):
    """Drawing style for rendered views. Sizes are scaled with the image (reference width 800 px)."""

    show_boxes: bool = Field(True, description="Draw box outlines (and a translucent fill for non-pixel tasks).")
    show_labels: bool = Field(True, description="Draw a label tag on each box/point.")
    show_masks: bool = Field(True, description="Tint masks in the overlay view (pixel tasks only).")
    mask_alpha: float = Field(0.45, ge=0, le=1, description="Opacity of the mask tint in the overlay view.")
    mask_threshold: float = Field(
        0.5, ge=0, le=1, description="Segmentation only: probability at which a pixel counts as foreground in views."
    )
    font_size: int = Field(16, ge=8, le=64, description="Label font size at 800 px reference width.")
    line_width: int = Field(3, ge=1, le=16, description="Box outline width at 800 px reference width.")


class Options(_Strict):
    """Generation, rendering and mask delivery options. All optional."""

    max_new_tokens: int = Field(
        1024,
        ge=1,
        le=1024,
        description="Generation budget. If hit, `truncated` is true and the tail of the answer may be missing.",
    )
    render: bool = Field(False, description="Produce annotated images listed in `renders`.")
    render_views: list[RenderView] = Field(
        default_factory=lambda: ["overlay"],
        description="Views to render when `render` is true. `overlay`: image with boxes/points/tags (and mask tint "
        "for pixel tasks). `alpha`: grayscale mask PNG. `green_screen`: foreground composited on green. `cutout`: "
        "RGBA PNG with transparent background. `alpha`, `green_screen`, `cutout` require task segmentation, "
        "matting or matting_box (else 422).",
    )
    render_format: Literal["jpeg", "png"] = Field(
        "jpeg", description="Encoding for `overlay` and `green_screen`. `alpha` and `cutout` are always PNG."
    )
    render_delivery: Literal["url", "base64"] = Field(
        "url", description="`url`: stored server-side, fetch via `renders[].url`. `base64`: inline in `renders[].b64`."
    )
    render_style: RenderStyle = Field(default_factory=RenderStyle)
    mask_delivery: Literal["url", "base64", "none"] = Field(
        "url",
        description="How per-instance mask PNGs are returned (pixel tasks). `none` skips them but still reports "
        "`mask_mean` and `mask_outside_box_fraction`.",
    )
    mask_resolution: Literal["processed", "original"] = Field(
        "processed",
        description="`processed`: masks at `processed_size` (cheaper). `original`: resized to `image_size`.",
    )


class PerceiveRequest(_Strict):
    """Run one perception task on one image."""

    image: ImageInput
    task: str = Field(
        description="Task name (case-insensitive). See `GET /v1/tasks` for what each task does and the query it needs.",
        json_schema_extra={"enum": list(TASKS)},
    )
    query: str | list[str] | None = Field(
        None,
        description="Depends on task. labels tasks (detection, point: 1-20 labels; count_box, count_point: exactly 1): "
        "a list of strings or one newline-separated string. phrase tasks (referring, segmentation, matting, "
        "ocr_grounding): one phrase, 1-200 chars. freeform: the full instruction, 1-2000 chars. ocr and "
        "visual_prompt: omit. `<` and `>` are stripped.",
    )
    reference_boxes: list[list[float]] | None = Field(
        None,
        description="visual_prompt only (required there, rejected elsewhere): 1-8 example boxes, each a " + BOX_PX,
    )
    options: Options = Field(default_factory=Options)


class MattingBoxRequest(_Strict):
    """Get a soft alpha matte for a box you already know; no detection step."""

    image: ImageInput
    box: list[float] = Field(min_length=4, max_length=4, description=BOX_PX)
    phrase: str = Field("object", description="Short description of the object inside the box, 1-200 chars.")
    options: Options = Field(default_factory=Options)


class Artifact(BaseModel):
    """A file produced by the request: a rendered view or an instance mask."""

    name: str = Field(description="File name, e.g. `overlay.jpg`, `cutout.png`, `mask_000.png`.")
    media_type: str = Field(description="`image/jpeg` or `image/png`.")
    width: int
    height: int
    url: str | None = Field(
        None,
        description="Set when delivery is `url`: `GET` it to download. Relative (`/v1/artifacts/...`) unless the "
        "server has VLOC_PUBLIC_BASE_URL; expires after the artifact TTL.",
    )
    b64: str | None = Field(None, description="Set when delivery is `base64`: the file bytes, base64-encoded.")


class Instance(BaseModel):
    """One detected object. Box tasks fill box_*, point tasks fill point_*."""

    box_norm: list[float] | None = Field(None, description="`[x1, y1, x2, y2]` as fractions 0..1 of width/height.")
    box_px: list[float] | None = Field(None, description=BOX_PX)
    point_norm: list[float] | None = Field(None, description="`[x, y]` as fractions 0..1 of width/height.")
    point_px: list[float] | None = Field(None, description="`[x, y]` pixels in the original image.")
    mask: Artifact | None = Field(
        None,
        description="Pixel tasks: mask PNG (grayscale). segmentation is binary 0/255; matting and matting_box are "
        "soft alpha 0..255.",
    )
    mask_mean: float | None = Field(None, description="Mean mask value 0..1 (fraction of the image covered).")
    mask_outside_box_fraction: float | None = Field(
        None, description="Share of mask mass outside its box (0..1); high values suggest a poor mask."
    )


class Group(BaseModel):
    """All instances for one label, in the order the model produced them."""

    label: str = Field(
        description="The requested label/phrase; for ocr the recognized text; for freeform the model's own phrase."
    )
    absent: bool = Field(description="True when the model answered that this label is not in the image.")
    instances: list[Instance]


class PerceiveResponse(BaseModel):
    request_id: str = Field(description="Unique id; also the artifact directory name.")
    task: str = Field(description="Normalized task name; `matting_box` for `/v1/matting_box`.")
    mode: Mode = Field(
        description="Output shape: `box` (box_*), `point` (point_*), `pixel` (boxes + masks), `text` (read "
        "raw_response; groups only hold phrases the model grounded inline)."
    )
    model: str = Field(description="Model checkpoint name.")
    prompt: str = Field(description="Exact instruction sent to the model.")
    image_size: list[int] = Field(description="`[width, height]` of the original image after EXIF rotation.")
    processed_size: list[int] = Field(description="`[width, height]` sent to the model (downscaled if too large).")
    scale: float = Field(description="processed width / original width.")
    groups: list[Group] = Field(description="One entry per label, in output order; absent labels included.")
    declared_total: int | None = Field(None, description="The `Total number: N` the model stated, if any.")
    count_by_items: int = Field(description="Total number of instances across groups.")
    total_consistent: bool | None = Field(
        None, description="count tasks only: whether declared_total equals count_by_items; null otherwise."
    )
    truncated: bool = Field(description="Generation stopped at max_new_tokens; results may be incomplete.")
    warnings: list[str] = Field(description="Non-fatal issues: absent labels, malformed or dropped model output.")
    raw_response: str = Field(
        description="Raw model text with `<ref>label</ref><box><x1><y1><x2><y2></box>` markup, coordinates on a "
        "0-1000 grid. For freeform this is the answer."
    )
    renders: list[Artifact] = Field(description="Rendered views; empty unless options.render.")
    latency_ms: dict[str, int] = Field(
        description="`queue` (waiting for the GPU), `model`, `render`, `total` in milliseconds."
    )


ErrorCode = Literal[
    "invalid_request",
    "image_too_large",
    "unsupported_image",
    "image_fetch_failed",
    "not_found",
    "not_ready",
    "busy",
    "timeout",
    "internal",
]


class ErrorDetail(BaseModel):
    code: ErrorCode = Field(description="Stable machine-readable error code; branch on this, not on message.")
    message: str = Field(description="Human-readable explanation.")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra context; for invalid_request from schema validation, `errors` holds the pydantic error list.",
    )


class ErrorResponse(BaseModel):
    """Body of every non-2xx response except /v1/health."""

    error: ErrorDetail


class HealthResponse(BaseModel):
    status: Literal["ready", "loading", "failed"] = Field(
        description="`ready`: accepts inference. `loading`: model still loading, retry later. `failed`: load "
        "error, will not recover without a restart."
    )
    model: str | None = None
    min_pixels: int | None = Field(None, description="Images below this pixel count are upscaled by the engine.")
    max_pixels: int | None = Field(None, description="Images above this pixel count are downscaled before inference.")
    version: str | None = None
    error: str | None = Field(None, description="Load failure message; only when status is failed.")


class LivenessResponse(BaseModel):
    status: Literal["alive"]


class TaskInfo(BaseModel):
    task: str
    mode: Mode
    description: str = Field(description="What the task does and when to use it.")
    output: str = Field(description="Where the result appears in PerceiveResponse.")
    query: QueryKind = Field(description="What the request must carry: `query` (labels/phrase/text), "
                             "`reference_boxes`, or nothing (`none`).")
    limits: dict[str, Any]
    prompt_example: str = Field(description="Prompt the server builds for `request_example`.")
    request_example: dict[str, Any] = Field(description="Ready-to-send body (replace the image URL).")


class TasksResponse(BaseModel):
    tasks: list[TaskInfo]
