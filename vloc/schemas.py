from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RenderView = Literal["overlay", "alpha", "green_screen", "cutout"]
Mode = Literal["box", "point", "pixel", "text"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ImageInput(_Strict):
    url: str | None = None
    b64: str | None = None  # raw base64 or data URL "data:image/...;base64,..."

    @model_validator(mode="after")
    def _exactly_one(self) -> ImageInput:
        if (self.url is None) == (self.b64 is None):
            raise ValueError("exactly one of image.url or image.b64 is required")
        return self


class RenderStyle(_Strict):
    show_boxes: bool = True
    show_labels: bool = True
    show_masks: bool = True
    mask_alpha: float = Field(0.45, ge=0, le=1)
    mask_threshold: float = Field(0.5, ge=0, le=1)
    font_size: int = Field(16, ge=8, le=64)
    line_width: int = Field(3, ge=1, le=16)


class Options(_Strict):
    max_new_tokens: int = Field(1024, ge=1, le=1024)
    render: bool = False
    render_views: list[RenderView] = Field(default_factory=lambda: ["overlay"])
    render_format: Literal["jpeg", "png"] = "jpeg"
    render_delivery: Literal["url", "base64"] = "url"
    render_style: RenderStyle = Field(default_factory=RenderStyle)
    mask_delivery: Literal["url", "base64", "none"] = "url"
    mask_resolution: Literal["processed", "original"] = "processed"


class PerceiveRequest(_Strict):
    image: ImageInput
    task: str
    query: str | list[str] | None = None
    reference_boxes: list[list[float]] | None = None  # pixel xyxy in the original (EXIF-corrected) image
    options: Options = Field(default_factory=Options)


class MattingBoxRequest(_Strict):
    image: ImageInput
    box: list[float] = Field(min_length=4, max_length=4)  # pixel xyxy in the original image
    phrase: str = "object"
    options: Options = Field(default_factory=Options)


class Artifact(BaseModel):
    name: str
    media_type: str
    width: int
    height: int
    url: str | None = None
    b64: str | None = None


class Instance(BaseModel):
    box_norm: list[float] | None = None
    box_px: list[float] | None = None
    point_norm: list[float] | None = None
    point_px: list[float] | None = None
    mask: Artifact | None = None
    mask_mean: float | None = None
    mask_outside_box_fraction: float | None = None


class Group(BaseModel):
    label: str
    absent: bool
    instances: list[Instance]


class PerceiveResponse(BaseModel):
    request_id: str
    task: str
    mode: Mode
    model: str
    prompt: str
    image_size: list[int]
    processed_size: list[int]
    scale: float
    groups: list[Group]
    declared_total: int | None
    count_by_items: int
    total_consistent: bool | None
    truncated: bool
    warnings: list[str]
    raw_response: str
    renders: list[Artifact]
    latency_ms: dict[str, int]


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
    code: ErrorCode
    message: str
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra context; for invalid_request from schema validation, `errors` holds the pydantic error list.",
    )


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: Literal["ready", "loading", "failed"]
    model: str | None = None
    min_pixels: int | None = None
    max_pixels: int | None = None
    version: str | None = None
    error: str | None = Field(None, description="Load failure message; only when status is failed.")
