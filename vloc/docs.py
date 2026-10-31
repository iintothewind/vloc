"""OpenAPI text, task guide, examples and the agent skill. Served via /openapi.json, /docs, /v1/tasks, /skill.md."""

from __future__ import annotations

import functools
from importlib.resources import files
from typing import Any

from fastapi.openapi.models import Example
from wevisone_vl.prompts import TASKS, build_prompt

from vloc import queries
from vloc.config import Settings
from vloc.schemas import ErrorResponse

EXAMPLE_URL = "https://example.com/image.jpg"


@functools.cache
def _skill_template() -> str:
    return files("vloc").joinpath("skill.md").read_text(encoding="utf-8")


def skill_markdown(s: Settings, base_url: str) -> str:
    """Agent skill with this deployment's base URL and limits filled in."""
    values = {
        "BASE_URL": base_url.rstrip("/"),
        "ARTIFACT_TTL_H": f"{s.artifact_ttl_s / 3600:g}",
        "MAX_PIXELS": f"{s.max_pixels:,}",
        "MAX_IMAGE_BYTES": f"{s.max_image_bytes:,}",
        "MAX_IMAGE_PIXELS": f"{s.max_image_pixels:,}",
        "URL_FETCH_NOTE": "" if s.allow_url_fetch else " `image.url` is disabled on this server; always send `b64`.",
    }
    text = _skill_template()
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    return text


# task -> (description, output, example query or reference_boxes)
TASK_GUIDE: dict[str, tuple[str, str, Any]] = {
    "detection": (
        "Find every instance of one or more object categories ('where are all the dogs and people?'). "
        "Labels not present in the image come back with absent=true.",
        "groups[] one per label in request order; instances[].box_px.",
        ["dog", "person"],
    ),
    "referring": (
        "Locate the object(s) matching a natural-language description that uses attributes or relations "
        "('the man in the red shirt', 'boys holding microphone').",
        "One group labeled with the phrase; instances[].box_px (can be several).",
        "boys holding microphone",
    ),
    "point": (
        "Mark one point on each instance of one or more categories. Lighter than boxes; use for targeting/clicking.",
        "groups[] per label; instances[].point_px.",
        ["dog"],
    ),
    "count_box": (
        "Count instances of exactly one category, with a box per counted item.",
        "declared_total (count the model stated), count_by_items (boxes returned), total_consistent; "
        "instances[].box_px.",
        "white chess pieces",
    ),
    "count_point": (
        "Count instances of exactly one category, with a point per counted item.",
        "declared_total, count_by_items, total_consistent; instances[].point_px.",
        "white chess pieces",
    ),
    "segmentation": (
        "Binary pixel masks for the object(s) matching a phrase.",
        "instances[].box_px + instances[].mask (binary 0/255 PNG). Views alpha/green_screen/cutout available.",
        "the dog on the left",
    ),
    "matting": (
        "Soft alpha mattes (hair, fur, semi-transparent edges) for the object(s) matching a phrase. "
        "Use for background removal.",
        "instances[].box_px + instances[].mask (soft alpha PNG). Views alpha/green_screen/cutout available.",
        "person",
    ),
    "ocr": (
        "Read all words in the image with their locations. Takes no query.",
        "One group per distinct word: label = recognized text, instances[].box_px; repeated words share a group.",
        None,
    ),
    "ocr_grounding": (
        "Find where a given text string appears in the image.",
        "One group labeled with the text; instances[].box_px.",
        "COFFEE",
    ),
    "visual_prompt": (
        "Find all objects similar to 1-8 example boxes you supply (search by example, no label needed).",
        "One group whose label the model names (e.g. 'object 1: Dog'); instances[].box_px, usually including the "
        "examples themselves.",
        [[13, 190, 406, 660]],
    ),
    "freeform": (
        "Any instruction: captions, visual question answering, detailed descriptions. The model may ground phrases "
        "inline.",
        "Answer text in raw_response; groups[] holds phrases the model grounded with boxes or points.",
        "Describe this image in one sentence.",
    ),
}

MATTING_BOX_GUIDE = (
    "Soft alpha matte for a box you already have (from detection or a user). Skips text generation, so it is the "
    "fastest pixel output. Call POST /v1/matting_box, not /v1/perceive.",
    "One group with one instance: box_px + mask (soft alpha PNG).",
)


def _request_example(task: str) -> dict[str, Any]:
    body: dict[str, Any] = {"image": {"url": EXAMPLE_URL}, "task": task}
    sample = TASK_GUIDE[task][2]
    if task == "visual_prompt":
        body["reference_boxes"] = sample
    elif sample is not None:
        body["query"] = sample
    if task in ("segmentation", "matting"):
        body["options"] = {"render": True, "render_views": ["overlay", "cutout"]}
    return body


MATTING_BOX_EXAMPLE: dict[str, Any] = {
    "image": {"url": EXAMPLE_URL},
    "box": [13.3, 190.7, 406.5, 660.5],
    "phrase": "dog",
    "options": {"render": True, "render_views": ["cutout"]},
}


def task_catalog() -> list[dict[str, Any]]:
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
        description, output, sample = TASK_GUIDE[t]
        if t == "visual_prompt":
            prompt = build_prompt(t, reference_boxes=sample, image_size=(1024, 696))
        else:
            prompt = build_prompt(t, sample or "")
        entries.append(
            {
                "task": t,
                "mode": queries.MODES[t],
                "description": description,
                "output": output,
                "query": kind,
                "limits": limits,
                "prompt_example": prompt,
                "request_example": _request_example(t),
            }
        )
    entries.append(
        {
            "task": "matting_box",
            "mode": "pixel",
            "description": MATTING_BOX_GUIDE[0],
            "output": MATTING_BOX_GUIDE[1],
            "query": "phrase",
            "limits": {"min_chars": 1, "max_chars": queries.MAX_PHRASE_LEN, "box": "pixel [x1, y1, x2, y2]"},
            "prompt_example": build_prompt("matting", "dog"),
            "request_example": MATTING_BOX_EXAMPLE,
        }
    )
    return entries


PERCEIVE_EXAMPLES: dict[str, Example] = {
    t: Example(summary=f"{t}: {TASK_GUIDE[t][0].split('.')[0]}", value=_request_example(t)) for t in TASKS
}
PERCEIVE_EXAMPLES["detection_render_base64"] = Example(
    summary="detection with an inline base64 overlay image",
    value={
        "image": {"b64": "<base64 JPEG/PNG bytes or data:image/jpeg;base64,...>"},
        "task": "detection",
        "query": "dog\nperson",
        "options": {"render": True, "render_delivery": "base64"},
    },
)

MATTING_BOX_EXAMPLES: dict[str, Example] = {
    "matting_box": Example(summary="Alpha matte for a known box", value=MATTING_BOX_EXAMPLE)
}

PERCEIVE_RESPONSE_EXAMPLE: dict[str, Any] = {
    "request_id": "4f3c2b1a0e9d8c7b6a5f4e3d2c1b0a99",
    "task": "detection",
    "mode": "box",
    "model": "WeVisOne-VL-2B",
    "prompt": "Locate all the instances that match the following description: dog. person.",
    "image_size": [1024, 696],
    "processed_size": [1024, 696],
    "scale": 1.0,
    "groups": [
        {
            "label": "dog",
            "absent": False,
            "instances": [
                {"box_norm": [0.013, 0.274, 0.397, 0.949], "box_px": [13.3, 190.7, 406.5, 660.5],
                 "point_norm": None, "point_px": None, "mask": None, "mask_mean": None,
                 "mask_outside_box_fraction": None},
                {"box_norm": [0.377, 0.222, 0.619, 0.953], "box_px": [386.0, 154.5, 633.9, 663.3],
                 "point_norm": None, "point_px": None, "mask": None, "mask_mean": None,
                 "mask_outside_box_fraction": None},
            ],
        },
        {"label": "person", "absent": True, "instances": []},
    ],
    "declared_total": None,
    "count_by_items": 2,
    "total_consistent": None,
    "truncated": False,
    "warnings": ["label 'person' absent"],
    "raw_response": "<ref>dog</ref><box><13><274><397><949>,<377><222><619><953></box>"
    "<ref>person</ref><box>none</box>",
    "renders": [
        {"name": "overlay.jpg", "media_type": "image/jpeg", "width": 1024, "height": 696,
         "url": "/v1/artifacts/4f3c2b1a0e9d8c7b6a5f4e3d2c1b0a99/overlay.jpg", "b64": None}
    ],
    "latency_ms": {"queue": 0, "model": 868, "render": 6, "total": 889},
}


def _error(description: str, *examples: tuple[str, str]) -> dict[str, Any]:
    """Error response doc with one example body per (code, message)."""
    bodies = {code: {"error": {"code": code, "message": message, "details": {}}} for code, message in examples}
    media = (
        {"example": next(iter(bodies.values()))}
        if len(bodies) == 1
        else {"examples": {code: {"summary": code, "value": body} for code, body in bodies.items()}}
    )
    return {"model": ErrorResponse, "description": description, "content": {"application/json": media}}


_INTERNAL = _error(
    "`internal`: unexpected server error (traceback is logged). Not retryable as-is.",
    ("internal", "internal error"),
)
_INVALID = _error(
    "`invalid_request`: schema validation (`details.errors` lists pydantic errors) or semantic validation "
    "(unknown task, query limits, reference_boxes, render view vs task, image shorter side < 8 px or aspect > 200). "
    "Fix the request; do not retry unchanged.",
    ("invalid_request", "query must contain 1-20 labels"),
)

INFERENCE_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {"content": {"application/json": {"example": PERCEIVE_RESPONSE_EXAMPLE}}},
    400: _error(
        "`image_fetch_failed`: image.url fetch error, non-2xx status, non-http(s) scheme, or URL fetch "
        "disabled. Send the image as `b64` instead.",
        ("image_fetch_failed", "image.url returned HTTP 404"),
    ),
    413: _error(
        "`image_too_large`: image exceeds the byte or pixel limit. Downscale/recompress and retry.",
        ("image_too_large", "image exceeds 20000000 bytes"),
    ),
    415: _error(
        "`unsupported_image`: not decodable or not JPEG/PNG/WEBP/BMP/GIF/MPO. Convert to JPEG or PNG.",
        ("unsupported_image", "cannot identify image file"),
    ),
    422: _INVALID,
    500: _INTERNAL,
    503: {
        **_error(
            "`not_ready`: model still loading (poll GET /v1/health) or failed to load. `busy`: wait queue full; "
            "retry after `Retry-After` seconds.",
            ("not_ready", "model is loading"),
            ("busy", "server busy, retry later"),
        ),
        "headers": {
            "Retry-After": {"description": "Seconds to wait; sent only with `busy`.", "schema": {"type": "integer"}}
        },
    },
    504: _error(
        "`timeout`: request exceeded the sync timeout (the GPU job still finishes in the background). "
        "Retry later, ideally with a smaller image or lower max_new_tokens.",
        ("timeout", "request exceeded 60s"),
    ),
}

ARTIFACT_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "Rendered view or mask file.",
        "content": {
            "image/png": {"schema": {"type": "string", "format": "binary"}},
            "image/jpeg": {"schema": {"type": "string", "format": "binary"}},
        },
    },
    404: _error(
        "`not_found`: malformed id/name, never written, or expired. Re-run the request to regenerate.",
        ("not_found", "artifact not found"),
    ),
    422: _INVALID,
    500: _INTERNAL,
}

OPENAPI_TAGS = [
    {"name": "perception", "description": "Run the vision-language model on an image."},
    {"name": "artifacts", "description": "Download rendered views and masks produced by perception calls."},
    {"name": "meta", "description": "Health, readiness and task discovery."},
]


def _task_table() -> str:
    rows = ["| task | use it to | query | result |", "|---|---|---|---|"]
    for entry in task_catalog():
        rows.append(
            f"| `{entry['task']}` | {entry['description']} | {entry['query']} | {entry['output']} |"
        )
    return "\n".join(rows)


def api_description(s: Settings) -> str:
    fetch = "enabled" if s.allow_url_fetch else "**disabled** (send `b64`)"
    return f"""
Synchronous visual perception over HTTP: object detection, referring expressions, pointing, counting, OCR,
segmentation, alpha matting, search-by-example and free-form visual Q&A on a single image, powered by the
`{s.model_path.rstrip('/').split('/')[-1]}` vision-language model on one GPU.

**Agents:** `GET /skill.md` returns a compact usage skill (when to use which task, code to send a local file,
recipes, model quirks) with this server's URL filled in.

## How to call it

1. `GET /v1/health` until `status` is `ready` (HTTP 503 while the model loads, typically under a minute).
2. Pick a task (table below, or `GET /v1/tasks` for machine-readable limits and ready-to-send bodies).
3. `POST /v1/perceive` with `{{"image": {{"url" | "b64": ...}}, "task": ..., "query": ...}}`.
   For a matte of a box you already know, use `POST /v1/matting_box` instead.
4. Read `groups[]`: one entry per label with `instances[]` holding `box_px` / `point_px` in original-image pixels.
5. Optional: set `options.render=true` to get annotated images in `renders[]`; download each `url` with `GET`.

Minimal request:

```json
{{"image": {{"url": "https://example.com/dogs.jpg"}}, "task": "detection", "query": ["dog", "person"]}}
```

## Tasks

{_task_table()}

## Coordinates

- Input boxes (`reference_boxes`, `/v1/matting_box` `box`) and output `box_px` / `point_px` are pixels in the
  **original image after EXIF rotation**, origin top-left, `[x1, y1, x2, y2]` / `[x, y]`.
- `box_norm` / `point_norm` are the same positions as fractions 0..1 (the model works on a 0-1000 grid).
- `image_size` is the original size; `processed_size` is what the model saw (downscaled above
  {s.max_pixels:,} pixels). Renders and `processed`-resolution masks use `processed_size`.

## Images

- `image.b64` (raw base64 or data URL) or `image.url` (URL fetch {fetch}, {s.url_fetch_timeout_s}s timeout).
- JPEG, PNG, WEBP, BMP, GIF (first frame), MPO. Transparency is flattened onto white.
- Limits: {s.max_image_bytes:,} bytes, {s.max_image_pixels:,} pixels, shorter side >= 8 px, aspect ratio <= 200.

## Reading results

- `groups` keep request order for label tasks; a label the model did not find has `absent: true`, no instances,
  and a warning.
- Counting: `declared_total` is the number the model stated, `count_by_items` the instances returned,
  `total_consistent` whether they agree.
- `truncated: true` means generation hit `max_new_tokens`; results may be incomplete.
- `warnings` reports absent labels and any malformed model output that was skipped.
- `raw_response` is the exact model text, useful for freeform answers and debugging.

## Renders and masks

- Views: `overlay` (all tasks), `alpha` / `green_screen` / `cutout` (segmentation, matting, matting_box only).
- `render_delivery` / `mask_delivery`: `url` (stored, fetch with `GET /v1/artifacts/{{request_id}}/{{name}}`,
  kept {s.artifact_ttl_s}s) or `base64` (inline). Artifact URLs are
  {"absolute (`" + s.public_base_url + "/...`)" if s.public_base_url else "relative to this server (`/v1/artifacts/...`)"}.
- Segmentation masks are binary (0/255); matting masks are soft alpha (0..255).

## Errors and retries

Every error except `/v1/health` uses `{{"error": {{"code", "message", "details"}}}}`. Branch on `code`:

| code | HTTP | retry? |
|---|---|---|
| `invalid_request` | 422 | no, fix the request (`details.errors` for schema problems) |
| `image_too_large` | 413 | no, shrink the image |
| `unsupported_image` | 415 | no, convert to JPEG/PNG |
| `image_fetch_failed` | 400 | no, send `b64` or fix the URL |
| `not_found` | 404 | no, artifact expired or wrong path |
| `not_ready` | 503 | yes, after `GET /v1/health` is `ready` |
| `busy` | 503 | yes, after `Retry-After` seconds |
| `timeout` | 504 | yes, later or with a smaller job |
| `internal` | 500 | no |

## Concurrency

One GPU job runs at a time; up to {s.max_queue} requests wait in line, beyond that `busy`. Each request must
finish within {s.sync_timeout_s}s including queueing, else `timeout`. Typical model time: 0.4-2 s per request
for a ~1 MP image.
"""
