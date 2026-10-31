---
name: vloc
description: Locate, count, read or cut out things in an image via the vloc HTTP service (WeVisOne-VL-2B on a remote GPU). Use for object detection/bounding boxes, "where is X", pointing, counting, OCR / finding text, segmentation masks, background removal / alpha matting, find-similar-by-example, and short visual Q&A. Never load a vision model locally; call this service.
---

# vloc perception service

Base URL: `{{BASE_URL}}`

One image per request, synchronous, ~0.4-2 s. The service is the source of truth:

- `GET {{BASE_URL}}/v1/tasks`: every task with what it does, required input, limits, and a ready-to-send body. Read this first if unsure which task fits.
- `GET {{BASE_URL}}/openapi.json`: full field-level contract, error codes, retry rules.
- `GET {{BASE_URL}}/v1/health`: `{"status":"ready"}` (200) before sending work; 503 `loading` means wait and retry.

## Pick the task

| Need | Endpoint + task | Input |
|---|---|---|
| Boxes for categories ("all dogs and cars") | `/v1/perceive` `detection` | `query`: list of 1-20 short nouns |
| The one described ("the man in the red shirt") | `/v1/perceive` `referring` | `query`: one phrase |
| A point per object | `/v1/perceive` `point` | `query`: list of nouns |
| How many | `/v1/perceive` `count_box` / `count_point` | `query`: exactly one noun |
| All text in the image | `/v1/perceive` `ocr` | no query |
| Where a given text is | `/v1/perceive` `ocr_grounding` | `query`: the text |
| Pixel mask (hard edges) | `/v1/perceive` `segmentation` | `query`: one phrase |
| Background removal / soft alpha | `/v1/perceive` `matting` | `query`: one phrase |
| Matte for a box you already have | `/v1/matting_box` | `box` (pixels) + `phrase` |
| More things like this example | `/v1/perceive` `visual_prompt` | `reference_boxes`: 1-8 pixel boxes, no query |
| Caption / question about the image | `/v1/perceive` `freeform` | `query`: the instruction |

## Call it with a local file

Send local files as base64 (`image.b64`); use `image.url` only for public http(s) URLs.

```python
import base64, json, urllib.error, urllib.request

BASE = "{{BASE_URL}}"

def vloc(path, body, endpoint="/v1/perceive"):
    with open(path, "rb") as f:
        body = {"image": {"b64": base64.b64encode(f.read()).decode()}, **body}
    req = urllib.request.Request(BASE + endpoint, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=90))
    except urllib.error.HTTPError as e:
        raise RuntimeError(json.load(e)["error"])  # {"code", "message", "details"}

r = vloc("photo.jpg", {"task": "detection", "query": ["dog", "person"]})
for g in r["groups"]:
    print(g["label"], "absent" if g["absent"] else [i["box_px"] for i in g["instances"]])
```

```bash
# Linux/macOS shell equivalent
printf '{"task":"detection","query":["dog"],"image":{"b64":"%s"}}' "$(base64 -w0 photo.jpg)" \
  | curl -s -H 'Content-Type: application/json' --data-binary @- {{BASE_URL}}/v1/perceive
```

## Read the result

- `groups[]`: one per label, request order. `absent: true` = model says it is not in the image.
- `instances[].box_px` `[x1,y1,x2,y2]` / `point_px` `[x,y]`: pixels in the original image (after EXIF rotation). `*_norm` are 0..1 fractions.
- Counting: trust it when `total_consistent` is true; if false, `count_by_items` (boxes returned) and `declared_total` (number stated) disagree, report both.
- `freeform`: the answer is `raw_response`; `groups` only hold phrases the model grounded.
- `truncated: true`: output hit `options.max_new_tokens` (max 1024); split the request (fewer labels) instead.
- `warnings`: absent labels and skipped malformed output; mention them if they matter.

## Get images back

Add `"options": {"render": true}` to get `renders[]`, an `overlay` with boxes/labels drawn. For segmentation, matting and matting_box, also `"render_views": ["overlay", "cutout"]` (cutout = transparent PNG); `alpha` and `green_screen` exist too.

- Each artifact has a `url`; if it starts with `/`, prefix `{{BASE_URL}}`. Files expire after {{ARTIFACT_TTL_H}} h; download them if needed later.
- To avoid a second request, set `"render_delivery": "base64"` and decode `renders[].b64`.
- Instance masks (`instances[].mask`): segmentation PNGs are 0/255, matting PNGs are soft 0..255.

## Recipes

- **Cut out one specific object:** `detection` or `referring` → choose the instance → `/v1/matting_box` with its `box_px` and a short `phrase`, `options.render_views: ["cutout"]`. Faster than `matting` and never comes back empty.
- **Show the user what was found:** any box/point task with `render: true`, then fetch the `overlay` URL and display or save it.
- **Find more items like one the user marked:** `visual_prompt` with their box; results usually include the example box itself.
- **Check a count:** `count_box` and look at `total_consistent`; render the overlay if the user needs proof.

## Tips and quirks

- Labels work best as short nouns (`"red car"`, not sentences). `count_*` takes exactly one label.
- `<` and `>` are stripped from queries.
- OCR returns one group per distinct word (label = the text); repeated words share a group.
- `visual_prompt` labels its group like `object 1: Dog` (the model names the category).
- `freeform` counting questions may come back as points plus `Total number: N` instead of a sentence.
- Images above {{MAX_PIXELS}} pixels are downscaled before inference (`processed_size`); coordinates are still returned in original pixels. Tiny details in huge images can be missed; crop first.
- Limits: {{MAX_IMAGE_BYTES}} bytes, {{MAX_IMAGE_PIXELS}} pixels, shorter side >= 8 px. Formats: JPEG, PNG, WEBP, BMP, GIF, MPO.{{URL_FETCH_NOTE}}

## Errors

Branch on `error.code`: fix and resend for `invalid_request` / `image_too_large` / `unsupported_image` / `image_fetch_failed` (422/413/415/400; `details.errors` explains schema problems). Retry for `busy` (503, wait `Retry-After` s), `not_ready` (503, poll `/v1/health`), `timeout` (504). One GPU job runs at a time, so don't fire many requests in parallel.
