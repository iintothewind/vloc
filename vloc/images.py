from __future__ import annotations

import base64
import binascii
import math
from io import BytesIO
from urllib.parse import urlparse

import httpx
from PIL import Image, ImageOps, UnidentifiedImageError

from vloc.config import Settings
from vloc.errors import ApiError
from vloc.schemas import ImageInput

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "GIF", "MPO"}


async def load_image(inp: ImageInput, s: Settings) -> bytes:
    if inp.b64 is not None:
        raw = inp.b64.strip()
        if raw.startswith("data:"):
            head, sep, rest = raw.partition(",")
            if not sep or not head.endswith(";base64"):
                raise ApiError("invalid_request", "image.b64 is not valid base64")
            raw = rest
        try:
            data = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError):
            raise ApiError("invalid_request", "image.b64 is not valid base64") from None
        if len(data) > s.max_image_bytes:
            raise ApiError("image_too_large", f"image exceeds {s.max_image_bytes} bytes")
        return data

    url = inp.url or ""
    if not s.allow_url_fetch:
        raise ApiError("image_fetch_failed", "url fetch disabled")
    if urlparse(url).scheme.lower() not in ("http", "https"):
        raise ApiError("image_fetch_failed", "image.url scheme must be http or https")
    buf = bytearray()
    try:
        async with httpx.AsyncClient(timeout=s.url_fetch_timeout_s, follow_redirects=True) as client:
            async with client.stream("GET", url) as resp:
                if not 200 <= resp.status_code < 300:
                    raise ApiError(
                        "image_fetch_failed",
                        f"image.url returned HTTP {resp.status_code}",
                        {"status": resp.status_code},
                    )
                async for chunk in resp.aiter_bytes():
                    buf.extend(chunk)
                    if len(buf) > s.max_image_bytes:
                        raise ApiError("image_too_large", f"image exceeds {s.max_image_bytes} bytes")
    except httpx.HTTPError as exc:
        raise ApiError("image_fetch_failed", f"image.url fetch failed: {str(exc) or type(exc).__name__}") from None
    return bytes(buf)


def decode_image(data: bytes, s: Settings) -> Image.Image:
    try:
        img = Image.open(BytesIO(data))
    except UnidentifiedImageError:
        raise ApiError("unsupported_image", "cannot identify image file") from None
    except Image.DecompressionBombError:
        raise ApiError("image_too_large", f"image exceeds {s.max_image_pixels} pixels") from None
    if img.format not in ALLOWED_FORMATS:
        raise ApiError("unsupported_image", f"image format {img.format} not supported")
    w, h = img.size
    if w * h > s.max_image_pixels:
        raise ApiError("image_too_large", f"image exceeds {s.max_image_pixels} pixels")
    try:
        if getattr(img, "is_animated", False):
            img.seek(0)
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, rgba).convert("RGB")
        else:
            img = img.convert("RGB")
    except Image.DecompressionBombError:
        raise ApiError("image_too_large", f"image exceeds {s.max_image_pixels} pixels") from None
    except (OSError, SyntaxError, ValueError) as exc:
        raise ApiError("unsupported_image", f"cannot decode image: {exc}") from None
    w, h = img.size
    if min(w, h) < 8:
        raise ApiError("invalid_request", "image shorter side must be >= 8 px")
    if max(w, h) / min(w, h) > 200:
        raise ApiError("invalid_request", "image aspect ratio must be <= 200")
    return img


def fit_to_budget(img: Image.Image, max_pixels: int) -> Image.Image:
    w, h = img.size
    if w * h <= max_pixels:
        return img
    s = math.sqrt(max_pixels / (w * h))
    size = (max(1, math.floor(w * s)), max(1, math.floor(h * s)))
    return img.resize(size, Image.Resampling.LANCZOS)
