from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class Settings:
    model_path: str = "/models/WeVisOne-VL-2B"
    min_pixels: int = 1048576
    max_pixels: int = 2007040
    artifact_dir: str = "/data/artifacts"
    artifact_ttl_s: int = 86400
    public_base_url: str = ""
    sync_timeout_s: int = 60
    max_queue: int = 8
    max_image_bytes: int = 20000000
    max_image_pixels: int = 50000000
    allow_url_fetch: bool = True
    url_fetch_timeout_s: int = 10

    @classmethod
    def from_env(cls) -> Settings:
        d = cls()
        return cls(
            model_path=os.environ.get("WEVISONE_MODEL") or d.model_path,
            min_pixels=_int("VLOC_MIN_PIXELS", d.min_pixels),
            max_pixels=_int("VLOC_MAX_PIXELS", d.max_pixels),
            artifact_dir=os.environ.get("VLOC_ARTIFACT_DIR") or d.artifact_dir,
            artifact_ttl_s=_int("VLOC_ARTIFACT_TTL_S", d.artifact_ttl_s),
            public_base_url=os.environ.get("VLOC_PUBLIC_BASE_URL", d.public_base_url).rstrip("/"),
            sync_timeout_s=_int("VLOC_SYNC_TIMEOUT_S", d.sync_timeout_s),
            max_queue=_int("VLOC_MAX_QUEUE", d.max_queue),
            max_image_bytes=_int("VLOC_MAX_IMAGE_BYTES", d.max_image_bytes),
            max_image_pixels=_int("VLOC_MAX_IMAGE_PIXELS", d.max_image_pixels),
            allow_url_fetch=_bool("VLOC_ALLOW_URL_FETCH", d.allow_url_fetch),
            url_fetch_timeout_s=_int("VLOC_URL_FETCH_TIMEOUT_S", d.url_fetch_timeout_s),
        )
