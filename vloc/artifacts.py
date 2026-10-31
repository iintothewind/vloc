from __future__ import annotations

import base64
import re
import shutil
import time
from pathlib import Path

from vloc.errors import ApiError
from vloc.schemas import Artifact

REQUEST_ID_RE = re.compile(r"^[0-9a-f]{32}$")
NAME_RE = re.compile(r"^[a-z_]+(_\d{3})?\.(png|jpg)$")
MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg"}


class ArtifactStore:
    def __init__(self, root: Path, ttl_s: int, public_base_url: str):
        self.root = Path(root)
        self.ttl_s = ttl_s
        self.base = public_base_url.rstrip("/")

    def put(
        self,
        request_id: str,
        name: str,
        data: bytes,
        media_type: str,
        width: int,
        height: int,
        delivery: str,
    ) -> Artifact:
        if delivery == "base64":
            return Artifact(
                name=name, media_type=media_type, width=width, height=height, b64=base64.b64encode(data).decode()
            )
        d = self.root / request_id
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_bytes(data)
        url = f"{self.base}/v1/artifacts/{request_id}/{name}"
        return Artifact(name=name, media_type=media_type, width=width, height=height, url=url)

    def _expired(self, d: Path, now: float) -> bool:
        return now - d.stat().st_mtime > self.ttl_s

    def path(self, request_id: str, name: str) -> Path:
        if not REQUEST_ID_RE.match(request_id) or not NAME_RE.match(name):
            raise ApiError("not_found", "artifact not found")
        d = self.root / request_id
        p = d / name
        if not p.is_file():
            raise ApiError("not_found", "artifact not found")
        if self._expired(d, time.time()):
            shutil.rmtree(d, ignore_errors=True)
            raise ApiError("not_found", "artifact expired")
        return p

    def sweep(self) -> int:
        if not self.root.is_dir():
            return 0
        now = time.time()
        removed = 0
        for d in self.root.iterdir():
            if d.is_dir() and REQUEST_ID_RE.match(d.name) and self._expired(d, now):
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
        return removed


def media_type_for(name: str) -> str:
    return MEDIA_TYPES[Path(name).suffix]
