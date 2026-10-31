from __future__ import annotations

import base64
import contextlib
import dataclasses
import time
from io import BytesIO
from typing import Callable, Iterator

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tests.fake_engine import FakeEngine
from vloc.app import create_app
from vloc.config import Settings

IMG_W, IMG_H = 1200, 800


def jpeg_b64(size: tuple[int, int] = (IMG_W, IMG_H), color=(90, 120, 150)) -> str:
    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(model_path="/models/fake-2B", artifact_dir=str(tmp_path / "artifacts"))


@contextlib.contextmanager
def running_client(
    settings: Settings, factory: Callable[[], object], wait_ready: bool = True
) -> Iterator[TestClient]:
    with TestClient(create_app(settings, factory)) as client:
        if wait_ready:
            deadline = time.monotonic() + 2
            while client.get("/v1/health").status_code != 200:
                assert time.monotonic() < deadline, "fake engine never became ready"
                time.sleep(0.02)
        yield client


@pytest.fixture
def make_client(settings) -> Callable[..., contextlib.AbstractContextManager[TestClient]]:
    def make(engine: FakeEngine | None = None, wait_ready: bool = True, **overrides):
        s = dataclasses.replace(settings, **overrides)
        eng = engine or FakeEngine()
        return running_client(s, lambda: eng, wait_ready)

    return make


@pytest.fixture
def client(make_client) -> Iterator[TestClient]:
    with make_client() as c:
        yield c
