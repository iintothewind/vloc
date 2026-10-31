from io import BytesIO

import pytest
from PIL import Image

from vloc.config import Settings
from vloc.errors import ApiError
from vloc.images import decode_image, fit_to_budget

S = Settings()


def _encode(img: Image.Image, fmt: str, **kw) -> bytes:
    buf = BytesIO()
    img.save(buf, format=fmt, **kw)
    return buf.getvalue()


def test_exif_orientation_applied():
    exif = Image.Exif()
    exif[0x0112] = 6
    data = _encode(Image.new("RGB", (40, 20), (10, 20, 30)), "JPEG", exif=exif)
    assert decode_image(data, S).size == (20, 40)


def test_transparent_rgba_becomes_white():
    data = _encode(Image.new("RGBA", (16, 16), (0, 0, 0, 0)), "PNG")
    img = decode_image(data, S)
    assert img.mode == "RGB" and img.getpixel((5, 5)) == (255, 255, 255)


@pytest.mark.parametrize("size", [(300, 1), (5, 5), (1700, 8)])
def test_degenerate_shapes_rejected(size):
    with pytest.raises(ApiError) as e:
        decode_image(_encode(Image.new("RGB", size), "PNG"), S)
    assert e.value.status == 422


def test_fit_to_budget_preserves_aspect():
    img = fit_to_budget(Image.new("RGB", (8000, 6000)), 2007040)
    w, h = img.size
    assert w * h <= 2007040
    assert abs(w * 6000 / 8000 - h) <= 1
    small = Image.new("RGB", (100, 50))
    assert fit_to_budget(small, 2007040) is small


def test_pixel_limit():
    data = _encode(Image.new("RGB", (200, 200)), "PNG")
    with pytest.raises(ApiError) as e:
        decode_image(data, Settings(max_image_pixels=39999))
    assert e.value.status == 413


def test_garbage_bytes_unsupported():
    with pytest.raises(ApiError) as e:
        decode_image(b"not an image at all", S)
    assert e.value.status == 415


def test_byte_limit_via_api(make_client):
    import base64

    data = base64.b64encode(_encode(Image.new("RGB", (64, 64)), "PNG")).decode()
    with make_client(max_image_bytes=10) as c:
        r = c.post("/v1/perceive", json={"image": {"b64": data}, "task": "detection", "query": "dog"})
    assert r.status_code == 413 and r.json()["error"]["code"] == "image_too_large"


def test_data_url_and_bad_base64(client):
    import base64

    raw = base64.b64encode(_encode(Image.new("RGB", (64, 64)), "PNG")).decode()
    ok = client.post(
        "/v1/perceive", json={"image": {"b64": f"data:image/png;base64,{raw}"}, "task": "detection", "query": "dog"}
    )
    assert ok.status_code == 200
    bad = client.post("/v1/perceive", json={"image": {"b64": "@@@"}, "task": "detection", "query": "dog"})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "invalid_request"
