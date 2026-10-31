import base64
import threading
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from tests.conftest import IMG_H, IMG_W, jpeg_b64
from tests.fake_engine import ABSENT, FakeEngine


def _req(task, query=None, **options):
    body = {"image": {"b64": jpeg_b64()}, "task": task, "options": options}
    if query is not None:
        body["query"] = query
    return body


def _fetch_png(client, art) -> np.ndarray:
    r = client.get(art["url"])
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    return np.asarray(Image.open(BytesIO(r.content)))


def test_detection_boxes_in_original_pixels(client):
    r = client.post("/v1/perceive", json=_req("detection", "dogs"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "box" and body["model"] == "fake-2B"
    assert body["image_size"] == [IMG_W, IMG_H] and body["renders"] == []
    inst = body["groups"][0]["instances"]
    assert len(inst) == 4
    assert inst[0]["box_px"][0] == round(0.013 * IMG_W, 1)
    assert inst[0]["box_px"][3] == round(0.949 * IMG_H, 1)
    assert body["count_by_items"] == 4 and body["total_consistent"] is None


def test_render_overlay_url(client, settings):
    r = client.post("/v1/perceive", json=_req("detection", "dogs", render=True)).json()
    [art] = r["renders"]
    assert art["name"] == "overlay.jpg" and art["url"] == f"/v1/artifacts/{r['request_id']}/overlay.jpg"
    g = client.get(art["url"])
    assert g.status_code == 200 and g.headers["content-type"] == "image/jpeg"
    assert Image.open(BytesIO(g.content)).size == tuple(r["processed_size"])


def test_render_base64_writes_nothing(client, settings):
    r = client.post("/v1/perceive", json=_req("detection", "dogs", render=True, render_delivery="base64")).json()
    [art] = r["renders"]
    assert art["url"] is None
    assert Image.open(BytesIO(base64.b64decode(art["b64"]))).format == "JPEG"
    assert not (Path(settings.artifact_dir) / r["request_id"]).exists()


def test_segmentation_all_views_and_binary_masks(client):
    r = client.post(
        "/v1/perceive",
        json=_req("segmentation", "dogs", render=True, render_views=["overlay", "alpha", "green_screen", "cutout"]),
    ).json()
    assert [a["name"] for a in r["renders"]] == ["overlay.jpg", "alpha.png", "green_screen.jpg", "cutout.png"]
    cutout = client.get(r["renders"][3]["url"])
    assert Image.open(BytesIO(cutout.content)).mode == "RGBA"
    insts = r["groups"][0]["instances"]
    assert [i["mask"]["name"] for i in insts] == [f"mask_{k:03d}.png" for k in range(4)]
    assert set(np.unique(_fetch_png(client, insts[0]["mask"]))) <= {0, 255}
    assert insts[0]["mask_outside_box_fraction"] > 0


def test_matting_mask_is_soft(client):
    r = client.post("/v1/perceive", json=_req("matting", "dogs")).json()
    vals = set(np.unique(_fetch_png(client, r["groups"][0]["instances"][0]["mask"])))
    assert vals - {0, 255}, "matting mask must keep intermediate alpha"


def test_mask_original_resolution_and_none(make_client):
    big = {"image": {"b64": jpeg_b64((2400, 1600))}, "task": "segmentation", "query": "dogs"}
    with make_client(max_pixels=600_000) as c:
        r = c.post("/v1/perceive", json={**big, "options": {"mask_resolution": "original"}}).json()
        m = r["groups"][0]["instances"][0]["mask"]
        assert r["processed_size"][0] < 2400 and (m["width"], m["height"]) == (2400, 1600)
        r = c.post("/v1/perceive", json={**big, "options": {"mask_delivery": "none"}}).json()
        inst = r["groups"][0]["instances"][0]
        assert inst["mask"] is None and inst["mask_mean"] is not None


def test_pixel_view_rejected_for_box_task(client):
    r = client.post("/v1/perceive", json=_req("detection", "dogs", render=True, render_views=["alpha"]))
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_request"


def test_count_consistency(client):
    r = client.post("/v1/perceive", json=_req("count_box", "dog")).json()
    assert (r["declared_total"], r["count_by_items"], r["total_consistent"]) == (3, 2, False)


def test_absent_label(make_client):
    with make_client(FakeEngine({"Locate all": ABSENT})) as c:
        r = c.post("/v1/perceive", json=_req("detection", ["dogs", "people"])).json()
    assert [(g["label"], g["absent"]) for g in r["groups"]] == [("dogs", False), ("people", True)]


def test_invalid_task_and_schema(client):
    r = client.post("/v1/perceive", json=_req("segment", "x"))
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_request"
    r = client.post("/v1/perceive", json={"image": {}, "task": "detection"})
    assert r.status_code == 422 and r.json()["error"]["details"]["errors"]


def test_artifact_lookup_rejects_bad_paths(client):
    assert client.get("/v1/artifacts/../../etc/passwd").status_code == 404
    assert client.get("/v1/artifacts/notanid/overlay.jpg").json()["error"]["code"] == "not_found"
    assert client.get(f"/v1/artifacts/{'a' * 32}/overlay.jpg").status_code == 404
    assert client.get(f"/v1/artifacts/{'a' * 32}/..%2Fx.png").status_code == 404


def test_matting_box(client):
    box = [0.013 * IMG_W, 0.274 * IMG_H, 0.398 * IMG_W, 0.949 * IMG_H]
    r = client.post("/v1/matting_box", json={"image": {"b64": jpeg_b64()}, "box": box, "phrase": "dog"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["task"] == "matting_box" and body["mode"] == "pixel"
    assert body["raw_response"] == "<ref>dog</ref><box><13><274><398><949><OBJ></box>"
    [inst] = body["groups"][0]["instances"]
    assert inst["mask"] is not None and 0 < inst["mask_mean"] < 1
    bad = client.post("/v1/matting_box", json={"image": {"b64": jpeg_b64()}, "box": [50, 50, 50, 80]})
    assert bad.status_code == 422


def test_tasks_catalog(client):
    tasks = client.get("/v1/tasks").json()["tasks"]
    assert len(tasks) == 12 and tasks[-1]["task"] == "matting_box"


def _post_in_thread(client, results, key):
    results[key] = client.post("/v1/perceive", json=_req("detection", "dogs"))


def test_queue_full_returns_busy(make_client):
    block, entered = threading.Event(), threading.Event()
    with make_client(FakeEngine(block=block, entered=entered), max_queue=1) as c:
        results = {}
        t = threading.Thread(target=_post_in_thread, args=(c, results, "first"))
        t.start()
        try:
            assert entered.wait(5)
            second = c.post("/v1/perceive", json=_req("detection", "dogs"))
            assert second.status_code == 503 and second.json()["error"]["code"] == "busy"
            assert second.headers["retry-after"] == "5"
        finally:
            block.set()
            t.join(10)
        assert results["first"].status_code == 200


def test_sync_timeout(make_client):
    block = threading.Event()
    with make_client(FakeEngine(block=block), sync_timeout_s=1) as c:
        try:
            r = c.post("/v1/perceive", json=_req("detection", "dogs"))
            assert r.status_code == 504 and r.json()["error"]["code"] == "timeout"
        finally:
            block.set()


def test_openapi_documents_error_contract(client):
    from vloc.errors import STATUS

    spec = client.get("/openapi.json").json()
    ref = {"$ref": "#/components/schemas/ErrorResponse"}
    perceive = spec["paths"]["/v1/perceive"]["post"]["responses"]
    expected = {str(s) for s in STATUS.values()} - {"404"}
    assert expected <= set(perceive)
    for status in expected:
        assert perceive[status]["content"]["application/json"]["schema"] == ref
    assert "Retry-After" in perceive["503"]["headers"]
    assert "HTTPValidationError" not in spec["components"]["schemas"]
    assert set(spec["components"]["schemas"]["ErrorDetail"]["properties"]["code"]["enum"]) == set(STATUS)
    artifact = spec["paths"]["/v1/artifacts/{request_id}/{name}"]["get"]["responses"]
    assert artifact["404"]["content"]["application/json"]["schema"] == ref
    assert {"image/png", "image/jpeg"} <= set(artifact["200"]["content"])
    for path, op in (("/v1/perceive", "post"), ("/v1/artifacts/{request_id}/{name}", "get")):
        for status, resp in spec["paths"][path][op]["responses"].items():
            if status == "200":
                continue
            media = resp["content"]["application/json"]
            bodies = [media["example"]] if "example" in media else [e["value"] for e in media["examples"].values()]
            assert all(STATUS[b["error"]["code"]] == int(status) for b in bodies), (path, status)


def _with_test_image(body):
    return {**body, "image": {"b64": jpeg_b64()}}


def test_documented_examples_are_accepted(client):
    """Every request example an agent can copy (/v1/tasks and OpenAPI) must pass validation and run."""
    spec = client.get("/openapi.json").json()
    tasks = client.get("/v1/tasks").json()["tasks"]
    for entry in tasks:
        path = "/v1/matting_box" if entry["task"] == "matting_box" else "/v1/perceive"
        r = client.post(path, json=_with_test_image(entry["request_example"]))
        assert r.status_code == 200, (entry["task"], r.text)
    for path in ("/v1/perceive", "/v1/matting_box"):
        examples = spec["paths"][path]["post"]["requestBody"]["content"]["application/json"]["examples"]
        for name, ex in examples.items():
            r = client.post(path, json=_with_test_image(ex["value"]))
            assert r.status_code == 200, (path, name, r.text)


def test_openapi_describes_every_operation_and_request_field(client):
    spec = client.get("/openapi.json").json()
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            assert op.get("summary") and op.get("description"), (method, path)
    schemas = spec["components"]["schemas"]
    for model in ("PerceiveRequest", "MattingBoxRequest", "ImageInput", "Options", "RenderStyle"):
        for field, prop in schemas[model]["properties"].items():
            assert prop.get("description") or "$ref" in prop, (model, field)
    assert schemas["PerceiveRequest"]["properties"]["task"]["enum"]


def test_health_loading_while_factory_blocks(settings):
    from tests.conftest import running_client

    gate = threading.Event()

    def factory():
        gate.wait(10)
        return FakeEngine()

    with running_client(settings, factory, wait_ready=False) as c:
        try:
            h = c.get("/v1/health")
            assert h.status_code == 503 and h.json()["status"] == "loading"
            assert c.get("/healthz").json() == {"status": "alive"}
            r = c.post("/v1/perceive", json=_req("detection", "dogs"))
            assert r.status_code == 503 and r.json()["error"]["code"] == "not_ready"
        finally:
            gate.set()
