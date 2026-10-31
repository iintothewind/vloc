import json, sys, time, resource
import numpy as np
import torch
from PIL import Image
from wevisone_vl import WeVisOneVL

E = "WeVisOne-VL/demo/examples/images"
MODEL = sys.argv[1]
TAG = MODEL.split("/")[-1]


def emit(row):
    row = {"model": TAG, **row}
    print(json.dumps(row, ensure_ascii=False), flush=True)


def peak():
    return round(torch.cuda.max_memory_allocated() / 2**20)


def rss_mib():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def run(eng, img, task, query, reps=2):
    """Warm once, then return (median latency, peak MiB, last result)."""
    eng.predict(img, task=task, query=query)
    lat, peaks, r = [], [], None
    for _ in range(reps):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        r = eng.predict(img, task=task, query=query)
        lat.append(time.perf_counter() - t0)
        peaks.append(peak())
    return round(sorted(lat)[len(lat) // 2], 3), max(peaks), r


def iou(a, b):
    a, b = a > 0.5, b > 0.5
    u = np.logical_or(a, b).sum()
    return round(float(np.logical_and(a, b).sum() / u), 4) if u else 1.0


t0 = time.perf_counter()
eng = WeVisOneVL(MODEL)
emit({"kind": "load", "load_s": round(time.perf_counter() - t0, 1), "peak_MiB": peak()})

dogs = Image.open(f"{E}/dogs.jpg").convert("RGB")

# A. input size sweep: segmentation (4 masks) + detection on upscaled copies
SIZES = [(1024, 696), (1920, 1080), (3840, 2160), (6000, 4000), (8000, 6000)]
base_seg = None
for w, h in SIZES:
    img = dogs if (w, h) == dogs.size else dogs.resize((w, h), Image.Resampling.LANCZOS)
    for task in ("detection", "segmentation"):
        lat, pk, r = run(eng, img, task, "dogs")
        row = {"kind": "size_sweep", "task": task, "w": w, "h": h, "mp": round(w * h / 1e6, 2),
               "p50_s": lat, "peak_MiB": pk, "n_masks": len(r.masks),
               "mask_bytes_total": int(sum(m.nbytes for m in r.masks)),
               "response_head": r.response[:160], "rss_max_MiB": rss_mib()}
        if task == "segmentation" and (w, h) == (1920, 1080):
            base_seg = (r.response, r.masks)
        emit(row)

# A2. server-side pre-downscale: 8000x6000 -> fit 2,073,600 px, run, upsample masks back; compare
big = dogs.resize((8000, 6000), Image.Resampling.LANCZOS)
scale = (2073600 / (8000 * 6000)) ** 0.5
small_wh = (int(8000 * scale), int(6000 * scale))
t0 = time.perf_counter()
small = big.resize(small_wh, Image.Resampling.LANCZOS)
t_resize = time.perf_counter() - t0
lat_s, pk_s, r_s = run(eng, small, "segmentation", "dogs")
_, _, r_b = run(eng, big, "segmentation", "dogs", reps=1)
ups = [np.asarray(Image.fromarray(m).resize((8000, 6000), Image.Resampling.BILINEAR)) for m in r_s.masks]
ious = [iou(a, b) for a, b in zip(ups, r_b.masks)]
emit({"kind": "predownscale", "from": [8000, 6000], "to": list(small_wh), "resize_s": round(t_resize, 3),
      "p50_s_small": lat_s, "peak_MiB_small": pk_s, "same_response": r_s.response == r_b.response,
      "n_masks_small": len(r_s.masks), "n_masks_big": len(r_b.masks), "mask_iou_vs_direct": ious})

# B. batch sweep (box-only detection, then segmentation)
for task, ns in (("detection", [1, 2, 4, 8, 16, 32]), ("segmentation", [1, 4, 8, 16])):
    for n in ns:
        imgs = [dogs] * n
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            res = eng.predict_batch(imgs, task=task, query="dogs")
            dt = time.perf_counter() - t0
            emit({"kind": "batch_sweep", "task": task, "n": n, "total_s": round(dt, 2),
                  "per_img_s": round(dt / n, 3), "peak_MiB": peak(),
                  "all_same": len({x.response for x in res}) == 1})
        except torch.OutOfMemoryError as exc:
            emit({"kind": "batch_sweep", "task": task, "n": n, "error": "OOM", "msg": str(exc)[:200]})
            torch.cuda.empty_cache()
            break

# C. label count sweep for detection (COCO names)
COCO = [c.replace("_", " ") for c in (
    "person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign "
    "parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella "
    "handbag tie suitcase frisbee skis snowboard sports_ball kite baseball_bat baseball_glove skateboard "
    "surfboard tennis_racket bottle wine_glass cup fork knife spoon bowl banana apple sandwich orange "
    "broccoli carrot hot_dog pizza donut cake chair couch potted_plant bed dining_table toilet tv laptop "
    "mouse remote keyboard cell_phone microwave oven toaster sink refrigerator book clock vase scissors "
    "teddy_bear hair_drier toothbrush").split()]
living = Image.open(f"{E}/livingroom.jpg").convert("RGB")
tok = eng.tokenizer
for n in (1, 5, 10, 20, 40, 80):
    labels = COCO[:n]
    lat, pk, r = run(eng, living, "detection", labels, reps=1)
    emit({"kind": "label_sweep", "n_labels": n, "p50_s": lat, "peak_MiB": pk,
          "prompt_tokens": len(tok(r.prompt)["input_ids"]),
          "response_tokens": len(tok(r.response)["input_ids"]),
          "n_absent": r.response.count("none"), "truncated": not r.response.rstrip().endswith(("</box>", "</point>", "."))})

# D. query length sweep for referring
for n_chars in (20, 200, 1000, 4000):
    base = "the brown dog lying on the left side of the image next to the other dogs "
    q = (base * (n_chars // len(base) + 1))[:n_chars]
    lat, pk, r = run(eng, dogs, "referring", q, reps=1)
    emit({"kind": "query_len_sweep", "chars": n_chars, "prompt_tokens": len(tok(r.prompt)["input_ids"]),
          "p50_s": lat, "peak_MiB": pk, "response_head": r.response[:120]})

emit({"done": True})
