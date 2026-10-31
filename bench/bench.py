import json, sys, time
import numpy as np
import torch
from PIL import Image
from wevisone_vl import WeVisOneVL

E = "WeVisOne-VL/demo/examples/images"
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
MODELS = sys.argv[2:]

def emit(row):
    print(json.dumps(row) + "\n", flush=True)

def pct(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 3)

def mem():
    return round(torch.cuda.max_memory_allocated() / 2**20)

big = Image.open(f"{E}/dogs.jpg").convert("RGB").resize((1920, 1080), Image.Resampling.LANCZOS)

CASES = [
    ("detection_multilabel_dogs", "dogs.jpg", "detection", "dogs\npeople"),
    ("referring_dogs", "dogs.jpg", "referring", "dogs"),
    ("point_dogs", "dogs.jpg", "point", "dogs"),
    ("count_box_chess16", "chess.jpg", "count_box", "white chess pieces"),
    ("count_point_chairs_dense", "livingroom.jpg", "count_point", "chairs"),
    ("ocr_coffee_sign", "coffee_sign.jpg", "ocr", ""),
    ("segmentation_dogs_0p71MP", "dogs.jpg", "segmentation", "dogs"),
    ("matting_person_1p75MP", "seabreeze.jpg", "matting", "person"),
    ("segmentation_dogs_2MP", "__big__", "segmentation", "dogs"),
]

def get_img(tag):
    return big if tag == "__big__" else Image.open(f"{E}/{tag}").convert("RGB")

def bench(eng, model_tag, cases, reps):
    rows = []
    for name, tag, task, query in cases:
        img = get_img(tag)
        lat, peaks = [], []
        warm = None
        for i in range(reps + 1):
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            r = eng.predict(img, task=task, query=query)
            dt = time.perf_counter() - t0
            if i == 0:
                warm = round(dt, 3)
                continue
            lat.append(round(dt, 3))
            peaks.append(mem())
        row = {
            "model": model_tag, "case": name, "task": task,
            "warmup_s": warm, "p50": pct(lat, 0.5), "p95": pct(lat, 0.95),
            "min_s": min(lat), "max_s": max(lat), "peak_MiB": max(peaks),
            "n_reps": len(lat), "n_boxes": len(r.boxes), "n_masks": len(r.masks),
        }
        rows.append(row)
        emit(row)
    return rows

def queue(eng, model_tag, items):
    lat = []
    t_start = time.perf_counter()
    for tag, task, query in items:
        img = get_img(tag)
        t0 = time.perf_counter()
        eng.predict(img, task=task, query=query)
        lat.append(round(time.perf_counter() - t0, 3))
    total = time.perf_counter() - t_start
    row = {"model": model_tag, "kind": "sequential_queue", "n": len(items),
           "total_s": round(total, 2), "rps": round(len(items) / total, 3),
           "p50": pct(lat, 0.5), "p95": pct(lat, 0.95), "per_latency": lat,
           "peak_MiB": mem()}
    emit(row)
    return row

def batch(eng, model_tag, tag, task, query, n):
    imgs = [get_img(tag) for _ in range(n)]
    t0 = time.perf_counter()
    res = eng.predict_batch(imgs, task=task, query=query)
    bt = time.perf_counter() - t0
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for im in imgs:
        eng.predict(im, task=task, query=query)
    st = time.perf_counter() - t0
    row = {"model": model_tag, "kind": "batch_vs_sequential", "task": task, "n": n,
           "batch_s": round(bt, 2), "per_img_batch_s": round(bt / n, 3),
           "sequential_s": round(st, 2), "per_img_seq_s": round(st / n, 3),
           "peak_batch_MiB": mem()}
    emit(row)
    return row

out = []
engs = {}
for m in MODELS:
    tag = m.split("/")[-1]
    t0 = time.perf_counter()
    engs[tag] = WeVisOneVL(model_name_or_path=m)
    load_s = round(time.perf_counter() - t0, 2)
    emit({"model": tag, "kind": "load", "load_s": load_s, "peak_MiB": mem()})
    out.extend(bench(engs[tag], tag, CASES, REPS))
    q_items = [("dogs.jpg", "detection", "dogs\npeople"),
               ("dogs.jpg", "point", "dogs"),
               ("chess.jpg", "count_box", "white chess pieces"),
               ("boys.jpg", "referring", "boys holding microphone"),
               ("coffee_sign.jpg", "ocr", ""),
               ("livingroom.jpg", "point", "sofa"),
               ("dogs.jpg", "segmentation", "dogs"),
               ("seabreeze.jpg", "matting", "person"),
               ("dogs.jpg", "detection", "dogs"),
               ("boys.jpg", "point", "boys"),
               ("livingroom.jpg", "count_box", "chairs"),
               ("dogs.jpg", "count_point", "dogs")]
    out.append(queue(engs[tag], tag, q_items))
    out.append(batch(engs[tag], tag, "dogs.jpg", "detection", "dogs", 8))

tags = list(engs)
if len(tags) >= 2:
    t2, t4 = tags[0], tags[1]
    torch.cuda.reset_peak_memory_stats()
    engs[t2].predict(get_img("dogs.jpg"), task="segmentation", query="dogs")
    emit({"model": t2, "kind": "co_resident", "peak_MiB": mem()})
    engs[t4].predict(get_img("dogs.jpg"), task="segmentation", query="dogs")
    emit({"model": t4, "kind": "co_resident", "peak_MiB": mem()})
    emit({"kind": "co_resident_summary", "models": tags,
          "note": "single-process combined allocator peak after both engines loaded"})

emit({"done": MODELS, "reps": REPS})
