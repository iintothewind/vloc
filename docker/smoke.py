"""In-container smoke: load engine from $WEVISONE_MODEL, run one box task and one mask task."""
import json, os, time

import numpy as np
import torch
from PIL import Image
from wevisone_vl import WeVisOneVL

model = os.environ["WEVISONE_MODEL"]
image = Image.open(os.environ.get("SMOKE_IMAGE", "/smoke/dogs.jpg")).convert("RGB")

t0 = time.perf_counter()
eng = WeVisOneVL(model)
load_s = time.perf_counter() - t0
rows = {"model": model, "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0), "load_s": round(load_s, 1)}
for task in ("detection", "segmentation"):
    eng.predict(image, task=task, query="dogs")
    torch.cuda.reset_peak_memory_stats()
    r = eng.predict(image, task=task, query="dogs")
    rows[task] = {"latency_s": round(r.latency_s, 3), "response": r.response[:120],
                  "n_masks": len(r.masks),
                  "mask_mean": [round(float(np.mean(m)), 4) for m in r.masks],
                  "peak_MiB": round(torch.cuda.max_memory_allocated() / 2**20)}
print(json.dumps(rows, ensure_ascii=False))
