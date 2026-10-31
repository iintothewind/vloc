"""Run one arm over coco/subset.jsonl and write raw predictions.

usage: cmp_arms.py native <model_dir> <out.jsonl> [min_tokens max_tokens]
       cmp_arms.py llama  <http://127.0.0.1:PORT> <out.jsonl>
Each image: detection over GT-present labels + referring on one absent label.
One visual token = 32x32 px, so token budgets map to min/max_pixels = tokens * 1024.
"""
import base64, io, json, sys, time, urllib.request
from PIL import Image
from wevisone_vl.prompts import build_prompt

KIND, TARGET, OUT = sys.argv[1:4]
rows = [json.loads(l) for l in open("coco/subset.jsonl")]

if KIND == "native":
    import torch
    from wevisone_vl import WeVisOneVL
    budget = {}
    if len(sys.argv) > 5:
        budget = {"min_pixels": int(sys.argv[4]) * 1024, "max_pixels": int(sys.argv[5]) * 1024}
    eng = WeVisOneVL(TARGET, **budget)

    def infer(img, prompt):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        text = eng.generate_text(img, prompt)
        return text, time.perf_counter() - t0
else:
    def infer(img, prompt):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        body = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {
                        "url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}},
                    {"type": "text", "text": prompt}]}],
            "temperature": 0, "max_tokens": 2048, "repeat_penalty": 1.05, "cache_prompt": False,
        }
        req = urllib.request.Request(TARGET + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t0 = time.perf_counter()
        resp = json.load(urllib.request.urlopen(req, timeout=600))
        return resp["choices"][0]["message"]["content"], time.perf_counter() - t0

warm = Image.open(rows[0]["path"]).convert("RGB")
infer(warm, build_prompt("detection", rows[0]["present"]))
with open(OUT, "w") as f:
    for r in rows:
        img = Image.open(r["path"]).convert("RGB")
        for task, query in (("detection", r["present"]), ("referring", r["absent"])):
            prompt = build_prompt(task, query)
            text, dt = infer(img, prompt)
            f.write(json.dumps({"id": r["id"], "task": task, "query": query,
                                "response": text, "latency_s": round(dt, 3)}) + "\n")
            f.flush()
print("done", OUT)
