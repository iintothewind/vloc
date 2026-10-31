"""Build a deterministic 60-image COCO val2017 subset with GT boxes + one absent category per image."""
import json, os, random, urllib.request, zipfile

ROOT = "coco"
os.makedirs(f"{ROOT}/img", exist_ok=True)
ann_zip = f"{ROOT}/annotations_trainval2017.zip"
ann_json = f"{ROOT}/instances_val2017.json"
if not os.path.exists(ann_json):
    if not os.path.exists(ann_zip):
        urllib.request.urlretrieve(
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip", ann_zip)
    with zipfile.ZipFile(ann_zip) as z, open(ann_json, "wb") as f:
        f.write(z.read("annotations/instances_val2017.json"))

d = json.load(open(ann_json))
cats = {c["id"]: c["name"] for c in d["categories"]}
by_img = {}
for a in d["annotations"]:
    by_img.setdefault(a["image_id"], []).append(a)
imgs = {i["id"]: i for i in d["images"]}

rng = random.Random(20261030)
ids = sorted(by_img)
rng.shuffle(ids)
out = []
for iid in ids:
    anns = by_img[iid]
    if any(a["iscrowd"] for a in anns) or not (2 <= len(anns) <= 20):
        continue
    present = sorted({cats[a["category_id"]] for a in anns})
    if len(present) > 4:
        continue
    absent = rng.choice(sorted(set(cats.values()) - set(present)))
    info = imgs[iid]
    path = f"{ROOT}/img/{info['file_name']}"
    if not os.path.exists(path):
        urllib.request.urlretrieve(info["coco_url"], path)
    gt = [{"label": cats[a["category_id"]],
           "xyxy": [a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]}
          for a in anns]
    out.append({"id": iid, "path": path, "w": info["width"], "h": info["height"],
                "present": present, "absent": absent, "gt": gt})
    if len(out) == 60:
        break

with open(f"{ROOT}/subset.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print(len(out), sum(len(r["gt"]) for r in out), "boxes")
