"""Score arm prediction files against coco/subset.jsonl (box F1 @0.5 and mean 0.5:0.95, rejection)."""
import json, re, sys

REF = re.compile(r"<ref>(.*?)</ref>\s*<(box|point)>(.*?)</(?:box|point)>", re.S | re.I)
NUM = re.compile(r"<(\d{1,4})>")
gt = {r["id"]: r for r in (json.loads(l) for l in open("coco/subset.jsonl"))}


def parse(text, w, h):
    out = {}
    for label, _, body in REF.findall(text):
        nums = [int(x) for x in NUM.findall(body)]
        boxes = [[nums[i] / 1000 * w, nums[i + 1] / 1000 * h, nums[i + 2] / 1000 * w, nums[i + 3] / 1000 * h]
                 for i in range(0, len(nums) - 3, 4)]
        out.setdefault(label.strip().lower(), []).extend(boxes)
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def match(pred, true, thr):
    used, tp = set(), 0
    for p in pred:
        best, bj = thr, None
        for j, t in enumerate(true):
            if j not in used and iou(p, t) >= best:
                best, bj = iou(p, t), j
        if bj is not None:
            used.add(bj)
            tp += 1
    return tp, len(pred) - tp, len(true) - tp


def score(path):
    rows = [json.loads(l) for l in open(path)]
    thrs = [0.5 + 0.05 * i for i in range(10)]
    tot = {t: [0, 0, 0] for t in thrs}
    rej, n_ref, lat_det, lat_ref, parse_fail = 0, 0, [], [], 0
    for r in rows:
        g = gt[r["id"]]
        pred = parse(r["response"], g["w"], g["h"])
        if r["task"] == "detection":
            lat_det.append(r["latency_s"])
            if not REF.search(r["response"]):
                parse_fail += 1
            for label in g["present"]:
                true = [x["xyxy"] for x in g["gt"] if x["label"] == label]
                p = pred.get(label.lower(), [])
                for t in thrs:
                    a = match(p, true, t)
                    for k in range(3):
                        tot[t][k] += a[k]
        else:
            lat_ref.append(r["latency_s"])
            n_ref += 1
            boxes = sum(len(v) for v in pred.values())
            if boxes == 0:
                rej += 1
    def f1(t):
        tp, fp, fn = tot[t]
        return 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    med = lambda xs: round(sorted(xs)[len(xs) // 2], 3)
    p95 = lambda xs: round(sorted(xs)[int(0.95 * (len(xs) - 1))], 3)
    return {"arm": path, "n_img": len(lat_det), "F1@0.5": round(100 * f1(0.5), 1),
            "F1_mean": round(100 * sum(f1(t) for t in thrs) / len(thrs), 1),
            "tp_fp_fn@0.5": tot[0.5], "rejection_rate": round(100 * rej / max(n_ref, 1), 1),
            "det_p50_s": med(lat_det), "det_p95_s": p95(lat_det), "ref_p50_s": med(lat_ref),
            "det_no_parse": parse_fail}


for p in sys.argv[1:]:
    print(json.dumps(score(p)))
