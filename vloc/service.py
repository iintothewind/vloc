from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass

import numpy as np
from PIL import Image
from wevisone_vl.prompts import build_prompt

from vloc import masks as maskops
from vloc import queries
from vloc.artifacts import ArtifactStore
from vloc.config import Settings
from vloc.engine import EngineOutput, EngineRunner
from vloc.errors import ApiError
from vloc.images import decode_image, fit_to_budget, load_image
from vloc.parse import Parsed, ParsedGroup, ParsedInstance, parse_response
from vloc.render import RenderGroup, RenderItem, encode, render_views
from vloc.schemas import (
    Artifact,
    Group,
    ImageInput,
    Instance,
    MattingBoxRequest,
    Options,
    PerceiveRequest,
    PerceiveResponse,
)


@dataclass
class _Prepared:
    request_id: str
    t0: float
    original: tuple[int, int]
    proc: Image.Image
    scale: float


async def _prepare(image: ImageInput, task: str, opts: Options, settings: Settings) -> _Prepared:
    t0 = time.perf_counter()
    if opts.render:
        for view in opts.render_views:
            if view != "overlay" and task not in queries.PIXEL_TASKS:
                raise ApiError(
                    "invalid_request", f"render view '{view}' requires task segmentation, matting or matting_box"
                )
    data = await load_image(image, settings)
    img = await asyncio.to_thread(decode_image, data, settings)
    original = img.size
    proc = await asyncio.to_thread(fit_to_budget, img, settings.max_pixels)
    return _Prepared(uuid.uuid4().hex, t0, original, proc, round(proc.width / original[0], 6))


def _assign_masks(groups: list[ParsedGroup], masks: list[np.ndarray], warnings: list[str]) -> dict[int, np.ndarray]:
    """Map id(instance) -> mask: each ``<OBJ>`` instance takes the next mask in order.

    If no instance carries ``<OBJ>`` but masks exist, assign them to box instances in order.
    """
    flat = [inst for g in groups for inst in g.instances]
    targets = [inst for inst in flat if inst.has_obj]
    if not targets and masks:
        targets = [inst for inst in flat if inst.kind == "box"]
    assigned = {id(inst): m for inst, m in zip(targets, masks)}
    extra = len(masks) - len(assigned)
    if extra > 0:
        warnings.append(f"{extra} masks could not be matched to boxes")
    return assigned


async def _finish(
    *,
    prep: _Prepared,
    task: str,
    prompt: str,
    parsed: Parsed,
    out: EngineOutput,
    queue_ms: int,
    truncated: bool,
    opts: Options,
    runner: EngineRunner,
    store: ArtifactStore,
) -> PerceiveResponse:
    warnings = list(parsed.warnings)
    if truncated:
        warnings.append("response truncated at max_new_tokens")
    W, H = prep.original
    pw, ph = prep.proc.size
    assigned = _assign_masks(parsed.groups, out.masks, warnings)

    groups: list[Group] = []
    render_groups: list[RenderGroup] = []
    mask_index = 0
    for pg in parsed.groups:
        instances: list[Instance] = []
        items: list[RenderItem] = []
        for inst in pg.instances:
            mask = assigned.get(id(inst))
            if inst.kind == "point":
                x, y = inst.norm
                api = Instance(point_norm=inst.norm, point_px=[round(x * W, 1), round(y * H, 1)])
                items.append(RenderItem("point", [x * pw, y * ph], mask))
            else:
                api = Instance(
                    box_norm=inst.norm, box_px=[round(v * s, 1) for v, s in zip(inst.norm, (W, H, W, H))]
                )
                items.append(RenderItem("box", [v * s for v, s in zip(inst.norm, (pw, ph, pw, ph))], mask))
            if mask is not None:
                api.mask_mean = maskops.mask_mean(mask)
                if inst.kind == "box":
                    api.mask_outside_box_fraction = maskops.outside_box_fraction(mask, inst.norm)
                if opts.mask_delivery != "none":
                    size = prep.original if opts.mask_resolution == "original" else (pw, ph)
                    png = await asyncio.to_thread(maskops.to_png, mask, task, size)
                    api.mask = store.put(
                        prep.request_id,
                        f"mask_{mask_index:03d}.png",
                        png,
                        "image/png",
                        size[0],
                        size[1],
                        opts.mask_delivery,
                    )
                mask_index += 1
            instances.append(api)
        groups.append(Group(label=pg.label, absent=pg.absent, instances=instances))
        render_groups.append(RenderGroup(pg.label, pg.absent, items))

    renders: list[Artifact] = []
    render_ms = 0
    if opts.render:
        t_r = time.perf_counter()
        views = await asyncio.to_thread(
            render_views, prep.proc, render_groups, task, opts.render_style, list(opts.render_views)
        )
        for name, img in views:
            file_name, media_type, data = await asyncio.to_thread(encode, name, img, opts.render_format)
            renders.append(
                store.put(prep.request_id, file_name, data, media_type, img.width, img.height, opts.render_delivery)
            )
        render_ms = round((time.perf_counter() - t_r) * 1000)

    count = sum(len(g.instances) for g in groups)
    total_consistent = None
    if task in queries.COUNT_TASKS and parsed.declared_total is not None:
        total_consistent = parsed.declared_total == count

    return PerceiveResponse(
        request_id=prep.request_id,
        task=task,
        mode=queries.MODES[task],
        model=runner.engine.name if runner.engine is not None else "",
        prompt=prompt,
        image_size=[W, H],
        processed_size=[pw, ph],
        scale=prep.scale,
        groups=groups,
        declared_total=parsed.declared_total,
        count_by_items=count,
        total_consistent=total_consistent,
        truncated=truncated,
        warnings=warnings,
        raw_response=out.response,
        renders=renders,
        latency_ms={
            "queue": queue_ms,
            "model": round(out.latency_s * 1000),
            "render": render_ms,
            "total": round((time.perf_counter() - prep.t0) * 1000),
        },
    )


async def perceive(
    req: PerceiveRequest, runner: EngineRunner, store: ArtifactStore, settings: Settings
) -> PerceiveResponse:
    task = queries.normalize_task(req.task)
    opts = req.options
    prep = await _prepare(req.image, task, opts, settings)
    prompt = queries.build(task, req.query, req.reference_boxes, prep.proc.size, prep.scale)
    proc = prep.proc
    out, queue_ms = await runner.run(lambda e: e.generate(proc, prompt, opts.max_new_tokens))
    parsed = parse_response(out.response)
    return await _finish(
        prep=prep,
        task=task,
        prompt=prompt,
        parsed=parsed,
        out=out,
        queue_ms=queue_ms,
        truncated=out.response_tokens >= opts.max_new_tokens,
        opts=opts,
        runner=runner,
        store=store,
    )


async def matting_box(
    req: MattingBoxRequest, runner: EngineRunner, store: ArtifactStore, settings: Settings
) -> PerceiveResponse:
    task = "matting_box"
    opts = req.options
    phrase = queries.text(req.phrase, queries.MAX_PHRASE_LEN, "phrase")
    prep = await _prepare(req.image, task, opts, settings)
    proc = prep.proc
    pw, ph = proc.size
    x1, y1, x2, y2 = queries.scale_box(req.box, prep.scale, (pw, ph), "box")
    box_norm = [round(x1 / pw, 6), round(y1 / ph, 6), round(x2 / pw, 6), round(y2 / ph, 6)]
    prompt = build_prompt("matting", phrase)
    synth = f"<ref>{phrase}</ref><box>{''.join(f'<{round(v * 1000)}>' for v in box_norm)}<OBJ></box>"
    out, queue_ms = await runner.run(lambda e: e.decode_box(proc, prompt, synth, box_norm))
    parsed = Parsed(
        groups=[ParsedGroup(phrase, absent=False, instances=[ParsedInstance("box", box_norm, True)])],
        declared_total=None,
        warnings=[],
    )
    return await _finish(
        prep=prep,
        task=task,
        prompt=prompt,
        parsed=parsed,
        out=out,
        queue_ms=queue_ms,
        truncated=False,
        opts=opts,
        runner=runner,
        store=store,
    )
