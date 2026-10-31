# Adapted from Tencent/WeVisOne-VL demo/app.py @9a6bcd8 (Apache-2.0)
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vloc.schemas import RenderStyle

PALETTE: list[tuple[int, int, int]] = [
    (255, 92, 124),
    (47, 212, 123),
    (64, 156, 255),
    (255, 184, 48),
    (176, 92, 255),
    (0, 204, 196),
    (255, 112, 67),
    (120, 200, 80),
]
GREEN_RGB = (0.0, 214.0, 90.0)
BOX_FILL_ALPHA = 38
FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)
MATTING_TASKS = {"matting", "matting_box"}
PIXEL_TASKS = {"segmentation", "matting", "matting_box"}


@dataclass
class RenderItem:
    kind: str  # "box" | "point"
    coords: list[float]  # canvas pixels: xyxy for box, xy for point
    mask: np.ndarray | None = None  # float32 0..1 at canvas size


@dataclass
class RenderGroup:
    label: str
    absent: bool
    items: list[RenderItem] = field(default_factory=list)


def color_for(group_index: int) -> tuple[int, int, int]:
    return PALETTE[group_index % len(PALETTE)]


@functools.lru_cache(maxsize=32)
def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    size = max(8, int(size))
    for path in FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def display_scale(image: Image.Image, target_w: float = 800.0) -> float:
    longest = max(int(image.width), int(image.height), 1)
    return max(0.75, min(longest / target_w, 5.0))


def contrast_ink(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    y = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    return (255, 255, 255) if y < 186 else (20, 24, 32)


def draw_tag(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
    *,
    above: bool = True,
) -> None:
    if not text:
        return
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x = max(5, int(round(th * 0.5)))
    pad_y = max(3, int(round(th * 0.3)))
    box_h = th + pad_y * 2
    box_w = tw + pad_x * 2
    top = y - box_h if above and y > box_h + 1 else y
    draw.rounded_rectangle([x, top, x + box_w, top + box_h], radius=max(3, box_h // 3), fill=color)
    draw.text((x + pad_x - bbox[0], top + pad_y - bbox[1]), text, fill=contrast_ink(color), font=font)


def draw_point(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    color: tuple[int, int, int],
    tag: str,
    font: ImageFont.ImageFont,
    scale: float,
    line_width: int,
    show_labels: bool,
) -> None:
    r = max(8, int(round((8 + max(line_width, 1)) * scale)))
    ring = max(2, int(round(2.5 * scale)))
    draw.ellipse([x - r - ring, y - r - ring, x + r + ring, y + r + ring], fill="white", outline="white")
    draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=color)
    inner = max(3, r // 3)
    draw.ellipse([x - inner, y - inner, x + inner, y + inner], fill="white", outline="white")
    if show_labels and tag:
        draw_tag(draw, tag, x + r + int(4 * scale), y - r, color, font, above=True)


def effective_alpha(groups: list[RenderGroup], task: str, size: tuple[int, int], threshold: float) -> np.ndarray:
    """Per-pixel max over instance masks; segmentation is binarized at ``threshold``."""
    w, h = size
    combined = np.zeros((h, w), dtype=np.float32)
    for group in groups:
        for item in group.items:
            if item.mask is not None:
                combined = np.maximum(combined, np.clip(item.mask, 0.0, 1.0))
    if task in MATTING_TASKS:
        return combined
    return (combined >= threshold).astype(np.float32)


def overlay_masks(image: Image.Image, groups: list[RenderGroup], task: str, style: RenderStyle) -> Image.Image:
    alpha = float(style.mask_alpha)
    if alpha <= 0:
        return image.copy()
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    painted = False
    for gi, group in enumerate(groups):
        color = np.array(color_for(gi), dtype=np.float32).reshape(1, 1, 3)
        for item in group.items:
            if item.mask is None:
                continue
            mask = np.clip(item.mask, 0.0, 1.0)
            if task in MATTING_TASKS:
                keep = mask > 0.0
                weight = (mask * alpha)[..., None]
            else:
                keep = mask >= float(style.mask_threshold)
                weight = (keep.astype(np.float32) * alpha)[..., None]
            if not np.any(keep):
                continue
            arr = np.where(keep[..., None], arr * (1.0 - weight) + color * weight, arr)
            painted = True
    if not painted:
        return image.copy()
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def render_overlay(image: Image.Image, groups: list[RenderGroup], task: str, style: RenderStyle) -> Image.Image:
    is_pixel = task in PIXEL_TASKS
    vis = image.convert("RGB")
    if is_pixel and style.show_masks:
        vis = overlay_masks(vis, groups, task, style)
    else:
        vis = vis.copy()

    drawn = [(gi, g) for gi, g in enumerate(groups) if not g.absent and g.items]
    if not drawn:
        return vis

    if style.show_boxes and not is_pixel:
        fill_layer = Image.new("RGBA", vis.size, (0, 0, 0, 0))
        fill_draw = ImageDraw.Draw(fill_layer)
        for gi, group in drawn:
            color = color_for(gi)
            for item in group.items:
                if item.kind == "box":
                    x0, y0, x1, y1 = (int(c) for c in item.coords)
                    if x1 > x0 and y1 > y0:
                        fill_draw.rectangle([x0, y0, x1, y1], fill=(*color, BOX_FILL_ALPHA))
        vis = Image.alpha_composite(vis.convert("RGBA"), fill_layer).convert("RGB")

    draw = ImageDraw.Draw(vis)
    scale = display_scale(vis)
    font = load_font(max(11, int(round(style.font_size * scale))))
    lw = max(1, int(round(style.line_width * scale)))
    for gi, group in drawn:
        color = color_for(gi)
        tag = group.label.strip() or "object"
        for item in group.items:
            if item.kind == "box":
                x0, y0, x1, y1 = (int(c) for c in item.coords)
                if x1 <= x0 or y1 <= y0:
                    continue
                if style.show_boxes:
                    draw.rectangle([x0, y0, x1, y1], outline=color, width=lw)
                    if style.show_labels:
                        draw_tag(draw, tag, x0, y0, color, font)
            elif item.kind == "point":
                x, y = int(item.coords[0]), int(item.coords[1])
                draw_point(draw, x, y, color, tag, font, scale, style.line_width, style.show_labels)
    return vis


def render_views(
    image: Image.Image,
    groups: list[RenderGroup],
    task: str,
    style: RenderStyle,
    views: list[str],
) -> list[tuple[str, Image.Image]]:
    """Render each requested view on ``image`` (the processed canvas).

    Callers validate that alpha/green_screen/cutout are only requested for pixel tasks.
    """
    out: list[tuple[str, Image.Image]] = []
    alpha: np.ndarray | None = None
    for view in dict.fromkeys(views):
        if view == "overlay":
            out.append((view, render_overlay(image, groups, task, style)))
            continue
        if alpha is None:
            alpha = effective_alpha(groups, task, image.size, float(style.mask_threshold))
        alpha_u8 = Image.fromarray(np.clip(alpha * 255.0, 0, 255).astype(np.uint8))
        if view == "alpha":
            out.append((view, alpha_u8))
        elif view == "green_screen":
            fg = np.asarray(image.convert("RGB"), dtype=np.float32)
            a = alpha[..., None]
            comp = fg * a + np.array(GREEN_RGB, dtype=np.float32) * (1.0 - a)
            out.append((view, Image.fromarray(np.clip(comp, 0, 255).astype(np.uint8))))
        elif view == "cutout":
            rgba = image.convert("RGBA")
            rgba.putalpha(alpha_u8)
            out.append((view, rgba))
    return out


def encode(name: str, img: Image.Image, fmt: str) -> tuple[str, str, bytes]:
    """Return (file name, media type, bytes). alpha and cutout are always PNG."""
    if name in ("alpha", "cutout") or fmt == "png":
        buf = BytesIO()
        img.save(buf, format="PNG")
        return f"{name}.png", "image/png", buf.getvalue()
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return f"{name}.jpg", "image/jpeg", buf.getvalue()
