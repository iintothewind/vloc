import numpy as np
from PIL import Image, ImageChops

from vloc.render import PALETTE, RenderGroup, RenderItem, render_views
from vloc.schemas import RenderStyle

BASE = Image.new("RGB", (800, 600), (128, 128, 128))
STYLE = RenderStyle()


def _overlay(groups, task="detection"):
    return render_views(BASE, groups, task, STYLE, ["overlay"])[0][1]


def test_box_outline_uses_group_color():
    groups = [
        RenderGroup("a", False, [RenderItem("box", [100, 100, 300, 300])]),
        RenderGroup("b", False, [RenderItem("box", [400, 300, 600, 500])]),
    ]
    img = _overlay(groups)
    assert img.getpixel((101, 200)) == PALETTE[0]
    assert img.getpixel((200, 299)) == PALETTE[0]
    assert img.getpixel((401, 400)) == PALETTE[1]
    # translucent fill inside the box, untouched outside
    assert img.getpixel((200, 200)) != (128, 128, 128)
    assert img.getpixel((700, 50)) == (128, 128, 128)


def test_absent_group_not_drawn():
    img = _overlay([RenderGroup("people", True, [])])
    assert ImageChops.difference(img, BASE).getbbox() is None


def test_colors_deterministic():
    groups = [RenderGroup("dog", False, [RenderItem("box", [10, 10, 200, 200]), RenderItem("box", [300, 300, 500, 500])])]
    a, b = _overlay(groups), _overlay(groups)
    assert ImageChops.difference(a, b).getbbox() is None


def test_matting_views():
    mask = np.zeros((600, 800), np.float32)
    mask[100:200, 100:200] = 1.0
    mask[300:310, 300:310] = 0.5
    groups = [RenderGroup("p", False, [RenderItem("box", [100, 100, 200, 200], mask)])]
    views = dict(render_views(BASE, groups, "matting", STYLE, ["alpha", "green_screen", "cutout"]))
    alpha = np.asarray(views["alpha"])
    assert alpha[150, 150] == 255 and alpha[305, 305] == 127 and alpha[0, 0] == 0
    assert views["green_screen"].getpixel((0, 0)) == (0, 214, 90)
    assert views["cutout"].mode == "RGBA" and views["cutout"].getpixel((150, 150))[3] == 255
    seg = dict(render_views(BASE, groups, "segmentation", STYLE, ["alpha"]))
    assert set(np.unique(np.asarray(seg["alpha"]))) == {0, 255}
