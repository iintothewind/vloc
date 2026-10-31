import pytest

from vloc.errors import ApiError
from vloc.queries import build, normalize_task


def _err(fn, *args) -> ApiError:
    with pytest.raises(ApiError) as e:
        fn(*args)
    assert e.value.status == 422 and e.value.code == "invalid_request"
    return e.value


def test_too_many_labels():
    _err(build, "detection", [f"l{i}" for i in range(21)], None, (100, 100), 1.0)
    assert build("detection", [f"l{i}" for i in range(20)], None, (100, 100), 1.0).count(". ") == 19


def test_labels_deduped_and_sanitized():
    p = build("detection", "dog\n dog \ncat<OBJ>\n\n", None, (100, 100), 1.0)
    assert p == "Locate all the instances that match the following description: dog. catOBJ."


def test_count_requires_single_label():
    _err(build, "count_box", ["dog", "cat"], None, (100, 100), 1.0)


def test_ocr_rejects_query():
    _err(build, "ocr", "hello", None, (100, 100), 1.0)
    assert build("ocr", None, None, (100, 100), 1.0) == "detect all the word in the box format"


def test_phrase_length():
    _err(build, "referring", "x" * 201, None, (100, 100), 1.0)
    _err(build, "referring", "  <> ", None, (100, 100), 1.0)


def test_visual_prompt_scaled_and_clipped():
    p = build("visual_prompt", None, [[100, 200, 300, 400]], (500, 500), 0.5)
    # scaled to (50,100,150,200) on a 500x500 processed image -> <100><200><300><400>
    assert p.endswith("<ref>object 1</ref><box><100><200><300><400></box>")
    _err(build, "visual_prompt", None, [[2000, 2000, 3000, 3000]], (500, 500), 0.5)
    _err(build, "visual_prompt", None, None, (500, 500), 0.5)
    _err(build, "visual_prompt", "dog", [[0, 0, 10, 10]], (500, 500), 1.0)


def test_reference_boxes_only_for_visual_prompt():
    _err(build, "detection", "dog", [[0, 0, 10, 10]], (100, 100), 1.0)


def test_unknown_task():
    e = _err(normalize_task, "segment")
    assert "choose from" in e.message
    assert normalize_task(" Detection ") == "detection"
