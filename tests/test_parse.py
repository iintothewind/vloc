from tests.fake_engine import ABSENT, COUNT_BOX, MALFORMED
from vloc.parse import parse_response


def test_absent_label_kept():
    p = parse_response(ABSENT)
    assert [(g.label, g.absent, len(g.instances)) for g in p.groups] == [("dogs", False, 1), ("people", True, 0)]
    assert "label 'people' absent" in p.warnings


def test_mismatched_closing_tag_is_absent_with_warning():
    p = parse_response(MALFORMED)
    assert len(p.groups) == 1 and p.groups[0].absent
    assert any(w.startswith("malformed closing tag for label 'chairs'") for w in p.warnings)


def test_missing_closing_tag_does_not_swallow_next_ref():
    p = parse_response("<ref>a</ref><box><1><2><3><4><ref>b</ref><box>none</box>")
    assert [(g.label, g.absent) for g in p.groups] == [("a", False), ("b", True)]
    assert p.groups[0].instances[0].norm == [0.001, 0.002, 0.003, 0.004]
    assert "unterminated <box> for label 'a'" in p.warnings


def test_degenerate_and_out_of_range_boxes_dropped():
    p = parse_response("<ref>x</ref><box><500><10><400><20>,<1><2><3><1001>,<1><2><3><4></box>")
    assert len(p.groups[0].instances) == 1
    assert "degenerate box dropped" in p.warnings
    assert "coordinate out of range dropped" in p.warnings


def test_declared_total():
    p = parse_response(COUNT_BOX)
    assert p.declared_total == 3
    assert len(p.groups[0].instances) == 2


def test_end_token_tail_ignored():
    p = parse_response("<ref>a</ref><box><1><2><3><4></box><|im_end|><ref>b</ref><box><1><2><3><4></box>")
    assert [g.label for g in p.groups] == ["a"]


def test_points_and_obj_flags():
    p = parse_response("<ref>p</ref><point><100><200>,<300><400></point><ref>s</ref><box><1><2><3><4><OBJ></box>")
    assert [i.norm for i in p.groups[0].instances] == [[0.1, 0.2], [0.3, 0.4]]
    assert p.groups[1].instances[0].has_obj and not p.groups[0].instances[0].has_obj


def test_repeated_label_merges():
    p = parse_response("<ref>a</ref><box>none</box><ref>a</ref><box><1><2><3><4></box>")
    assert len(p.groups) == 1 and not p.groups[0].absent and len(p.groups[0].instances) == 1
