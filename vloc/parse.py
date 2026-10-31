from __future__ import annotations

import re
from dataclasses import dataclass, field

REF_RE = re.compile(
    r"<ref>(.*?)</ref>\s*<(box|point|polygon)>(.*?)(?:</(box|point|polygon)>|(?=<ref>)|$)",
    re.S | re.I,
)
COORD_RE = re.compile(r"<(\d{1,4})>")
OBJ_RE = re.compile(r"<OBJ>", re.I)
COUNT_RE = re.compile(r"Total number:\s*(\d+)", re.I)
END_RE = re.compile(r"<\|im_end\|>|<\|endoftext\|>")


@dataclass
class ParsedInstance:
    kind: str  # "box" | "point"
    norm: list[float]
    has_obj: bool


@dataclass
class ParsedGroup:
    label: str
    absent: bool
    instances: list[ParsedInstance] = field(default_factory=list)


@dataclass
class Parsed:
    groups: list[ParsedGroup]
    declared_total: int | None
    warnings: list[str]


def parse_response(text: str) -> Parsed:
    text = END_RE.split(text or "", maxsplit=1)[0]
    groups: dict[str, ParsedGroup] = {}
    warnings: list[str] = []

    def warn(msg: str) -> None:
        if msg not in warnings:
            warnings.append(msg)

    for m in REF_RE.finditer(text):
        label = m.group(1).strip() or "object"
        opening = m.group(2).lower()
        closing = m.group(4)
        if closing is None:
            warn(f"unterminated <{opening}> for label '{label}'")
        elif closing.lower() != opening:
            warn(f"malformed closing tag for label '{label}': {opening} closed by {closing.lower()}")
        body = m.group(3).strip()
        group = groups.get(label)
        if body.lower() == "none":
            if group is None:
                groups[label] = ParsedGroup(label, absent=True)
                warn(f"label '{label}' absent")
            continue
        if group is None:
            group = groups[label] = ParsedGroup(label, absent=False)
        group.absent = False
        if opening == "polygon":
            warn("polygon output ignored")
            continue
        step = 2 if opening == "point" else 4
        for segment in body.split(","):
            values = [int(v) for v in COORD_RE.findall(segment)]
            has_obj = bool(OBJ_RE.search(segment))
            for off in range(0, len(values) - step + 1, step):
                chunk = values[off : off + step]
                if any(v > 1000 for v in chunk):
                    warn("coordinate out of range dropped")
                    continue
                if step == 4 and (chunk[2] <= chunk[0] or chunk[3] <= chunk[1]):
                    warn("degenerate box dropped")
                    continue
                group.instances.append(ParsedInstance(opening, [v / 1000.0 for v in chunk], has_obj))

    totals = COUNT_RE.findall(text)
    return Parsed(list(groups.values()), int(totals[-1]) if totals else None, warnings)
