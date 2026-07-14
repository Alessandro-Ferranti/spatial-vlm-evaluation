from __future__ import annotations

import math
import re

VALID_RELATIONS = ["left", "right", "above", "below"]
VALID_YES_NO = ["yes", "no"]


def normalize_relation_answer(answer: str) -> str:
    text = answer.lower().strip()
    phrase_map = {
        "to the left": "left",
        "left of": "left",
        "on the left": "left",
        "to the right": "right",
        "right of": "right",
        "on the right": "right",
        "on top of": "above",
        "over": "above",
        "under": "below",
        "beneath": "below",
    }
    for phrase, canonical in phrase_map.items():
        if phrase in text:
            return canonical

    tokens = re.sub(r"[^a-z\s]", " ", text).split()
    for relation in VALID_RELATIONS:
        if relation in tokens:
            return relation
    return " ".join(tokens)


def normalize_choice_answer(answer: str, choices: list[str]) -> str:
    text = re.sub(r"[^a-z0-9_\-\s]", " ", answer.lower()).strip()
    tokens = text.split()
    for choice in choices:
        if choice.lower() in tokens or choice.lower() == text:
            return choice
    for choice in choices:
        if choice.lower() in text:
            return choice
    return text


def normalize_yes_no(answer: str) -> str:
    return normalize_choice_answer(answer, VALID_YES_NO)


def normalize_bbox(bbox_pixels: tuple[int, int, int, int], width: int, height: int) -> list[float]:
    x_min, y_min, x_max, y_max = bbox_pixels
    return [x_min / width, y_min / height, x_max / width, y_max / height]


def parse_bbox_from_text(text: str) -> list[float] | None:
    match = re.search(
        r"\[\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*"
        r"(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*\]",
        text,
    )
    if match is None:
        return None
    bbox = [float(value) for value in match.groups()]
    if any(not math.isfinite(value) for value in bbox):
        return None
    return bbox


def bbox_iou(predicted: list[float], target: list[float]) -> float:
    px1, py1, px2, py2 = predicted
    tx1, ty1, tx2, ty2 = target
    inter_x1 = max(px1, tx1)
    inter_y1 = max(py1, ty1)
    inter_x2 = min(px2, tx2)
    inter_y2 = min(py2, ty2)
    intersection = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    predicted_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    target_area = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)
    union = predicted_area + target_area - intersection
    return 0.0 if union <= 0 else intersection / union
