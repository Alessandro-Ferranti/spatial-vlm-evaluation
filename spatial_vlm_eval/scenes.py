from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .metrics import normalize_bbox
from .schemas import Sample

COLORS = ["red", "blue", "green", "orange", "purple"]
SHAPES = ["circle", "square", "triangle"]
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

RELATION_POSITIONS = {
    "left": ((160, 240), (480, 240)),
    "right": ((480, 240), (160, 240)),
    "above": ((320, 130), (320, 350)),
    "below": ((320, 350), (320, 130)),
}


def make_bbox(center_x: int, center_y: int, size: int) -> tuple[int, int, int, int]:
    half = size // 2
    return center_x - half, center_y - half, center_x + half, center_y + half


def draw_shape(draw: ImageDraw.ImageDraw, shape: str, bbox: tuple[int, int, int, int], color: str) -> None:
    if shape == "circle":
        draw.ellipse(bbox, fill=color, outline="black", width=4)
    elif shape == "square":
        draw.rectangle(bbox, fill=color, outline="black", width=4)
    elif shape == "triangle":
        x1, y1, x2, y2 = bbox
        draw.polygon([((x1 + x2) // 2, y1), (x1, y2), (x2, y2)], fill=color, outline="black")
    else:
        raise ValueError(f"Unsupported shape: {shape}")


def _blank() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), "white")
    return image, ImageDraw.Draw(image)


def _object_pair(rng: random.Random) -> tuple[str, str, str, str]:
    color_a, color_b = rng.sample(COLORS, 2)
    shape_a, shape_b = rng.sample(SHAPES, 2)
    return color_a, shape_a, color_b, shape_b


def create_direct_relation_sample(sample_id: int, relation: str, image_dir: Path, rng: random.Random) -> Sample:
    image, draw = _blank()
    color_a, shape_a, color_b, shape_b = _object_pair(rng)
    position_a, position_b = RELATION_POSITIONS[relation]
    center_a = (position_a[0] + rng.randint(-15, 15), position_a[1] + rng.randint(-15, 15))
    center_b = (position_b[0] + rng.randint(-15, 15), position_b[1] + rng.randint(-15, 15))
    bbox_a = make_bbox(*center_a, size=110)
    bbox_b = make_bbox(*center_b, size=110)
    draw_shape(draw, shape_a, bbox_a, color_a)
    draw_shape(draw, shape_b, bbox_b, color_b)
    path = image_dir / "direct_relations" / f"direct_{sample_id:04d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return Sample(
        id=f"direct_{sample_id:04d}",
        task="direct_relation",
        image_paths=[str(path)],
        question=f"Where is the {color_a} {shape_a} relative to the {color_b} {shape_b}? Answer only: left, right, above, or below.",
        answer=relation,
        answer_type="relation",
        metadata={"target": f"{color_a} {shape_a}", "reference": f"{color_b} {shape_b}", "target_bbox": normalize_bbox(bbox_a, IMAGE_WIDTH, IMAGE_HEIGHT)},
    )


def create_distractor_sample(sample_id: int, relation: str, image_dir: Path, rng: random.Random) -> Sample:
    image, draw = _blank()
    color_a, shape_a, color_b, shape_b = _object_pair(rng)
    position_a, position_b = RELATION_POSITIONS[relation]
    draw_shape(draw, shape_a, make_bbox(*position_a, size=95), color_a)
    draw_shape(draw, shape_b, make_bbox(*position_b, size=95), color_b)
    used = {f"{color_a} {shape_a}", f"{color_b} {shape_b}"}
    centers = [(120, 100), (520, 100), (120, 380), (520, 380)]
    rng.shuffle(centers)
    for center in centers[:3]:
        for _ in range(50):
            color = rng.choice(COLORS)
            shape = rng.choice(SHAPES)
            if f"{color} {shape}" not in used:
                used.add(f"{color} {shape}")
                draw_shape(draw, shape, make_bbox(*center, size=65), color)
                break
    path = image_dir / "distractors" / f"distractor_{sample_id:04d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return Sample(
        id=f"distractor_{sample_id:04d}",
        task="distractor_relation",
        image_paths=[str(path)],
        question=f"Ignore unrelated objects. Where is the {color_a} {shape_a} relative to the {color_b} {shape_b}? Answer only: left, right, above, or below.",
        answer=relation,
        answer_type="relation",
        metadata={"target": f"{color_a} {shape_a}", "reference": f"{color_b} {shape_b}"},
    )


def create_robot_pick_sample(sample_id: int, image_dir: Path, rng: random.Random) -> Sample:
    image, draw = _blank()
    table = (70, 80, 570, 410)
    draw.rectangle(table, outline="gray", width=4)
    candidates = [("red", "circle"), ("blue", "square"), ("green", "triangle")]
    rng.shuffle(candidates)
    centers = [(170, 250), (320, 180), (470, 300)]
    target_index = rng.randrange(len(candidates))
    choices = []
    for idx, ((color, shape), center) in enumerate(zip(candidates, centers)):
        draw_shape(draw, shape, make_bbox(*center, size=80), color)
        choices.append(f"object_{idx + 1}")
        draw.text((center[0] - 32, center[1] + 55), choices[-1], fill="black")
    target_color, target_shape = candidates[target_index]
    path = image_dir / "robot_pick" / f"robot_pick_{sample_id:04d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return Sample(
        id=f"robot_pick_{sample_id:04d}",
        task="robot_pick_target",
        image_paths=[str(path)],
        question=f"A robot must pick the {target_color} {target_shape}. Which labeled object should it pick? Answer only one label: {', '.join(choices)}.",
        answer=choices[target_index],
        answer_type="choice",
        metadata={"choices": choices, "target": f"{target_color} {target_shape}"},
    )


def create_navigation_clearance_sample(sample_id: int, image_dir: Path, rng: random.Random) -> Sample:
    image, draw = _blank()
    draw.rectangle((120, 80, 520, 400), outline="black", width=3)
    draw.line((320, 400, 320, 80), fill="green", width=14)
    blocked = rng.choice([True, False])
    if blocked:
        obstacle_bbox = (260, 205, 380, 275)
    else:
        obstacle_bbox = rng.choice([(150, 160, 230, 240), (410, 240, 490, 320)])
    draw.rectangle(obstacle_bbox, fill="red", outline="black", width=4)
    path = image_dir / "navigation" / f"navigation_{sample_id:04d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return Sample(
        id=f"navigation_{sample_id:04d}",
        task="navigation_clearance",
        image_paths=[str(path)],
        question="The green line is the planned robot path and the red rectangle is an obstacle. Is the path blocked? Answer only yes or no.",
        answer="yes" if blocked else "no",
        answer_type="yes_no",
        metadata={"blocked": blocked},
    )


def create_grounding_sample(sample_id: int, image_dir: Path, rng: random.Random) -> Sample:
    image, draw = _blank()
    color, shape = rng.choice(COLORS), rng.choice(SHAPES)
    center = (rng.randint(150, 490), rng.randint(130, 350))
    bbox = make_bbox(*center, size=100)
    draw_shape(draw, shape, bbox, color)
    path = image_dir / "grounding" / f"grounding_{sample_id:04d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return Sample(
        id=f"grounding_{sample_id:04d}",
        task="language_grounding",
        image_paths=[str(path)],
        question=f"Return the normalized bounding box of the {color} {shape}. Use exactly [x_min, y_min, x_max, y_max].",
        answer=str(normalize_bbox(bbox, IMAGE_WIDTH, IMAGE_HEIGHT)),
        answer_type="bbox",
        metadata={"target": f"{color} {shape}", "bbox": normalize_bbox(bbox, IMAGE_WIDTH, IMAGE_HEIGHT)},
    )


def generate_dataset(output_dir: Path, samples_per_task: int, seed: int = 42) -> list[Sample]:
    rng = random.Random(seed)
    image_dir = output_dir / "images"
    relations = list(RELATION_POSITIONS)
    samples: list[Sample] = []
    for idx in range(samples_per_task):
        relation = relations[idx % len(relations)]
        samples.append(create_direct_relation_sample(idx, relation, image_dir, rng))
        samples.append(create_distractor_sample(idx, relation, image_dir, rng))
        samples.append(create_robot_pick_sample(idx, image_dir, rng))
        samples.append(create_navigation_clearance_sample(idx, image_dir, rng))
        samples.append(create_grounding_sample(idx, image_dir, rng))
    return samples
