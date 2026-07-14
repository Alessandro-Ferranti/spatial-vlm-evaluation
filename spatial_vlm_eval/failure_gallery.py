from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw


def load_dataset(path: Path) -> dict[str, dict]:
    samples = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                sample = json.loads(line)
                samples[sample["id"]] = sample
    return samples


def draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, width: int, line_height: int = 16) -> int:
    x, y = xy
    for line in textwrap.wrap(text, width=width):
        draw.text((x, y), line, fill="black")
        y += line_height
    return y


def build_gallery(dataset_path: Path, failures_path: Path, output_path: Path, limit: int = 12) -> None:
    samples = load_dataset(dataset_path)
    failures = pd.read_csv(failures_path).head(limit)

    tile_w = 420
    tile_h = 430
    image_h = 260
    cols = 2
    rows = max(1, (len(failures) + cols - 1) // cols)
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), "white")
    draw = ImageDraw.Draw(canvas)

    for idx, row in failures.iterrows():
        sample = samples[row["id"]]
        col = idx % cols
        row_idx = idx // cols
        x0 = col * tile_w
        y0 = row_idx * tile_h

        image = Image.open(sample["image_paths"][0]).convert("RGB")
        image.thumbnail((tile_w - 24, image_h))
        canvas.paste(image, (x0 + 12, y0 + 12))

        y = y0 + image_h + 24
        draw.text((x0 + 12, y), f'{row["task"]} | {row["id"]}', fill="black")
        y += 20
        draw.text((x0 + 12, y), f'GT: {row["ground_truth"]} | Pred: {row["prediction"]}', fill="black")
        y += 20
        raw = str(row["raw_prediction"]).replace("\n", " ")
        y = draw_wrapped(draw, (x0 + 12, y), f"Raw: {raw}", width=52)
        question = sample["question"].replace("\n", " ")
        draw_wrapped(draw, (x0 + 12, y + 6), f"Q: {question}", width=52)

        draw.rectangle((x0, y0, x0 + tile_w - 1, y0 + tile_h - 1), outline=(220, 220, 220))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a visual gallery from VLM failure cases.")
    parser.add_argument("--dataset", required=True, help="Path to dataset.jsonl")
    parser.add_argument("--failures", required=True, help="Path to failures.csv")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    build_gallery(Path(args.dataset), Path(args.failures), Path(args.output), limit=args.limit)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
