# Spatial VLM Evaluation

A practical evaluation harness for Vision-Language Models on spatial reasoning tasks.

The original notebook is kept as a guided walkthrough. The Python package adds a reusable project structure: procedural scene generation, exact ground truth, task-level metrics, model adapters, and robotics-inspired spatial AI scenarios.

## Why this project is useful

Most VLM demos ask qualitative questions about a single image. This project turns spatial understanding into measurable tasks:

- relative object position: left, right, above, below;
- distractor robustness: target/reference selection with irrelevant objects;
- language grounding: normalized bounding boxes and IoU;
- robot pick-and-place: choose the correct target object;
- navigation clearance: decide whether a corridor/path is blocked;
- tabletop affordances: decide whether an object is reachable without collision.

These tasks are intentionally synthetic. That is a feature: the renderer controls the scene, so labels are exact and failure cases can be inspected visually.

## Quickstart

Create datasets without loading a VLM:

```bash
python -m spatial_vlm_eval.cli generate --output runs/demo --samples-per-task 8
```

Run the deterministic mock adapter to verify the full evaluation pipeline:

```bash
python -m spatial_vlm_eval.cli evaluate --dataset runs/demo/dataset.jsonl --output runs/demo/results --adapter mock
```

Use SmolVLM2 when the optional dependencies and model weights are available:

```bash
pip install -e ".[vlm]"
python -m spatial_vlm_eval.cli evaluate \
  --dataset runs/demo/dataset.jsonl \
  --output runs/smolvlm2/results \
  --adapter smolvlm \
  --model-id HuggingFaceTB/SmolVLM2-2.2B-Instruct
```

## Outputs

The generator writes:

- `dataset.jsonl`: one sample per line with image paths, question, answer, and metadata;
- `images/`: rendered scenes.

The evaluator writes:

- `predictions.csv`: raw and normalized predictions;
- `summary.csv`: accuracy and IoU by task;
- `failures.csv`: cases to inspect first.

## Project layout

```text
spatial_vlm_eval/
  adapters.py      # mock and Hugging Face VLM adapters
  cli.py           # generate/evaluate commands
  evaluation.py    # metrics and result tables
  metrics.py       # answer normalization, bbox parsing, IoU
  schemas.py       # sample/result dataclasses
  scenes.py        # procedural renderers and task generators
```

## Suggested GitHub positioning

This is strongest as a portfolio project if the README shows three concrete runs:

1. a lightweight mock run proving reproducibility;
2. SmolVLM2 results on all tasks;
3. a comparison run with a stronger VLM, especially on grounding and robotics tasks.

Add a few failure-case images to the README once you have real model outputs. That makes the project feel practical rather than just benchmark-shaped.
