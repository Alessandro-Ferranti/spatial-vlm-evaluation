from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from .adapters import MockAdapter, VLMAdapter
from .metrics import bbox_iou, normalize_choice_answer, normalize_relation_answer, normalize_yes_no, parse_bbox_from_text
from .schemas import Prediction, Sample


def load_jsonl(path: Path) -> list[Sample]:
    samples = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                samples.append(Sample.from_dict(json.loads(line)))
    return samples


def write_jsonl(samples: list[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(sample.to_dict()) + "\n")


def normalize_prediction(sample: Sample, raw_prediction: str) -> tuple[str, bool, float | None]:
    if sample.answer_type == "relation":
        prediction = normalize_relation_answer(raw_prediction)
        return prediction, prediction == sample.answer, None
    if sample.answer_type == "yes_no":
        prediction = normalize_yes_no(raw_prediction)
        return prediction, prediction == sample.answer, None
    if sample.answer_type == "choice":
        choices = sample.metadata.get("choices", [])
        prediction = normalize_choice_answer(raw_prediction, choices)
        return prediction, prediction == sample.answer, None
    if sample.answer_type == "bbox":
        prediction_bbox = parse_bbox_from_text(raw_prediction)
        target_bbox = sample.metadata["bbox"]
        if prediction_bbox is None:
            return "", False, 0.0
        iou = bbox_iou(prediction_bbox, target_bbox)
        return str(prediction_bbox), iou >= 0.5, iou
    raise ValueError(f"Unsupported answer_type: {sample.answer_type}")


def evaluate(samples: list[Sample], adapter: VLMAdapter, max_new_tokens: int = 40) -> pd.DataFrame:
    rows: list[Prediction] = []
    for sample in tqdm(samples, desc="Evaluating"):
        try:
            if isinstance(adapter, MockAdapter):
                adapter.set_next_answer(sample.answer)
            raw = adapter.ask(sample.image_paths, sample.question, max_new_tokens=max_new_tokens)
            prediction, correct, metric_value = normalize_prediction(sample, raw)
            error = None
        except Exception as exc:
            raw = ""
            prediction = ""
            correct = False
            metric_value = None
            error = repr(exc)
        rows.append(
            Prediction(
                id=sample.id,
                task=sample.task,
                question=sample.question,
                ground_truth=sample.answer,
                raw_prediction=raw,
                prediction=prediction,
                correct=correct,
                metric_value=metric_value,
                error=error,
            )
        )
    return pd.DataFrame([asdict(row) for row in rows])


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    grouped = results.groupby("task", dropna=False)
    summary = grouped["correct"].agg(["mean", "count"]).reset_index()
    summary = summary.rename(columns={"mean": "accuracy", "count": "num_examples"})
    metric_rows = results.dropna(subset=["metric_value"]).groupby("task")["metric_value"].mean()
    summary["mean_metric_value"] = summary["task"].map(metric_rows).astype("float")
    return summary


def save_results(results: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "predictions.csv", index=False)
    summarize(results).to_csv(output_dir / "summary.csv", index=False)
    failures = results[~results["correct"]].copy()
    failures.to_csv(output_dir / "failures.csv", index=False)
