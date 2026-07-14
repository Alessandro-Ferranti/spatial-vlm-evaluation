from __future__ import annotations

import argparse
from pathlib import Path

from .adapters import build_adapter
from .evaluation import evaluate, load_jsonl, save_results, summarize, write_jsonl
from .scenes import generate_dataset


def generate_command(args: argparse.Namespace) -> None:
    output = Path(args.output)
    samples = generate_dataset(output, samples_per_task=args.samples_per_task, seed=args.seed)
    dataset_path = output / "dataset.jsonl"
    write_jsonl(samples, dataset_path)
    print(f"Wrote {len(samples)} samples to {dataset_path}")


def evaluate_command(args: argparse.Namespace) -> None:
    samples = load_jsonl(Path(args.dataset))
    adapter = build_adapter(args.adapter, model_id=args.model_id)
    results = evaluate(samples, adapter, max_new_tokens=args.max_new_tokens)
    save_results(results, Path(args.output))
    print(summarize(results).to_string(index=False))
    print(f"Saved results to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spatial VLM evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate procedural spatial datasets")
    generate.add_argument("--output", required=True, help="Output run directory")
    generate.add_argument("--samples-per-task", type=int, default=8)
    generate.add_argument("--seed", type=int, default=42)
    generate.set_defaults(func=generate_command)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a VLM adapter on a dataset")
    evaluate_parser.add_argument("--dataset", required=True, help="Path to dataset.jsonl")
    evaluate_parser.add_argument("--output", required=True, help="Output result directory")
    evaluate_parser.add_argument("--adapter", choices=["mock", "smolvlm", "qwen2_5_vl"], default="mock")
    evaluate_parser.add_argument("--model-id", default=None)
    evaluate_parser.add_argument("--max-new-tokens", type=int, default=40)
    evaluate_parser.set_defaults(func=evaluate_command)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
