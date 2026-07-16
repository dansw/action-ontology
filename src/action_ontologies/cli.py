from __future__ import annotations

import argparse

from .infer import run_inference
from .prepare import prepare_project


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="action-ontologies")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="extract frames and create training JSONL")
    prepare.add_argument("--project-dir", required=True)
    prepare.add_argument("--sample-fps", type=float, default=2.0)
    prepare.add_argument("--output-jsonl", required=True)

    infer = subparsers.add_parser("infer", help="extract ontology JSON from a video")
    infer.add_argument("--video", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    infer.add_argument("--adapter")
    infer.add_argument("--sample-fps", type=float, default=2.0)
    infer.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "rocm", "mps"])
    infer.add_argument("--max-new-tokens", type=int, default=768)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare":
        count = prepare_project(args.project_dir, args.sample_fps, args.output_jsonl)
        print(f"created {count} training records at {args.output_jsonl}")
        return 0
    if args.command == "infer":
        run_inference(
            video=args.video,
            output=args.output,
            model_name=args.model,
            adapter=args.adapter,
            sample_fps=args.sample_fps,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"wrote ontology output to {args.output}")
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
