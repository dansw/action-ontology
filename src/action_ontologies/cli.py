from __future__ import annotations

import argparse
import json

from .infer import run_inference
from .json_utils import write_json
from .prepare import prepare_project
from .summarize import summarize_file


def _add_sampling_arguments(subparser: argparse.ArgumentParser, *, context_help: str) -> None:
    subparser.add_argument(
        "--sampling",
        choices=["fixed", "adaptive", "information-gain"],
        default="fixed",
        help=(
            "fixed: uniform --sample-fps. adaptive: variable rate capped between --min-fps/--max-fps, "
            "driven by frame-to-frame motion. information-gain: no rate cap at all -- accumulates "
            "frame-to-frame change and samples a frame as soon as --change-threshold worth of new visual "
            "information has built up since the last sample, so brief single-frame motion is never stepped over."
        ),
    )
    subparser.add_argument("--sample-fps", type=float, default=2.0, help="fixed sampling: frames per second")
    subparser.add_argument(
        "--min-fps", type=float, default=1.0, help="adaptive sampling: floor rate for static stretches"
    )
    subparser.add_argument(
        "--max-fps", type=float, default=15.0, help="adaptive sampling: ceiling rate during motion"
    )
    subparser.add_argument(
        "--motion-threshold",
        type=float,
        default=6.0,
        help="adaptive sampling: mean grayscale pixel difference (0-255) needed to sample between min_fps and max_fps",
    )
    subparser.add_argument(
        "--change-threshold",
        type=float,
        default=45.0,
        help=(
            "information-gain sampling: cumulative --percentile-th percentile pixel difference (0-255) "
            "needed to trigger a new sample"
        ),
    )
    subparser.add_argument(
        "--percentile",
        type=float,
        default=90.0,
        help=(
            "information-gain sampling: which percentile of the frame's pixelwise change to measure per step "
            "(0-100). A high percentile stays sensitive to a change confined to part of the frame -- e.g. hands "
            "unwrapping something while the rest of the body and background hold still -- that a plain mean would "
            "dilute against the static majority of pixels and under-react to."
        ),
    )
    subparser.add_argument(
        "--max-gap-seconds",
        type=float,
        default=2.0,
        help="information-gain sampling: force a checkpoint frame after this long with no sample; use 0 to disable",
    )
    subparser.add_argument("--context-frames", type=int, default=4, help=context_help)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="action-ontologies")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="extract frames and create training JSONL")
    prepare.add_argument("--project-dir", required=True)
    prepare.add_argument("--output-jsonl", required=True)
    _add_sampling_arguments(
        prepare,
        context_help="include up to this many preceding ground-truth frames as history in each training prompt; 0 disables",
    )

    infer = subparsers.add_parser("infer", help="extract ontology JSON from a video")
    infer.add_argument("--video", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    infer.add_argument("--adapter")
    _add_sampling_arguments(
        infer,
        context_help="include up to this many preceding frames' descriptions as history in each prompt, so the "
        "model doesn't re-describe an in-progress or finished action as just starting; 0 disables",
    )
    infer.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "rocm", "mps"])
    infer.add_argument("--max-new-tokens", type=int, default=768)

    summarize = subparsers.add_parser(
        "summarize", help="build a deduplicated master list of resources, entities, and actions from an inference output"
    )
    summarize.add_argument("--input", required=True)
    summarize.add_argument("--output")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare":
        count = prepare_project(
            args.project_dir,
            args.sample_fps,
            args.output_jsonl,
            sampling=args.sampling,
            min_fps=args.min_fps,
            max_fps=args.max_fps,
            motion_threshold=args.motion_threshold,
            change_threshold=args.change_threshold,
            percentile=args.percentile,
            max_gap_seconds=args.max_gap_seconds if args.max_gap_seconds > 0 else None,
            context_frames=args.context_frames,
        )
        print(f"created {count} training records at {args.output_jsonl}")
        return 0
    if args.command == "infer":
        run_inference(
            video=args.video,
            output=args.output,
            model_name=args.model,
            adapter=args.adapter,
            sample_fps=args.sample_fps,
            sampling=args.sampling,
            min_fps=args.min_fps,
            max_fps=args.max_fps,
            motion_threshold=args.motion_threshold,
            change_threshold=args.change_threshold,
            percentile=args.percentile,
            max_gap_seconds=args.max_gap_seconds if args.max_gap_seconds > 0 else None,
            context_frames=args.context_frames,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"wrote ontology output to {args.output}")
        return 0
    if args.command == "summarize":
        summary = summarize_file(args.input)
        if args.output:
            write_json(args.output, summary)
            print(f"wrote summary to {args.output}")
        else:
            print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
