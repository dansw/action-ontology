# Action Ontologies

Frame-by-frame ontology extraction for videos. The project uses a pretrained
vision-language model as the base model, then supports LoRA fine-tuning on your
own annotated videos.

The ontology has three main element types:

- `entities`: separable objects or object parts involved in the visible task.
- `resources`: entities controlled by an autonomous mover, including body parts
  and held tools or materials.
- `actions`: meaningful interactions, state changes, or task motions.

The model output is intentionally scoped to the main activity in the video so
background objects are omitted unless they participate in the action.

## Hardware Targets

The code runs on:

- CPU: works everywhere, slow for large vision-language models.
- NVIDIA CUDA: install a CUDA PyTorch wheel.
- AMD ROCm: install a ROCm PyTorch wheel on supported Linux systems.
- Apple Silicon MPS: useful for local experimentation, but CUDA/ROCm are better
  for training.

Device selection is automatic by default and can be overridden with `--device`.
Use `--device rocm` on AMD systems; PyTorch exposes ROCm through its CUDA API.

## Install

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install PyTorch for your accelerator before installing the ML extras if needed:

CPU:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[ml,dev]"
```

NVIDIA CUDA 12.1:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e ".[ml,dev]"
```

AMD ROCm 6.1:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1
python -m pip install -e ".[ml,dev]"
```

## Folder Layout

Keep raw and generated data out of git.

```text
data/
  egg_catch/
    videos/
      egg_catch_001.mp4
    annotations/
      egg_catch_001.json
    frames/
    prepared/
  diverse_actions/
    videos/
    annotations/
    frames/
    prepared/
  survey_actions/
    videos/
    annotations/
    frames/
    prepared/
  survey_actions_trash/     # take_out_trash needs a much higher --change-threshold
    videos/                 # than the rest (continuous walking motion), so it's
    annotations/             # prepared as its own project with its own sampling args
    frames/
    prepared/
  combined_v3/
    prepared/
      train.jsonl        # concatenation of every project's prepared/train.jsonl
models/
  ontology-lora-diverse/      # first combined-set adapter (egg_catch + diverse_actions)
  ontology-lora-diverse-v2/   # granular-resource retrain of the above; superseded by v3
  ontology-lora-v3/           # current default adapter, trained on the full combined set
outputs/
  egg_catch_001.ontology.json
```

Each activity gets its own project directory (own `videos/`, `annotations/`,
`frames/`, `prepared/`), so `action-ontologies prepare` can be run per project
independently. To fine-tune across all of them, concatenate their
`prepared/train.jsonl` files (they use paths relative to the repo root, so a
plain `cat` works) into one combined training file before running
`train_lora.py`.

Annotation files use one JSON file per video:

```json
{
  "video_id": "egg_catch_001",
  "video_path": "videos/egg_catch_001.mp4",
  "frames": [
    {
      "frame_id": "egg_catch_001_000000",
      "timestamp_seconds": 0.0,
      "description": "hands are open and moving toward a falling egg",
      "resources": [
        {"name": "left hand", "description": "open hand preparing to catch"},
        {"name": "right hand", "description": "open hand preparing to catch"}
      ],
      "entities": [
        {"name": "egg", "description": "falling egg between the hands"}
      ],
      "actions": [
        {
          "name": "prepare to catch",
          "actor": "left hand and right hand",
          "target": "egg",
          "description": "hands move into position under the egg"
        }
      ],
      "ontological_phrases": ["hands prepare to catch egg"]
    }
  ]
}
```

## Prepare Training Data

Extract sampled frames and build a JSONL training file:

```bash
action-ontologies prepare \
  --project-dir data/egg_catch \
  --sample-fps 2 \
  --output-jsonl data/egg_catch/prepared/train.jsonl
```

Each JSONL record contains an image path plus the expected ontology JSON for
that frame. Review the prepared file before training.

## Run Inference

The default model is a small, practical starting point. For higher accuracy, use
a stronger vision-language base model that fits your hardware.

```bash
action-ontologies infer \
  --video data/egg_catch/videos/egg_catch_001.mp4 \
  --output outputs/egg_catch_001.ontology.json \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --sample-fps 2
```

With the tuned LoRA adapter (`models/ontology-lora-v3`, the current default --
trained on 195 examples across egg-catch, six other activity videos, and nine
further survey videos; see "Train A Tuned Model" below):

```bash
action-ontologies infer \
  --video data/egg_catch/videos/egg_catch_001.mp4 \
  --output outputs/egg_catch_001.ontology.json \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --adapter models/ontology-lora-v3 \
  --sample-fps 2
```

### Sampling strategies

`--sample-fps` (the default) samples uniformly, which wastes time on static
stretches and can step over motion that happens faster than the rate allows.
Two alternatives sample at a variable rate instead:

```bash
# adaptive: capped between --min-fps and --max-fps, driven by motion
action-ontologies infer --video ... --output ... \
  --sampling adaptive --min-fps 1 --max-fps 15

# information-gain: no rate cap -- fires as soon as enough frame-to-frame
# change has accumulated since the last sample, so a single fast-moving raw
# frame can still be caught even if a fixed or capped rate would skip past it
action-ontologies infer --video ... --output ... \
  --sampling information-gain --change-threshold 45 --percentile 90 --max-gap-seconds 2
```

`information-gain` measures change per step as the `--percentile`-th
percentile (default 90th) of the pixelwise absolute difference between
consecutive frames, not the mean. A mean is diluted by however much of the
frame stays static, so a change confined to part of the frame -- hands
unwrapping something while the rest of the body and background hold still,
say -- can sit well under a mean-based threshold for many consecutive frames
even while genuinely, continuously happening, and the whole multi-second
transition collapses into a single stale sample. A high percentile reports
the magnitude of the frame's most-changed pixels instead of averaging them
away against the static majority, so it stays sensitive to that kind of
localized, gradual change while still ignoring sensor noise (which nudges
most pixels a little rather than a concentrated region a lot). The fast-motion
case needs no special handling either way: a single consecutive-frame delta
large enough on its own crosses `--change-threshold` immediately, so brief
motion between two raw frames is never stepped over regardless of the metric.

`information-gain` is implemented in the standalone `frame_sampling` package
(`src/frame_sampling/`), independent of the ontology-specific code, so it can
be reused for other video-analysis tasks. See
`frame_sampling.sample_by_information_gain` for the algorithm and its
`device="cuda"` option for GPU-accelerated batch diffing on longer videos.

### Frame history

Each frame is otherwise inferred independently, with no memory of what came
before it -- on longer videos this shows up as an in-progress or finished
action getting re-described as "about to start", or a state that was already
established (e.g. a completed task) flickering back to an earlier one.
`--context-frames N` (default 4, use 0 to disable) includes the last N
frames' descriptions and actions as history in the prompt, oldest to newest,
so the model can judge progress state consistently:

```bash
action-ontologies infer --video ... --output ... --context-frames 4
```

The history instructs the model to use it only for progress-state
consistency, not to copy its wording -- without that explicit
anti-copying instruction, models tend to lean on the provided text and
collapse into repeating the same sentence for long static-looking stretches
instead of describing each frame's actual visible detail. `prepare` accepts
the same flag so training prompts are built the same way inference will see
them, using the ground-truth annotation sequence as history (teacher
forcing).

## Train A Tuned Model

Start with focused annotations for a narrow activity domain, then expand to
more videos before fine-tuning. A single narrow video is only enough data to
memorize that video, not to teach a generalizable concept: a model trained on
just one clip reproduces its training frames closely but blends or
hallucinates on new frames from the *same* clip that weren't annotated, let
alone a different video. Generalization needs diversity across videos
(different subjects, objects, backgrounds, camera angles) covering the same
action categories, plus a held-out validation clip to actually measure it.

Good training examples should include:

- important object parts, not only whole objects;
- resource transitions, such as an egg becoming a held resource;
- state changes, such as `egg` becoming `broken egg`;
- concise frame descriptions;
- ontological phrases that match visible interactions.

When annotating a resource whose engagement changes moment to moment (a hand
that releases something and goes idle, or switches roles with the other
hand), include a training example that straddles that exact transition --
one frame just before the change and one just after, both with the released
limb correctly omitted from `resources` per the granularity rule. Without at
least one such example, a history-conditioned model tends to keep repeating
whatever symmetric description it used for several consecutive frames even
after one limb's real, visible state has changed underneath it.

Prepare each project directory separately, then combine before training.
Pick `--change-threshold` per project to fit its motion character -- a video
with continuous motion throughout (e.g. someone walking with a swaying
object) needs a much higher threshold than a mostly-static one, or the
sampler will select far more frames than are practical to hand-annotate:

```bash
action-ontologies prepare --project-dir data/egg_catch --output-jsonl data/egg_catch/prepared/train.jsonl
action-ontologies prepare --project-dir data/diverse_actions --output-jsonl data/diverse_actions/prepared/train.jsonl \
  --sampling information-gain --change-threshold 45 --percentile 90 --max-gap-seconds 5
action-ontologies prepare --project-dir data/survey_actions --output-jsonl data/survey_actions/prepared/train.jsonl \
  --sampling information-gain --change-threshold 45 --percentile 90 --max-gap-seconds 3
action-ontologies prepare --project-dir data/survey_actions_trash --output-jsonl data/survey_actions_trash/prepared/train.jsonl \
  --sampling information-gain --change-threshold 1000 --percentile 90 --max-gap-seconds 4

mkdir -p data/combined_v3/prepared
cat data/egg_catch/prepared/train.jsonl data/diverse_actions/prepared/train.jsonl \
    data/survey_actions/prepared/train.jsonl data/survey_actions_trash/prepared/train.jsonl \
    > data/combined_v3/prepared/train.jsonl
```

Run LoRA fine-tuning:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/train_lora.py \
  --train-jsonl data/combined_v3/prepared/train.jsonl \
  --base-model Qwen/Qwen3-VL-4B-Instruct \
  --output-dir models/ontology-lora-v3 \
  --epochs 20 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 3e-4
```

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` avoids a CUDA
out-of-memory crash from allocator fragmentation -- training batches one
differently-sized image at a time, and the varying tensor shapes can
fragment the allocator's cache badly enough to fail an allocation well
before the GPU is actually full (the OOM error's own message names this
fragmentation and suggests this exact fix).

A handful of epochs is rarely enough to shift output conventions on a small
dataset -- loss plateauing (watch `grad_norm` flatten toward zero) is the
signal that it has actually converged, not just run out of epochs. The v3
adapter's loss went from ~7.1 to ~2.3-2.5 over 500 steps (195 examples, 20
epochs), taking about 11.5 hours on 2x GTX 1080 Ti.

Training on CPU is supported for correctness checks, but expect it to be slow.
Use CUDA or ROCm for real tuning.

## Output Format

Inference writes:

```json
{
  "video_path": "data/egg_catch/videos/egg_catch_001.mp4",
  "sample_fps": 2.0,
  "frames": [
    {
      "frame_id": "egg_catch_001_000000",
      "frame_index": 0,
      "timestamp_seconds": 0.0,
      "description": "...",
      "resources": [],
      "entities": [],
      "actions": [],
      "ontological_phrases": []
    }
  ]
}
```

## Summarize

Fold a video's per-frame output into a deduplicated master list of the
resources, entities, and actions seen anywhere in the video:

```bash
action-ontologies summarize \
  --input outputs/egg_catch_001.ontology.json \
  --output outputs/egg_catch_001.summary.json
```

Omit `--output` to print the summary to stdout instead. Deduplication is
case-insensitive and, for resources/entities, also keys on `identifier` when
present; the first description seen for each unique item is kept.

## Development

```bash
python -m pytest -q
```

The tests avoid large model downloads and cover parsing, validation, prompt
construction, and frame sampling behavior.
