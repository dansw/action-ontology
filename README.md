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
  combined_v6/
    prepared/
      train.jsonl        # concatenation of every project's prepared/train.jsonl
models/
  ontology-lora-diverse/      # first combined-set adapter (egg_catch + diverse_actions)
  ontology-lora-diverse-v2/   # granular-resource retrain of the above; superseded by v3
  ontology-lora-v3/           # first full-combined-set adapter; superseded by v4
  ontology-lora-v4/           # same data as v3 plus six descriptions rewritten to fix
                               # an eating-hallucination bug; superseded by v6
  ontology-lora-v5/           # experimental: removed ALL speculative "about to X" /
                               # negation language project-wide; regressed output
                               # coherence on unrelated frames (meta-language leaking
                               # into descriptions) -- do not use as a base yet
  ontology-lora-v6/           # current default adapter -- v4's data plus two corrective
                               # annotations that fix a "grain" hallucination; trained
                               # alongside the known_identifiers registry code fix (see
                               # "Identifier drift" below), which applies to every
                               # adapter's inference regardless of training data
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
  --sample-fps 8 \
  --output-jsonl data/egg_catch/prepared/train.jsonl
```

(`egg_catch`'s annotations were built at 8 fps -- `--sample-fps` must match the rate the
annotation frame indices assume, or most sampled frames won't match any annotation and
`prepare` will silently produce far fewer training records than expected. Always sanity-check
the printed record count against how many frames you actually annotated.)

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

With the tuned LoRA adapter (`models/ontology-lora-v6`, the current default --
trained on 197 examples across egg-catch, six other activity videos, and nine
further survey videos; see "Train A Tuned Model" below):

```bash
action-ontologies infer \
  --video data/egg_catch/videos/egg_catch_001.mp4 \
  --output outputs/egg_catch_001.ontology.json \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --adapter models/ontology-lora-v6 \
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

### Identifier drift

`--context-frames` alone doesn't stop a video-local `identifier` (the string
used to track one real-world object across frames, e.g. `wood_block`) from
splitting into two: the sliding history window forgets an object once it
falls out of the last N frames, and the model then has nothing telling it
"this name was already assigned an identifier" and picks a fresh one.

Both `infer` and `prepare` maintain a second, unbounded-for-the-whole-video
registry (`known_identifiers`, capped at 30 entries as a safety net) that is
listed in every frame's prompt regardless of the history window, so the
model can reuse an identifier it hasn't seen in a while. On its own this
still isn't enough: once the model *does* invent a duplicate identifier for
an object it already named, both identifiers looked equally valid from then
on, and the model would keep alternating between them for the rest of the
video (observed on a 4.5-minute video: a duvet/blanket entity split across
`bedding` and `fabric`, and a mattress entity split across `mattress` and
`mattress_fabric`, both oscillating for the remainder of the clip).

The fix tracks every (tokenized) name ever used under each identifier, not
just the latest one, and rewrites an element onto the existing identifier
whenever its name matches -- exactly, or by whole-word containment (e.g.
`"duvet"` vs. `"duvet fabric"`) -- a name already registered under a
*different* identifier. This runs identically whether the earlier
registration happened in a previous frame (cross-frame drift) or earlier in
the very same frame's own `resources`/`entities` list (a same-frame
duplicate, e.g. both `"duvet"` and `"bedding"` listed as separate entities
in one frame); elements that end up sharing an identifier after
canonicalizing are then collapsed to a single entry. The containment check
is whole-word only, not substring or single-shared-word, specifically so it
does not undo a deliberate state-change re-identification (e.g. `wrapper` ->
`wrapper fragment` once a piece tears off, or `nail` -> `driven_nail`-style
namings the model may legitimately want) -- `"bed"` vs. `"bedding"` and
`"granola bar"` vs. `"wrapper fragment"` correctly do not match.

This is inference-time/prepare-time logic, not something baked into any
adapter's weights -- it benefits every adapter's inference immediately, and
regenerating training data with `prepare` also cleans up any drift already
present in existing hand-authored annotations before training on them. See
`_canonicalize_known_identifiers` in `infer.py` and `prepare.py`.

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
action-ontologies prepare --project-dir data/egg_catch --output-jsonl data/egg_catch/prepared/train.jsonl --sample-fps 8
action-ontologies prepare --project-dir data/diverse_actions --output-jsonl data/diverse_actions/prepared/train.jsonl \
  --sampling information-gain --change-threshold 45 --percentile 90 --max-gap-seconds 5
action-ontologies prepare --project-dir data/survey_actions --output-jsonl data/survey_actions/prepared/train.jsonl \
  --sampling information-gain --change-threshold 45 --percentile 90 --max-gap-seconds 3
action-ontologies prepare --project-dir data/survey_actions_trash --output-jsonl data/survey_actions_trash/prepared/train.jsonl \
  --sampling information-gain --change-threshold 1000 --percentile 90 --max-gap-seconds 4

mkdir -p data/combined_v6/prepared
cat data/egg_catch/prepared/train.jsonl data/diverse_actions/prepared/train.jsonl \
    data/survey_actions/prepared/train.jsonl data/survey_actions_trash/prepared/train.jsonl \
    > data/combined_v6/prepared/train.jsonl
```

`--change-threshold`/`--sampling`/etc. must exactly match whatever was used when the
project's annotation frame indices were originally picked, or `prepare` will silently
match only a handful of frames instead of your full annotated set (this has bitten this
project twice: once for `egg_catch`'s fixed-fps mismatch, once for `diverse_actions`
after `frame_sampling`'s change-detection metric was rewritten -- always check the
printed record count against your actual annotation count before training on it).

Run LoRA fine-tuning:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/train_lora.py \
  --train-jsonl data/combined_v6/prepared/train.jsonl \
  --base-model Qwen/Qwen3-VL-4B-Instruct \
  --output-dir models/ontology-lora-v6 \
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
signal that it has actually converged, not just run out of epochs. The v6
adapter's loss went from ~7.0 to ~2.1-2.2 over 500 steps (197 examples, 20
epochs), taking about 12.5 hours on 2x GTX 1080 Ti.

Training on CPU is supported for correctness checks, but expect it to be slow.
Use CUDA or ROCm for real tuning.

### A note on fixing model-side hallucinations via data, not prompts

If validation surfaces a specific hallucination (e.g. the model inventing an
action that isn't visible, like assuming a held food item is about to be
eaten), the most reliable fix found in practice was adding a small number of
training examples that straddle the exact failure -- not patching the system
prompt at inference time. Prompt-only patches proved fragile here: each one
fixed the reported frame but introduced a *different* hallucination elsewhere,
since the model's own weights, not the prompt, are where the bad prior lives.

Also resist the urge to over-generalize a narrow fix. `ontology-lora-v5` was
an experiment that went further -- stripping ALL speculative "about to X"
phrasing and all negation language ("not eating it") project-wide, plus
adding a matching general instruction to `SYSTEM_PROMPT` -- and while it did
eliminate the target hallucination cleanly, it introduced a broader
regression: the model started leaking instruction-like meta-phrasing into
descriptions (e.g. echoing "each limb's own current position and contact"
almost verbatim from the system prompt) and fabricating unrelated scene
details on a second, unrelated video. `ontology-lora-v4` -- which fixed only
the specific reported frames with a handful of added training examples,
leaving everything else untouched -- was kept as the default over v5 for
this reason. Prefer minimal, targeted data changes over broad rewrites of
the system prompt or the whole training set, even when the broader change
seems more principled.

`ontology-lora-v6` is the current default: v4's data plus two corrective
annotations (in `opening_granola_bar`) that fix a hallucinated "grain"
detail, following the same narrow-fix approach. Separately, v6 was also the
first adapter trained and validated after the identifier-drift registry fix
described above (that fix is inference-time code, not training data, so it
applies retroactively to older adapters' inference too, but v6 is the one
it's been most thoroughly validated against). One known minor issue remains
in v6: a single spot-checked frame in a `hammer_and_nail` validation run
leaked prompt template text ("frame id hammer_and_nail_000256 (timestamp
4.267s)") into the generated description -- isolated to that one frame out
of everything spot-checked, much milder than v5's pervasive leakage, but
real and not yet root-caused.

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
