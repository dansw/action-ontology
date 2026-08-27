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

## Reproduce V8 from the Original Videos

After cloning, create one folder and place all 15 unmodified `.mov` files in
it. The filenames must match this list:

```text
data/original_videos/
  egg_catch_001.mov
  carry_coffee_table.mov
  clean_ketchup.mov
  cook_sunny_side_up_egg.mov
  duvet_cover.mov
  fitted_sheet_on_bed.mov
  get_into_car.mov
  hammer_and_nail.mov
  make_a_bed.mov
  opening_granola_bar.mov
  separate_the_yolk.mov
  sort_poker_hand.mov
  take_out_trash.mov
  turn_page_paperback.mov
  yo_yo.mov
```

The original videos remain ignored by git. The V8 annotation JSON files and
the exact 854-record training manifest are tracked, so no other dataset files
need to be restored separately. Install the project, then reproduce the V8
fine-tuning dataset with one command:

```bash
python -m pip install -e ".[ml,dev]"
action-ontologies prepare-v8 --videos-dir data/original_videos
```

This validates all filenames, reads every source video directly from the one
folder, extracts the exact frame indices referenced by the committed V8
manifest, and recreates each project's `prepared/train.jsonl`. It does not copy
the original videos into four different directories. A successful run reports
854 records at `data/combined_v8/prepared/train.jsonl`.

## Generated Folder Layout

Raw videos, extracted frames, per-project prepared JSONL files, model weights,
and outputs remain outside git. The annotation JSON files and exact combined
V8 training manifest under `data/` are tracked.

```text
data/
  egg_catch/
    annotations/
      egg_catch_001.json
    frames/
    prepared/
  diverse_actions/
    annotations/
    frames/
    prepared/
  survey_actions/
    annotations/
    frames/
    prepared/
  survey_actions_trash/     # take_out_trash needs a much higher --change-threshold
    annotations/            # prepared separately because take_out_trash needs a
                            # much higher change threshold than the other videos
    frames/
    prepared/
  combined_v8/
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
  ontology-lora-v6/           # v4's data plus two corrective annotations that fix a
                               # "grain" hallucination; trained alongside the
                               # known_identifiers registry code fix (see "Identifier
                               # drift" below); superseded by v7
  ontology-lora-v7/           # survey_actions annotations expanded from a partial
                               # slice to the full nine-video set (782 total examples,
                               # up from v6's 197); superseded by v8
  ontology-lora-v8/           # current recommended adapter -- v7's data plus 72 more
                               # survey_actions examples (854 total); see "Validating
                               # a new adapter" below for how it compares to v7
outputs/
  <video_id>.ontology.json         # one inference run
  diverse_validation_v8/           # one inference run per validation video, all with
    <video_id>.ontology.json       # the same adapter -- for comparing adapters
  videos/                          # captioned review videos rendered from an
    <video_id>.mp4                 # inference output (see "Generate captioned
    <video_id>.webm                # review videos" below)
```

Each activity retains its own annotations, extracted frames, and prepared
records. `prepare-v8` handles the different sampling settings and combines the
records automatically. The lower-level `action-ontologies prepare` command
remains available for preparing a new or individual project.

Annotation files use one JSON file per video:

```json
{
  "video_id": "egg_catch_001",
  "video_path": "videos/egg_catch_001.mov",
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

The annotation's `video_path` preserves the original per-project layout used
by the lower-level `prepare` command. The V8 reproduction command instead
maps each manifest video ID to the matching filename in `--videos-dir`.

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
  --video data/original_videos/egg_catch_001.mov \
  --output outputs/egg_catch_001.ontology.json \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --sample-fps 2
```

With the tuned LoRA adapter (`models/ontology-lora-v8`, the current
recommended adapter --
trained on 854 examples across egg-catch, five other activity videos, nine
survey videos, and a separate take-out-trash project; see "Train A Tuned
Model" below):

```bash
action-ontologies infer \
  --video data/original_videos/egg_catch_001.mov \
  --output outputs/egg_catch_001.ontology.json \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --adapter models/ontology-lora-v8 \
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

For the published V8 dataset, use the single-folder reproduction command:

```bash
action-ontologies prepare-v8 --videos-dir data/original_videos
```

The committed manifest is the source of truth because V8 includes 72
deliberately added survey frames beyond the V7 sampling pass. Re-running only
the older per-project sampling commands produces 782 records, not V8's 854.
`prepare-v8` extracts the manifest's precise frame indices, preserving the
dataset actually used for fine-tuning.

Run LoRA fine-tuning:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/train_lora.py \
  --train-jsonl data/combined_v8/prepared/train.jsonl \
  --base-model Qwen/Qwen3-VL-4B-Instruct \
  --output-dir models/ontology-lora-v8 \
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
signal that it has actually converged, not just run out of epochs. All runs
use 2x GTX 1080 Ti:

| adapter | examples | steps (20 epochs) | loss | wall time |
| --- | --- | --- | --- | --- |
| v6 | 197 | 500 | ~7.0 -> ~2.1-2.2 | ~12.5 hours |
| v7 | 782 | 1960 | ~7.3 -> ~2.05 | ~52 hours |
| v8 | 854 | 2140 | ~6.8 -> ~2.08 | ~57 hours |

None of the runs logged an eval/validation loss during training (no eval
split was configured) -- loss here is purely on the training set, so
"converged" means the training loss plateaued, not that generalization was
measured; that's what the validation runs in "Validating a new adapter"
below are for.

Training on CPU is supported for correctness checks, but expect it to be slow.
Use CUDA or ROCm for real tuning.

### A note on fixing model-side hallucinations via data, not prompts

If validation surfaces a specific hallucination (e.g. the model inventing an
action that isn't visible, like assuming a held food item is about to be
eaten), the most reliable fix found in practice was adding a small number of
training examples that straddle the exact failure -- not patching the system
prompt alone. Prompt-only patches proved fragile here: each one fixed the
reported frame but introduced a *different* hallucination elsewhere, since
the model's own weights, not the prompt, are where the bad prior lives. The
inference pipeline does retain one narrow safeguard: when the first pass
claims that a person is eating, `_verify_eating` requests a targeted second
pass before accepting that claim. Treat this as runtime verification, not a
substitute for corrective training data.

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

`ontology-lora-v6` was the default before v8: v4's data plus two corrective
annotations (in `opening_granola_bar`) that fix a hallucinated "grain"
detail, following the same narrow-fix approach. Separately, v6 was also the
first adapter trained and validated after the identifier-drift registry fix
described above (that fix is inference-time code, not training data, so it
applies retroactively to older adapters' inference too, but v6 is the one
it's been most thoroughly validated against). One known minor issue remained
in v6: a single spot-checked frame in a `hammer_and_nail` validation run
leaked prompt template text ("frame id hammer_and_nail_000256 (timestamp
4.267s)") into the generated description -- isolated to that one frame out
of everything spot-checked, much milder than v5's pervasive leakage. It did
not reproduce in v7 or v8's validation runs (the same frame, t=4.267s,
describes the hammer strike cleanly in both), though it was never
root-caused, so treat that as unconfirmed rather than fixed.

`ontology-lora-v7` and `ontology-lora-v8` are pure data-scale increases on
top of v6, with no prompt or system-prompt changes: v7 folded in a much
larger `survey_actions` annotation pass (438 examples across all nine survey
videos, versus a partial slice for v6), taking the combined set from 197 to
782 examples; v8 added 72 further `survey_actions` examples on top of that
(854 total). `ontology-lora-v8` is the current recommended adapter. The CLI
does not load an adapter implicitly; pass `--adapter models/ontology-lora-v8`
when tuned inference is intended.

### Validating a new adapter against the previous one

Every `infer` run records its own sampling parameters in the output file's
top-level fields (`sampling`, `change_threshold`, `percentile`,
`max_gap_seconds`, `context_frames`, `adapter`, ...), so a later run can be
made directly comparable to an earlier one by reading those fields back out
of the old output and reusing them verbatim with the new adapter -- the
frame timestamps then line up 1:1 between runs, since `information-gain`
sampling is deterministic for a given set of parameters.

`outputs/diverse_validation_v8/` was built this way against
`outputs/diverse_validation_v7/`, across all 15 validation videos (783
frames total). Comparing the two:

- 74% of frame descriptions are byte-for-byte identical between v7 and v8 --
  expected, since v8's training data is a strict superset of v7's.
- Aggregate resource/entity/action counts per frame are essentially flat
  (e.g. 1.41 -> 1.42 resources/frame), so the two adapters agree on *how
  much* to tag; where they disagree is *how*.
- Where frames differ, it's concentrated in two `diverse_actions` videos
  (`carry_coffee_table`, `get_into_car` -- only 13-25% identical), and the
  difference is systematic: v8 consistently uses the fine-grained resource
  naming the system prompt asks for (`"right fingers"`, `"left
  fingertips"`) where v7 fell back to coarser naming (`"hands"`, `"right
  hand"`), and v7 mislabeled a dog's `"front paws"` as a resource in one
  `carry_coffee_table` frame where it wasn't touching anything -- v8 doesn't
  make that error.
- v7 had one frame with a completely empty description
  (`get_into_car`, t=4.67s); v8 has none.
- Frames with zero resources tagged dropped from 122 to 109 (of 783).

Net: a small, consistent improvement, not a dramatic jump -- consistent
with v8 being v7's exact dataset plus 72 more examples rather than a
substantively different training run. One metric moved the "wrong" way
(frames with a duplicate-named action rose from 86 to 94), but spot-checking
shows this is mostly two actors legitimately performing the same named
action in one frame, not a new hallucination.

## Generate Captioned Review Videos

`scripts/build_description_videos.py` burns each frame's `description` into
the source video as a caption bar, for watching an inference run instead of
reading its JSON. Its `build_video(slug, video_path, ontology_path)`
function takes any video and any matching `*.ontology.json` output, so it
can be pointed at any adapter's validation run, not just the handful hardcoded
in the script's own `VIDEOS` dict:

```python
import sys
sys.path.insert(0, "scripts")
from pathlib import Path
import build_description_videos as m

m.OUTPUT_DIR = Path("outputs/videos")
m.build_video(
    "hammer_and_nail",
    Path("data/original_videos/hammer_and_nail.mov"),
    Path("outputs/diverse_validation_v8/hammer_and_nail.ontology.json"),
)
```

This writes `outputs/videos/<slug>.mp4` (H.264, already run through
`ffmpeg` internally). For a `.webm` sibling -- smaller, and needed for
browsers that won't play the H.264 profile inline -- transcode separately:

```bash
ffmpeg -y -i outputs/videos/hammer_and_nail.mp4 \
  -c:v libvpx-vp9 -crf 32 -b:v 0 -row-mt 1 \
  outputs/videos/hammer_and_nail.webm
```

Rendering is CPU-bound frame overlay plus an `ffmpeg` encode, not model
inference -- expect low single-digit minutes per video even for longer
clips, far faster than the inference run that produced the captions.

## Output Format

Inference writes:

```json
{
  "video_path": "data/original_videos/egg_catch_001.mov",
  "sampling": "fixed",
  "sample_fps": 2.0,
  "min_fps": null,
  "max_fps": null,
  "motion_threshold": null,
  "change_threshold": null,
  "percentile": null,
  "max_gap_seconds": null,
  "context_frames": 4,
  "model": "Qwen/Qwen3-VL-4B-Instruct",
  "adapter": "models/ontology-lora-v8",
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

The top-level sampling/model/adapter fields record exactly how the run was
produced (only the fields relevant to the chosen `--sampling` mode are
non-null), which is what makes it possible to reproduce a directly
comparable run later -- see "Validating a new adapter against the previous
one" above.

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
