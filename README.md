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
models/
  ontology-lora/
outputs/
  egg_catch_001.ontology.json
```

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
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --sample-fps 2
```

With a tuned LoRA adapter:

```bash
action-ontologies infer \
  --video data/egg_catch/videos/egg_catch_001.mp4 \
  --output outputs/egg_catch_001.ontology.json \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --adapter models/ontology-lora \
  --sample-fps 2
```

## Train A Tuned Model

Start with focused annotations for a narrow activity domain, then expand. Good
training examples should include:

- important object parts, not only whole objects;
- resource transitions, such as an egg becoming a held resource;
- state changes, such as `egg` becoming `broken egg`;
- concise frame descriptions;
- ontological phrases that match visible interactions.

Run LoRA fine-tuning:

```bash
python scripts/train_lora.py \
  --train-jsonl data/egg_catch/prepared/train.jsonl \
  --base-model Qwen/Qwen2.5-VL-3B-Instruct \
  --output-dir models/ontology-lora \
  --epochs 3 \
  --batch-size 1 \
  --gradient-accumulation-steps 8
```

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

## Development

```bash
python -m pytest -q
```

The tests avoid large model downloads and cover parsing, validation, prompt
construction, and frame sampling behavior.
