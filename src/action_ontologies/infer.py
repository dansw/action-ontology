from __future__ import annotations

import tempfile
from collections import deque
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from .device import needs_eager_attention, select_device, torch_dtype_for_device
from .json_utils import extract_json_object, write_json
from .prompts import SYSTEM_PROMPT, frame_prompt
from .schema import FrameOntology
from frame_sampling import sample_by_information_gain

from .video import sample_video_frames, sample_video_frames_adaptive


def run_inference(
    *,
    video: str | Path,
    output: str | Path,
    model_name: str,
    adapter: str | None = None,
    sample_fps: float = 2.0,
    sampling: str = "fixed",
    min_fps: float = 1.0,
    max_fps: float = 15.0,
    motion_threshold: float = 6.0,
    change_threshold: float = 45.0,
    percentile: float = 90.0,
    max_gap_seconds: float | None = 2.0,
    context_frames: int = 4,
    device: str = "auto",
    max_new_tokens: int = 768,
) -> None:
    if sampling not in ("fixed", "adaptive", "information-gain"):
        raise ValueError(f"unsupported sampling strategy: {sampling!r}")
    if context_frames < 0:
        raise ValueError("context_frames must be zero or greater")
    device = select_device(device)
    model, processor = _load_model(model_name, device=device, adapter=adapter)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="action_ontology_frames_") as tmpdir:
        if sampling == "adaptive":
            frames = sample_video_frames_adaptive(
                video, tmpdir, min_fps=min_fps, max_fps=max_fps, motion_threshold=motion_threshold
            )
        elif sampling == "information-gain":
            frames = sample_by_information_gain(
                video,
                tmpdir,
                change_threshold=change_threshold,
                percentile=percentile,
                max_gap_seconds=max_gap_seconds,
            )
        else:
            frames = sample_video_frames(video, tmpdir, sample_fps)
        results = []
        history: deque[dict[str, Any]] = deque(maxlen=context_frames if context_frames > 0 else 0)
        for frame in tqdm(frames, desc="frames"):
            image = Image.open(frame.image_path).convert("RGB")
            prompt = frame_prompt(frame.frame_id, frame.timestamp_seconds, history=list(history))
            raw = _generate(model, processor, image, prompt, max_new_tokens)
            try:
                parsed = extract_json_object(raw)
                ontology = FrameOntology.from_dict(
                    parsed,
                    fallback_frame_id=frame.frame_id,
                    fallback_timestamp=frame.timestamp_seconds,
                )
                ontology = FrameOntology(
                    **{
                        **ontology.__dict__,
                        "frame_id": frame.frame_id,
                        "timestamp_seconds": frame.timestamp_seconds,
                        "frame_index": frame.frame_index,
                    }
                )
                results.append(ontology.to_dict())
                if context_frames > 0:
                    history.append(
                        {
                            "timestamp_seconds": ontology.timestamp_seconds,
                            "description": ontology.description,
                            "actions": [action.name for action in ontology.actions],
                        }
                    )
            except Exception as exc:
                results.append(
                    {
                        "frame_id": frame.frame_id,
                        "frame_index": frame.frame_index,
                        "timestamp_seconds": round(frame.timestamp_seconds, 6),
                        "description": "",
                        "resources": [],
                        "entities": [],
                        "actions": [],
                        "ontological_phrases": [],
                        "error": f"failed to parse model output: {exc}",
                        "raw_output": raw,
                    }
                )
    write_json(
        str(output),
        {
            "video_path": str(video),
            "sampling": sampling,
            "sample_fps": sample_fps if sampling == "fixed" else None,
            "min_fps": min_fps if sampling == "adaptive" else None,
            "max_fps": max_fps if sampling == "adaptive" else None,
            "motion_threshold": motion_threshold if sampling == "adaptive" else None,
            "change_threshold": change_threshold if sampling == "information-gain" else None,
            "percentile": percentile if sampling == "information-gain" else None,
            "max_gap_seconds": max_gap_seconds if sampling == "information-gain" else None,
            "context_frames": context_frames,
            "model": model_name,
            "adapter": adapter,
            "frames": results,
        },
    )


def _load_model(model_name: str, *, device: str, adapter: str | None):
    import torch
    from transformers import AutoProcessor

    model_class = _resolve_model_class()
    dtype = torch_dtype_for_device(device)
    # Cap vision tokens per frame; uncapped, high-resolution frames (e.g. phone
    # screen recordings) produce thousands of vision tokens whose O(n^2) eager
    # attention matrix can exceed GPU memory.
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        min_pixels=256 * 28 * 28,
        max_pixels=768 * 28 * 28,
    )
    multi_gpu = device == "cuda" and torch.cuda.device_count() > 1
    load_kwargs = dict(
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if multi_gpu:
        load_kwargs["device_map"] = "auto"
    if needs_eager_attention(device):
        load_kwargs["attn_implementation"] = "eager"
    model = model_class.from_pretrained(model_name, **load_kwargs)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    if not multi_gpu:
        model.to(device)
    model.eval()
    return model, processor


def _resolve_model_class():
    import transformers

    for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "Qwen2_5_VLForConditionalGeneration"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    raise ImportError("installed transformers version does not provide a supported vision-language model class")


def _generate(model: Any, processor: Any, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
    import torch

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    if hasattr(processor, "apply_chat_template"):
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = f"{SYSTEM_PROMPT}\n\n{prompt}"
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    prompt_length = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
    generated_only = generated[:, prompt_length:] if prompt_length else generated
    return processor.batch_decode(generated_only, skip_special_tokens=True)[0]

