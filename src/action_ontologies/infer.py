from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from .device import select_device, torch_dtype_for_device
from .json_utils import extract_json_object, write_json
from .prompts import SYSTEM_PROMPT, frame_prompt
from .schema import FrameOntology
from .video import sample_video_frames


def run_inference(
    *,
    video: str | Path,
    output: str | Path,
    model_name: str,
    adapter: str | None = None,
    sample_fps: float = 2.0,
    device: str = "auto",
    max_new_tokens: int = 768,
) -> None:
    device = select_device(device)
    model, processor = _load_model(model_name, device=device, adapter=adapter)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="action_ontology_frames_") as tmpdir:
        frames = sample_video_frames(video, tmpdir, sample_fps)
        results = []
        for frame in tqdm(frames, desc="frames"):
            image = Image.open(frame.image_path).convert("RGB")
            prompt = frame_prompt(frame.frame_id, frame.timestamp_seconds)
            raw = _generate(model, processor, image, prompt, device, max_new_tokens)
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
                        "frame_index": frame.frame_index,
                    }
                )
                results.append(ontology.to_dict())
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
            "sample_fps": sample_fps,
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
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = model_class.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
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


def _generate(model: Any, processor: Any, image: Image.Image, prompt: str, device: str, max_new_tokens: int) -> str:
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
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    prompt_length = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
    generated_only = generated[:, prompt_length:] if prompt_length else generated
    return processor.batch_decode(generated_only, skip_special_tokens=True)[0]

