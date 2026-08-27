from __future__ import annotations

import dataclasses
import tempfile
from collections import deque
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from .device import needs_eager_attention, select_device, torch_dtype_for_device
from .json_utils import extract_json_object, write_json
from .prompts import SYSTEM_PROMPT, frame_prompt
from .schema import FrameOntology, normalize_key
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
        known_identifiers: dict[str, str] = {}
        identifier_names: dict[str, set] = {}
        for frame in tqdm(frames, desc="frames"):
            image = Image.open(frame.image_path).convert("RGB")
            prompt = frame_prompt(
                frame.frame_id, frame.timestamp_seconds, history=list(history), known_identifiers=known_identifiers
            )
            raw = _generate(model, processor, image, prompt, max_new_tokens)
            history_entry = None
            try:
                parsed = extract_json_object(raw)
                if _mentions_eating(parsed) and not _verify_eating(model, processor, image):
                    raw = _generate(model, processor, image, prompt + EATING_CORRECTION, max_new_tokens)
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
                ontology = _canonicalize_known_identifiers(known_identifiers, identifier_names, ontology)
                results.append(ontology.to_dict())
                history_entry = {
                    "timestamp_seconds": ontology.timestamp_seconds,
                    "actions": [action.name for action in ontology.actions],
                }
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
                # Still record a history entry for this timestamp even though we have
                # no real content for it -- otherwise the next frame's history window
                # silently skips this moment entirely, making the model believe less
                # real time has passed than actually did and anchoring it one frame
                # behind for the rest of the video (confirmed via a lag-correlation
                # analysis on a real validation run: get_into_car's one parse failure
                # produced a clean +1-frame lag in the four frames right after it).
                history_entry = {
                    "timestamp_seconds": frame.timestamp_seconds,
                    "actions": [],
                }
            if context_frames > 0:
                history.append(history_entry)
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
    elif device == "cuda":
        # Loading directly onto a single ROCm/CUDA GPU avoids an extremely
        # slow whole-model post-load migration (minutes on gfx1030).
        load_kwargs["device_map"] = {"": "cuda"}
    if needs_eager_attention(device):
        load_kwargs["attn_implementation"] = "eager"
    model = model_class.from_pretrained(model_name, **load_kwargs)
    if device == "cuda":
        # Direct loading uses asynchronous device copies; finish them before
        # PEFT creates and moves adapter tensors on the same ROCm/CUDA device.
        torch.cuda.synchronize()
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    if device != "cuda":
        model.to(device)
    model.eval()
    return model, processor


_MAX_KNOWN_IDENTIFIERS = 30


def _name_tokens(name: str) -> frozenset:
    return frozenset(normalize_key(name).split())


def _same_referent(tokens_a: frozenset, tokens_b: frozenset) -> bool:
    """True if one name's words are a non-empty subset of the other's, e.g.
    "duvet" vs "duvet fabric" or "mattress" vs "mattress fabric" -- but NOT
    for names that merely share a generic word ("bed" vs "bedding", "granola
    bar" vs "wrapper fragment"), so a deliberate state-change re-identifier
    (e.g. "wrapper" -> "wrapper fragment" once a piece tears off) is not
    forced back onto the original identifier."""
    return bool(tokens_a) and bool(tokens_b) and (tokens_a <= tokens_b or tokens_b <= tokens_a)


def _dedupe_by_identifier(elements: list) -> list:
    seen: set = set()
    result = []
    for element in elements:
        key = element.identifier or normalize_key(element.name)
        if key in seen:
            continue
        seen.add(key)
        result.append(element)
    return result


def _canonicalize_known_identifiers(
    known_identifiers: dict[str, str],
    identifier_names: dict[str, set],
    ontology: FrameOntology,
) -> FrameOntology:
    """Accumulate (identifier -> name) for every resource/entity seen so far in
    the video, unbounded by the sliding description/action history window --
    an object's identifier needs to stay reusable even if it hasn't appeared
    in the last few frames, not just the last N.

    Also detects drifted duplicates: identifier_names keeps every (tokenized)
    name ever used under each identifier, not just the latest one. When an
    element's name matches -- exactly, or by whole-word containment -- a name
    already registered under a *different* identifier, that's the same
    real-world object being given a second identifier; it is rewritten onto
    the existing one instead of being allowed to establish a second one.
    This check runs identically whether the earlier registration happened in
    a previous frame (cross-frame drift, e.g. "bedding" then later "fabric")
    or earlier in the SAME frame's own element list (same-frame duplicate,
    e.g. both "duvet" and "bedding" listed as separate entities in one
    frame), since both are folded into this one incrementally updated
    registry. Once identifiers are canonicalized, elements within the same
    resources/entities list that now share one identifier are collapsed to a
    single entry -- otherwise a same-frame duplicate would still show up
    twice with matching identifiers but different wording.

    Caps at _MAX_KNOWN_IDENTIFIERS (evicting oldest first) only as a safety
    net for unusually long videos.
    """
    resources = list(ontology.resources)
    entities = list(ontology.entities)
    for elements in (resources, entities):
        for index, element in enumerate(elements):
            if not element.identifier or not element.name:
                continue
            new_tokens = _name_tokens(element.name)
            canonical = element.identifier
            for identifier, historical in identifier_names.items():
                if identifier == element.identifier:
                    continue
                if any(_same_referent(new_tokens, existing) for existing in historical):
                    canonical = identifier
                    break
            if canonical != element.identifier:
                elements[index] = dataclasses.replace(element, identifier=canonical)
            identifier_names.setdefault(canonical, set()).add(new_tokens)
            known_identifiers[canonical] = elements[index].name

    ontology = dataclasses.replace(
        ontology,
        resources=_dedupe_by_identifier(resources),
        entities=_dedupe_by_identifier(entities),
    )

    while len(known_identifiers) > _MAX_KNOWN_IDENTIFIERS:
        oldest = next(iter(known_identifiers))
        known_identifiers.pop(oldest)
        identifier_names.pop(oldest, None)
    return ontology


_EATING_KEYWORDS = ("eat", "bite", "bitten", "chew", "swallow")

EATING_CORRECTION = (
    "\n\nNote: on closer inspection, the held food item is NOT touching or "
    "entering the mouth in this frame. Do not describe eating, biting, "
    "chewing, or swallowing -- describe only the hand/food contact and "
    "position that is actually visible."
)


def _mentions_eating(parsed: dict[str, Any]) -> bool:
    text_parts = [str(parsed.get("description", ""))]
    for action in parsed.get("actions") or []:
        if isinstance(action, dict):
            text_parts.append(str(action.get("name", "")))
            text_parts.append(str(action.get("description", "")))
    combined = " ".join(text_parts).lower()
    return any(keyword in combined for keyword in _EATING_KEYWORDS)


def _verify_eating(model: Any, processor: Any, image: Image.Image) -> bool:
    """Ask a narrow, targeted yes/no question to check a claimed eating action
    against the image directly, independent of the main JSON-generation prompt.
    This is a cheap guard against the base model's strong pretrained prior that
    unwrapped food is about to be or is being eaten -- it only fires on the rare
    frames where the main pass already claimed an eating-type action, and it
    never touches the trained model or the main system prompt, so it can't
    regress grounding elsewhere the way a permanent prompt change already has."""
    question = (
        "Answer with exactly one word, yes or no: in this image, is a food "
        "item currently touching or entering the person's mouth?"
    )
    raw = _generate(
        model,
        processor,
        image,
        question,
        max_new_tokens=8,
        system_prompt="Answer the user's question about the image in exactly one word: yes or no.",
    )
    return raw.strip().lower().startswith("y")


def _resolve_model_class():
    import transformers

    for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "Qwen2_5_VLForConditionalGeneration"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    raise ImportError("installed transformers version does not provide a supported vision-language model class")


def _generate(
    model: Any, processor: Any, image: Image.Image, prompt: str, max_new_tokens: int, system_prompt: str = SYSTEM_PROMPT
) -> str:
    import torch

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
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
        text = f"{system_prompt}\n\n{prompt}"
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    prompt_length = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
    generated_only = generated[:, prompt_length:] if prompt_length else generated
    return processor.batch_decode(generated_only, skip_special_tokens=True)[0]

