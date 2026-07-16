from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from action_ontologies.device import select_device, torch_dtype_for_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for frame ontology extraction")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "rocm", "mps"])
    parser.add_argument("--max-length", type=int, default=4096)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = select_device(args.device)

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Trainer, TrainingArguments

    model_class = _resolve_model_class()
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model = model_class.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype_for_device(device),
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    model.to(device)

    records = _load_records(Path(args.train_jsonl))
    dataset = Dataset.from_list(records)

    def collate(batch):
        texts = []
        images = []
        for record in batch:
            image = Image.open(record["image_path"]).convert("RGB")
            messages = [
                {"role": "system", "content": [{"type": "text", "text": record["messages"][0]["content"]}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": record["messages"][1]["content"]},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": record["messages"][2]["content"]}]},
            ]
            if hasattr(processor, "apply_chat_template"):
                texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
            else:
                texts.append("\n\n".join(message["content"] for message in record["messages"]))
            images.append(image)
        inputs = processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=5,
        save_strategy="epoch",
        remove_unused_columns=False,
        fp16=device == "cuda",
        report_to=[],
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=collate)
    trainer.train()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    return 0


def _load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "image_path" not in record or "messages" not in record:
                raise ValueError(f"invalid record at line {line_number}: expected image_path and messages")
            records.append(record)
    if not records:
        raise ValueError(f"no records found in {path}")
    return records


def _resolve_model_class():
    import transformers

    for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "Qwen2_5_VLForConditionalGeneration"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    raise ImportError("installed transformers version does not provide a supported vision-language model class")


if __name__ == "__main__":
    raise SystemExit(main())
