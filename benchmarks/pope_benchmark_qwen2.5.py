import os
import sys
import re
import json
import argparse
import warnings
from typing import Dict, Tuple

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Hugging Face & Qwen imports
# Hugging Face & Qwen imports
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore", category=UserWarning)

# Add project root for src imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.depth_captioning.depth_blip import DepthBlipCaptioner

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen 2.5-VL on POPE with LDP or LDP+Spatial.")
    parser.add_argument("--dry_run", action="store_true", help="Process 5 samples only.")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL file (auto-generated from mode if not set)")
    parser.add_argument("--dataset", type=str, default="Rajarshi-Roy-research/lmms-lab-POPE")
    parser.add_argument("--split", type=str, default="test_with_depth")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for depth/caption pipeline.")
    parser.add_argument("--mode", type=str, default="ldp_spatial", choices=["ldp", "ldp_spatial"])
    
    # Qwen-specific arguments
    parser.add_argument("--qwen_model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", 
                        help="Hugging Face model path for Qwen 2.5-VL.")
    parser.add_argument("--depth_encoder", type=str, default="vits", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--yolo_model", type=str, default="yolov8n.pt")
    parser.add_argument("--max_new_tokens", type=int, default=10, help="Max tokens for Qwen generation.")
    return parser.parse_args()


def normalize_yes_no(text: str) -> str:
    if not text:
        return "unknown"
    t = _NON_ALNUM_RE.sub(" ", text.strip().lower()).strip()
    first = t.split(" ", 1)[0] if t else ""
    if first in ("yes", "y"):
        return "yes"
    if first in ("no", "n"):
        return "no"
    return "unknown"


def build_ldp_context(depth_captioner: DepthBlipCaptioner, image: Image.Image, mode: str) -> str:
    layer_names = ["Closest", "Farthest", "Mid Range"]
    layer_imgs, masks = depth_captioner.depth_context.make_depth_context_img(
        image, top_threshold=70, bottom_threshold=30, return_masks=True
    )
    spatial = depth_captioner.spatial_analyzer.analyze(np.array(image), masks, max_relations_per_layer=6)

    blocks = []
    for idx, layer_np in enumerate(layer_imgs):
        caption = depth_captioner.captioner.get_caption(layer_np)
        block = f"{layer_names[idx]}: {caption}"
        if mode == "ldp_spatial" and spatial[idx]:
            block += f"\nSpatial Relationships: {spatial[idx]}"
        blocks.append(block)
    return "\n----\n".join(blocks)


def ask_qwen_yes_no(
    model,
    processor,
    image: Image.Image,
    question: str,
    context: str,
    max_new_tokens: int
) -> Tuple[str, str]:
    prompt = (
        "You are answering a visual yes/no question.\n"
        "Use the image as primary evidence. Use the depth context as auxiliary evidence.\n"
        "Return only one token: yes or no.\n\n"
        f"Depth Context:\n{context}\n\n"
        f"Question: {question}"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Process formatting using Qwen's Chat Template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Generate answer
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            temperature=0.0, # Greedy decoding for benchmark consistency
            do_sample=False
        )

    # Trim the prompt from the generated ids
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    raw = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    norm = normalize_yes_no(raw)
    return norm, raw


def main():
    args = parse_args()

    # Auto-generate output filename from mode if not explicitly provided
    if args.output is None:
        args.output = f"data/pope_qwen25vl_{args.mode}_predictions.jsonl"
    print(f"Starting Qwen benchmark mode={args.mode} model={args.qwen_model_path} device={args.device}")
    print(f"Output: {args.output}")
    
    print("\n[1/4] Initializing depth + caption + spatial pipeline...")
    depth_captioner = DepthBlipCaptioner(
        device=torch.device(args.device),
        encoder=args.depth_encoder,
        yolo_model_path=args.yolo_model,
    )

    print(f"\n[2/4] Loading Qwen Model ({args.qwen_model_path})...")
    # Using device_map="auto" to efficiently distribute the model across available GPUs
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.qwen_model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    # Put Qwen in evaluation mode
    qwen_model.eval()
    qwen_processor = AutoProcessor.from_pretrained(args.qwen_model_path)

    print(f"\n[3/4] Loading dataset: {args.dataset} ({args.split})")
    ds = load_dataset(args.dataset, split=args.split)
    num_samples = len(ds)
    if args.dry_run:
        num_samples = min(5, num_samples)
        ds = ds.select(range(num_samples))
        print(f"DRY RUN enabled ({num_samples} samples)")
    else:
        print(f"Loaded {num_samples} samples")

    print("\n[4/4] Running evaluation...")
    y_true, y_pred = [], []

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f_out:
        for idx, row in enumerate(tqdm(ds)):
            image = row["image"]
            question = row["question"]
            gt = str(row["answer"]).strip().lower()
            qid = row.get("question_id", str(idx))
            if image.mode != "RGB":
                image = image.convert("RGB")

            try:
                context = build_ldp_context(depth_captioner, image, args.mode)
            except Exception as exc:
                print(f"[{idx}] context error: {exc}")
                context = "No depth context available."

            try:
                pred, raw = ask_qwen_yes_no(
                    model=qwen_model,
                    processor=qwen_processor,
                    image=image,
                    question=question,
                    context=context,
                    max_new_tokens=args.max_new_tokens,
                )
            except Exception as exc:
                print(f"[{idx}] qwen generation error: {exc}")
                pred, raw = "no", ""

            y_true.append(1 if gt == "yes" else 0)
            y_pred.append(1 if pred == "yes" else 0)

            entry = {
                "question_id": qid,
                "question": question,
                "ground_truth": gt,
                "pred_answer": pred,
                "raw_pred": raw,
                "mode": args.mode,
                "model": args.qwen_model_path,
                "depth_context": context,
            }
            f_out.write(json.dumps(entry) + "\n")
            f_out.flush()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n" + "=" * 50)
    print("   POPE Benchmark Results (Qwen LDP/LDP+Spatial)   ")
    print("=" * 50)
    print(f"Mode:          {args.mode}")
    print(f"Model:         {args.qwen_model_path}")
    print(f"Total Samples: {num_samples}")
    print(f"Accuracy:      {acc:.4f}")
    print(f"Precision:     {prec:.4f}")
    print(f"Recall:        {rec:.4f}")
    print(f"F1 Score:      {f1:.4f}")
    print("=" * 50)
    print(f"Detailed predictions saved to {args.output}")


if __name__ == "__main__":
    main()