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
from sklearn.metrics import accuracy_score

# Hugging Face & Qwen imports
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore", category=UserWarning)

# Add project root for src imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.depth_captioning.depth_blip import DepthBlipCaptioner

def normalize_word(word: str) -> str:
    if not word:
        return ""
    word = re.sub(r'[^a-zA-Z0-9\s]', '', word)
    return word.strip().lower()

def exact_match(prediction: str, ground_truth: str) -> bool:
    p_norm = normalize_word(prediction)
    g_norm = normalize_word(ground_truth)
    if not p_norm or not g_norm:
        return False
    return p_norm in g_norm or g_norm in p_norm

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen 2.5-VL on GQA with LDP or LDP+Spatial.")
    parser.add_argument("--dry_run", action="store_true", help="Process 5 samples only.")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL file (auto-generated from mode if not set)")
    parser.add_argument("--dataset", type=str, default="Rajarshi-Roy-research/GQA-dataset-150")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for depth/caption pipeline.")
    parser.add_argument("--mode", type=str, default="ldp_spatial", choices=["ldp", "ldp_spatial"])
    
    # Qwen-specific arguments
    parser.add_argument("--qwen_model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--depth_encoder", type=str, default="vits", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--yolo_model", type=str, default="yolov8n.pt")
    parser.add_argument("--max_new_tokens", type=int, default=20, help="Max tokens for Qwen generation.")
    return parser.parse_args()


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


def ask_qwen_generative(
    model,
    processor,
    image: Image.Image,
    question: str,
    context: str,
    max_new_tokens: int
) -> str:
    prompt = (
        "You are answering a visual question.\n"
        "Provide a very short and concise answer (1-3 words).\n"
        "Use the image as primary evidence. Use the depth context as auxiliary evidence.\n\n"
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

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    raw = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    return raw


def main():
    args = parse_args()

    # Auto-generate output filename from mode if not explicitly provided
    if args.output is None:
        args.output = f"data/gqa_qwen25vl_{args.mode}_predictions.jsonl"
    print(f"Starting Qwen GQA benchmark mode={args.mode} model={args.qwen_model_path} device={args.device}")
    print(f"Output: {args.output}")
    
    print("\n[1/4] Initializing depth + caption + spatial pipeline...")
    depth_captioner = DepthBlipCaptioner(
        device=torch.device(args.device),
        encoder=args.depth_encoder,
        yolo_model_path=args.yolo_model,
    )

    print(f"\n[2/4] Loading Qwen Model ({args.qwen_model_path})...")
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.qwen_model_path,
        torch_dtype="auto",
        device_map="auto"
    )
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
    correct = 0
    total = 0

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f_out:
        for idx, row in enumerate(tqdm(ds)):
            image = row["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")

            try:
                context = build_ldp_context(depth_captioner, image, args.mode)
            except Exception as exc:
                print(f"[{idx}] context error: {exc}")
                context = "No depth context available."

            qa_list = row.get('qa', [])
            if not isinstance(qa_list, list):
                if isinstance(qa_list, dict):
                    qa_list = [qa_list]
                else:
                    continue

            for qa_idx, qa_item in enumerate(qa_list):
                question = str(qa_item.get('question', ''))
                gt = str(qa_item.get('answer', '')).strip()
                qid = str(qa_item.get('question_id', f"{idx}_{qa_idx}"))

                try:
                    raw_pred = ask_qwen_generative(
                        model=qwen_model,
                        processor=qwen_processor,
                        image=image,
                        question=question,
                        context=context,
                        max_new_tokens=args.max_new_tokens,
                    )
                except Exception as exc:
                    print(f"[{qid}] qwen generation error: {exc}")
                    raw_pred = ""

                is_correct = exact_match(raw_pred, gt)
                if is_correct:
                    correct += 1
                total += 1

                entry = {
                    "question_id": qid,
                    "question": question,
                    "ground_truth": gt,
                    "pred_answer": raw_pred,
                    "is_correct": is_correct,
                    "mode": args.mode,
                    "model": args.qwen_model_path,
                    "depth_context": context,
                }
                f_out.write(json.dumps(entry) + "\n")
                f_out.flush()

    accuracy = correct / total if total > 0 else 0

    print("\n" + "=" * 50)
    print("   GQA Benchmark Results (Qwen LDP/LDP+Spatial)   ")
    print("=" * 50)
    print(f"Mode:          {args.mode}")
    print(f"Model:         {args.qwen_model_path}")
    print(f"Total Samples: {total}")
    print(f"Correct:       {correct}")
    print(f"Accuracy:      {accuracy:.4f}")
    print("=" * 50)
    print(f"Detailed predictions saved to {args.output}")


if __name__ == "__main__":
    main()
