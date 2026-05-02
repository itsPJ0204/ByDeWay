"""
VSR (Visual Spatial Reasoning) Benchmark — Qwen 2.5-VL
=======================================================
Evaluates Qwen 2.5-VL on the VSR dataset with three modes:
  1. baseline    — Qwen answers with the original image only
  2. ldp         — Qwen answers with depth context (LDP captions)
  3. ldp_spatial — Qwen answers with LDP + enhanced spatial analysis

The VSR task: given an image and a caption describing a spatial relation
between two objects, predict True (1) or False (0).

Dataset: cambridgeltl/vsr_random (random split, plain JSONL — no loading script)
"""

import os
import sys
import json
import re
import argparse
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import requests as http_requests
from io import BytesIO

# Hugging Face & Qwen imports
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore", category=UserWarning)

# Add project root for src imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.depth_captioning.depth_blip import DepthBlipCaptioner
from src.depth_captioning.spatial_analysis import RELATION_TO_CATEGORY, VSR_CATEGORIES

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


# ─── Utility Functions ──────────────────────────────────────────────

def _shorten(text: str, max_words: int = 18) -> str:
    if not text:
        return ""
    words = str(text).strip().split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + " ..."


def parse_vsr_caption(caption, relation):
    """Extract subject and object names from a VSR caption.

    VSR captions follow patterns like:
      "The {subj} is {relation} the {obj}."
      "The {subj} {relation} the {obj}."

    Returns:
        tuple: (subject_name, object_name) or (None, None) if parsing fails.
    """
    text = caption.strip().rstrip(".")
    for marker in [f" is {relation} the ", f" {relation} the "]:
        if marker in text:
            left, right = text.split(marker, 1)
            subj = left[4:] if left.startswith("The ") else left
            return subj.strip(), right.strip()
    return None, None


def normalize_true_false(text: str) -> str:
    """Normalize model output to 'true' or 'false'."""
    if not text:
        return "unknown"
    t = _NON_ALNUM_RE.sub(" ", text.strip().lower()).strip()
    first = t.split(" ", 1)[0] if t else ""
    if first in ("yes", "y", "true", "1", "correct"):
        return "true"
    if first in ("no", "n", "false", "0", "incorrect", "wrong"):
        return "false"
    for w in t.split():
        if w in ("yes", "true", "correct"):
            return "true"
        if w in ("no", "false", "incorrect", "wrong"):
            return "false"
    return "unknown"


def build_ldp_context(depth_captioner, image, mode):
    """Build LDP or LDP+Spatial context string."""
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


def build_vsr_spatial_context(depth_captioner, image):
    """Build enhanced VSR spatial context."""
    image_array = np.array(image)
    try:
        depth_map = depth_captioner.depth_context.predict_depth(image_array[:, :, ::-1])
    except Exception:
        depth_map = None
    return depth_captioner.spatial_analyzer.analyze_vsr_for_caption(image_array, depth_map=depth_map)


def ask_qwen_true_false(
    model, processor, image, caption, context, vsr_spatial, mode, max_new_tokens
):
    """
    Ask Qwen whether the spatial caption is true or false for the given image.
    Returns (normalized_prediction, raw_text).
    """
    if mode == "baseline":
        prompt = (
            "You are evaluating whether a spatial description of an image is true or false.\n"
            "Look at the image carefully and determine if the following statement is correct.\n"
            "Answer with only one word: true or false.\n\n"
            f'Statement: "{caption}"'
        )
    elif mode == "ldp":
        prompt = (
            "You are evaluating whether a spatial description of an image is true or false.\n"
            "Use the image as primary evidence. Use the depth context as auxiliary evidence.\n"
            "Answer with only one word: true or false.\n\n"
            f"Depth Context:\n{context}\n\n"
            f'Statement: "{caption}"'
        )
    else:  # ldp_spatial
        spatial_section = ""
        if vsr_spatial:
            spatial_section = f"\nDetected Spatial Relations:\n{vsr_spatial}\n"

        prompt = (
            "You are evaluating whether a spatial description of an image is true or false.\n"
            "Use the image as primary evidence. Use the depth context and detected spatial relations "
            "as auxiliary evidence to help you understand spatial arrangements.\n"
            "Answer with only one word: true or false.\n\n"
            f"Depth Context:\n{context}\n"
            f"{spatial_section}\n"
            f'Statement: "{caption}"'
        )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image, "max_pixels": 313600},
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
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    raw = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    norm = normalize_true_false(raw)
    return norm, raw


# ─── Argument Parsing ────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen 2.5-VL on VSR with LDP/Spatial.")
    parser.add_argument("--dry_run", action="store_true", help="Process 5 samples only.")
    parser.add_argument("--num_samples", type=int, default=0, help="Limit samples (0=all).")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL file (auto-generated from mode if not set).")
    parser.add_argument("--dataset", type=str, default="cambridgeltl/vsr_random",
                        help="HF dataset repo (cambridgeltl/vsr_random or cambridgeltl/vsr_zeroshot).")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for depth/caption pipeline.")
    parser.add_argument("--mode", type=str, default="ldp_spatial",
                        choices=["baseline", "ldp", "ldp_spatial"])
    # Qwen-specific
    parser.add_argument("--qwen_model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--depth_encoder", type=str, default="vits",
                        choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--yolo_model", type=str, default="yolov8l-worldv2.pt")
    parser.add_argument("--max_new_tokens", type=int, default=10)
    return parser.parse_args()


# ─── Main ────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    # Auto-generate output filename from mode if not explicitly provided
    if args.output is None:
        args.output = f"data/vsr_qwen25vl_{args.mode}_predictions.jsonl"
    print(f"╔{'═'*60}╗")
    print(f"║  VSR Benchmark (Qwen 2.5-VL) — mode={args.mode:<20} ║")
    print(f"╚{'═'*60}╝")
    print(f"  Output: {args.output}")

    # 1. Initialize depth + spatial pipeline
    depth_captioner = None
    if args.mode != "baseline":
        print("\n[1/4] Initializing depth + caption + spatial pipeline...")
        depth_captioner = DepthBlipCaptioner(
            device=torch.device(args.device),
            encoder=args.depth_encoder,
            yolo_model_path=args.yolo_model,
        )
    else:
        print("\n[1/4] Baseline mode — skipping depth/spatial pipeline.")

    # 2. Load Qwen
    print(f"\n[2/4] Loading Qwen Model ({args.qwen_model_path})...")
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.qwen_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    qwen_model.eval()
    qwen_processor = AutoProcessor.from_pretrained(args.qwen_model_path)

    print(f"\n[3/4] Loading dataset: {args.dataset} (split: {args.split})")
    try:
        data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
        ds = load_dataset(args.dataset, data_files=data_files, split=args.split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    num_samples = len(ds)
    if args.dry_run:
        num_samples = min(5, num_samples)
        ds = ds.select(range(num_samples))
        print(f"DRY RUN enabled ({num_samples} samples)")
    elif args.num_samples > 0:
        num_samples = min(args.num_samples, num_samples)
        ds = ds.select(range(num_samples))
        print(f"Limited to {num_samples} samples")
    else:
        print(f"Loaded {num_samples} samples")

    # ── Evaluation Loop ──────────────────────────────────────────
    print("\n[4/4] Running evaluation...")
    y_true = []
    y_pred = []
    per_category = defaultdict(lambda: {"y_true": [], "y_pred": []})

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Configure session with retries to prevent TCP/DNS exhaustion
    session = http_requests.Session()
    retries = __import__('urllib3').util.retry.Retry(total=5, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    adapter = http_requests.adapters.HTTPAdapter(max_retries=retries, pool_connections=100, pool_maxsize=100)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    # ── Async image prefetch helper ─────────────────────────────
    def _fetch_image(row_with_idx):
        """Download a single image in a background thread. Returns (idx, row, PIL.Image|None)."""
        i, r = row_with_idx
        link = r.get("image_link", "")
        if not link:
            return i, r, None
        try:
            resp = session.get(link, timeout=15)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content))
            if img.mode != "RGB":
                img = img.convert("RGB")
            return i, r, img
        except Exception:
            return i, r, None

    prefetch_workers = 8  # parallel HTTP downloads

    with open(args.output, "w", encoding="utf-8") as f_out:
      with ThreadPoolExecutor(max_workers=prefetch_workers) as pool:
        # Submit all downloads eagerly; iterate results in order
        futures = pool.map(_fetch_image, enumerate(ds))
        for idx, row, image in tqdm(futures, desc="Evaluating", total=len(ds)):
            caption = row.get("caption", "")
            label = int(row.get("label", 0))
            relation = row.get("relation", "")
            image_link = row.get("image_link", "")
            category = RELATION_TO_CATEGORY.get(relation, "unallocated")

            if image is None:
                print(f"[{idx}] Failed to load image from {image_link}, skipping.")
                continue

            # Build context
            context = ""
            vsr_spatial = ""
            if args.mode != "baseline" and depth_captioner is not None:
                # Set YOLO-World target classes for this sample's objects
                subj, obj = parse_vsr_caption(caption, relation)
                if subj and obj:
                    depth_captioner.spatial_analyzer.set_classes([subj, obj])

                try:
                    context = build_ldp_context(depth_captioner, image, args.mode)
                except Exception as e:
                    print(f"[{idx}] LDP context error: {e}")
                    context = ""

                if args.mode == "ldp_spatial":
                    try:
                        vsr_spatial = build_vsr_spatial_context(depth_captioner, image)
                    except Exception as e:
                        print(f"[{idx}] VSR spatial error: {e}")
                        vsr_spatial = ""

            # Ask Qwen
            try:
                norm_pred, raw_pred = ask_qwen_true_false(
                    model=qwen_model,
                    processor=qwen_processor,
                    image=image,
                    caption=caption,
                    context=context,
                    vsr_spatial=vsr_spatial,
                    mode=args.mode,
                    max_new_tokens=args.max_new_tokens,
                )
            except Exception as e:
                print(f"[{idx}] Qwen generation error: {e}")
                norm_pred, raw_pred = "false", f"error: {e}"

            # Convert: true->1, false->0
            if norm_pred == "true":
                pred_label = 1
            elif norm_pred == "false":
                pred_label = 0
            else:
                pred_label = 0  # default to false for unknown

            y_true.append(label)
            y_pred.append(pred_label)
            per_category[category]["y_true"].append(label)
            per_category[category]["y_pred"].append(pred_label)

            entry = {
                "idx": idx,
                "caption": caption,
                "relation": relation,
                "category": category,
                "ground_truth": label,
                "pred_label": pred_label,
                "norm_pred": norm_pred,
                "raw_pred": raw_pred,
                "mode": args.mode,
                "model": args.qwen_model_path,
                "depth_context": context[:200] if context else "",
                "vsr_spatial": vsr_spatial[:200] if vsr_spatial else "",
            }
            f_out.write(json.dumps(entry) + "\n")
            f_out.flush()

    # ── Results ──────────────────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n{'='*60}")
    print(f"  VSR Benchmark Results (Qwen 2.5-VL) — Mode: {args.mode}")
    print(f"{'='*60}")
    print(f"  Model:          {args.qwen_model_path}")
    print(f"  Total Samples:  {len(y_true)}")
    print(f"  Accuracy:       {acc:.4f}")
    print(f"  Precision:      {prec:.4f}")
    print(f"  Recall:         {rec:.4f}")
    print(f"  F1 Score:       {f1:.4f}")
    print(f"{'='*60}")

    # Per-category breakdown
    print(f"\n{'─'*60}")
    print(f"  Per-Category Breakdown")
    print(f"{'─'*60}")
    print(f"  {'Category':<16} {'N':>5} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print(f"  {'─'*52}")

    for cat in sorted(per_category.keys()):
        data = per_category[cat]
        yt = data["y_true"]
        yp = data["y_pred"]
        n = len(yt)
        if n == 0:
            continue
        c_acc = accuracy_score(yt, yp)
        c_prec = precision_score(yt, yp, zero_division=0)
        c_rec = recall_score(yt, yp, zero_division=0)
        c_f1 = f1_score(yt, yp, zero_division=0)
        print(f"  {cat:<16} {n:>5} {c_acc:>7.4f} {c_prec:>7.4f} {c_rec:>7.4f} {c_f1:>7.4f}")

    print(f"{'─'*60}")
    print(f"  Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
