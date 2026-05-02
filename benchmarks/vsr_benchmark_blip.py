"""
VSR (Visual Spatial Reasoning) Benchmark — BLIP
================================================
Evaluates BLIP-VQA on the VSR dataset with three modes:
  1. baseline  — BLIP answers with the original image only
  2. ldp       — BLIP answers with depth-layer images + captions
  3. ldp_spatial — BLIP answers with depth-layer images + spatial analysis

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

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BlipProcessor, BlipForQuestionAnswering
import requests
from io import BytesIO

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
    if first in ("yes", "y", "true", "1"):
        return "true"
    if first in ("no", "n", "false", "0"):
        return "false"
    # Check if any word is yes/no/true/false
    for w in t.split():
        if w in ("yes", "true"):
            return "true"
        if w in ("no", "false"):
            return "false"
    return "unknown"


def _score_yes_no_from_first_step(vqa_model, tokenizer, inputs):
    """
    Score yes vs no from the first generated token logits.
    Returns (prediction, scores_dict).
    """
    yes_id = tokenizer("yes", add_special_tokens=False).input_ids[0]
    no_id = tokenizer("no", add_special_tokens=False).input_ids[0]

    with torch.no_grad():
        gen = vqa_model.generate(
            **inputs,
            max_new_tokens=1,
            num_beams=5,
            return_dict_in_generate=True,
            output_scores=True,
        )

    step_logits = gen.scores[0][0]
    yes_score = float(step_logits[yes_id].detach().cpu())
    no_score = float(step_logits[no_id].detach().cpu())
    pred = "yes" if yes_score >= no_score else "no"
    return pred, {"yes": yes_score, "no": no_score}


def build_ldp_context(depth_captioner: DepthBlipCaptioner, image: Image.Image, mode: str) -> str:
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


def build_vsr_spatial_context(depth_captioner: DepthBlipCaptioner, image: Image.Image) -> str:
    """Build enhanced VSR spatial context using analyze_vsr."""
    image_array = np.array(image)
    # Get raw depth map for depth-based relations
    try:
        depth_map = depth_captioner.depth_context.predict_depth(image_array[:, :, ::-1])
    except Exception:
        depth_map = None

    spatial_desc = depth_captioner.spatial_analyzer.analyze_vsr_for_caption(
        image_array, depth_map=depth_map
    )
    return spatial_desc


# ─── Argument Parsing ────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BLIP on VSR with LDP/Spatial.")
    parser.add_argument("--dry_run", action="store_true", help="Process 5 samples only.")
    parser.add_argument("--num_samples", type=int, default=0, help="Limit samples (0=all).")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL file (auto-generated from mode if not set).")
    parser.add_argument("--dataset", type=str, default="cambridgeltl/vsr_random",
                        help="HF dataset repo (cambridgeltl/vsr_random or cambridgeltl/vsr_zeroshot).")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split (train/dev/test).")
    parser.add_argument("--vqa_model", type=str, default="Salesforce/blip-vqa-base",
                        help="BLIP VQA checkpoint.")
    parser.add_argument("--depth_encoder", type=str, default="vits",
                        choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--yolo_model", type=str, default="yolov8l-worldv2.pt")
    parser.add_argument("--mode", type=str, default="ldp_spatial",
                        choices=["baseline", "ldp", "ldp_spatial"],
                        help="Evaluation mode.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


# ─── Main ────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    # Auto-generate output filename from mode if not explicitly provided
    if args.output is None:
        args.output = f"data/vsr_blip_{args.mode}_predictions.jsonl"
    print(f"╔{'═'*56}╗")
    print(f"║  VSR Benchmark (BLIP) — mode={args.mode:<22} ║")
    print(f"╚{'═'*56}╝")
    print(f"  Output: {args.output}")

    # 1. Initialize depth + spatial pipeline (skip for baseline)
    depth_captioner = None
    if args.mode != "baseline":
        print("\n[1/3] Initializing DepthBlipCaptioner (Depth + YOLO + BLIP Caption)...")
        depth_captioner = DepthBlipCaptioner(
            device=torch.device(args.device),
            encoder=args.depth_encoder,
            yolo_model_path=args.yolo_model,
        )
    else:
        print("\n[1/3] Baseline mode — skipping depth/spatial pipeline.")

    # 2. Initialize BLIP VQA
    print(f"\n[2/3] Initializing BLIP VQA model ({args.vqa_model})...")
    vqa_processor = BlipProcessor.from_pretrained(args.vqa_model)
    vqa_model = BlipForQuestionAnswering.from_pretrained(args.vqa_model).to(args.device)
    vqa_model.eval()

    print(f"\n[3/3] Loading dataset: {args.dataset} (split: {args.split})")
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
    y_true = []
    y_pred = []
    per_category = defaultdict(lambda: {"y_true": [], "y_pred": []})

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Configure session with retries to prevent TCP/DNS exhaustion
    session = requests.Session()
    retries = __import__('urllib3').util.retry.Retry(total=5, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    adapter = requests.adapters.HTTPAdapter(max_retries=retries, pool_connections=100, pool_maxsize=100)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    with open(args.output, "w", encoding="utf-8") as f_out:
        for idx, row in enumerate(tqdm(ds, desc="Evaluating")):
            # VSR fields
            caption = row.get("caption", "")
            label = int(row.get("label", 0))  # 1=True, 0=False
            relation = row.get("relation", "")
            image = row.get("image", None)
            image_link = row.get("image_link", "")

            # Determine the category
            category = RELATION_TO_CATEGORY.get(relation, "unallocated")

            # Load image from COCO URL
            if image_link:
                try:
                    resp = session.get(image_link, timeout=15)
                    resp.raise_for_status()
                    image = Image.open(BytesIO(resp.content))
                except Exception as e:
                    print(f"[{idx}] Failed to load image from {image_link}: {e}")
                    continue
            else:
                print(f"[{idx}] No image_link available, skipping.")
                continue

            if image.mode != "RGB":
                image = image.convert("RGB")

            # Build context based on mode
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

            # Build the question for BLIP
            # VSR is a true/false classification: "Is this caption true for the image?"
            question = f'Is this statement true or false? "{caption}" Answer yes or no.'

            try:
                if args.mode == "baseline":
                    # Baseline: question + original image
                    inputs = vqa_processor(
                        images=image, text=question,
                        return_tensors="pt", truncation=True, max_length=96
                    ).to(args.device)
                    pred_yn, scores = _score_yes_no_from_first_step(
                        vqa_model, vqa_processor.tokenizer, inputs
                    )
                    raw_pred = json.dumps({"pred": pred_yn, "scores": scores})

                elif args.mode == "ldp":
                    # LDP: depth-layer voting
                    layer_imgs, masks = depth_captioner.depth_context.make_depth_context_img(
                        image, top_threshold=70, bottom_threshold=30, return_masks=True
                    )
                    layer_names = ["Closest", "Farthest", "Mid Range"]

                    vote_margin = 0.0
                    details = []

                    # Original image vote
                    base_inputs = vqa_processor(
                        images=image, text=question,
                        return_tensors="pt", truncation=True, max_length=96
                    )
                    base_inputs = {k: v.to(args.device) for k, v in base_inputs.items()}
                    base_pred, base_scores = _score_yes_no_from_first_step(
                        vqa_model, vqa_processor.tokenizer, base_inputs
                    )
                    base_margin = base_scores["yes"] - base_scores["no"]
                    vote_margin += base_margin
                    details.append({"view": "original", "pred": base_pred, "margin": base_margin})

                    # Layer votes
                    for li, layer_np in enumerate(layer_imgs):
                        layer_pil = Image.fromarray(layer_np.astype("uint8"))
                        layer_caption = _shorten(depth_captioner.captioner.get_caption(layer_np), max_words=12)
                        prompt = f'{question} Context: [{layer_names[li]}] {layer_caption}.'

                        layer_inputs = vqa_processor(
                            images=layer_pil, text=prompt,
                            return_tensors="pt", truncation=True, max_length=128
                        )
                        layer_inputs = {k: v.to(args.device) for k, v in layer_inputs.items()}
                        lp, ls = _score_yes_no_from_first_step(
                            vqa_model, vqa_processor.tokenizer, layer_inputs
                        )
                        m = ls["yes"] - ls["no"]
                        vote_margin += m * 0.5  # weight layer votes lower
                        details.append({"view": layer_names[li], "pred": lp, "margin": m})

                    pred_yn = "yes" if vote_margin >= 0 else "no"
                    raw_pred = json.dumps({"strategy": "ldp_vote", "margin": vote_margin, "details": details})

                else:
                    # LDP+Spatial: depth layers + enhanced spatial context
                    layer_imgs, masks = depth_captioner.depth_context.make_depth_context_img(
                        image, top_threshold=70, bottom_threshold=30, return_masks=True
                    )
                    layer_names = ["Closest", "Farthest", "Mid Range"]

                    vote_margin = 0.0
                    details = []

                    # Original image + spatial context
                    spatial_ctx_bits = []
                    if vsr_spatial:
                        spatial_ctx_bits.append(f"Detected spatial relations: {_shorten(vsr_spatial, 30)}")
                    if context:
                        spatial_ctx_bits.append(f"Depth: {_shorten(context, 20)}")
                    spatial_ctx = ". ".join(spatial_ctx_bits) if spatial_ctx_bits else ""

                    enriched_q = question
                    if spatial_ctx:
                        enriched_q = f'{question} Context: {spatial_ctx}.'

                    base_inputs = vqa_processor(
                        images=image, text=enriched_q,
                        return_tensors="pt", truncation=True, max_length=160
                    )
                    base_inputs = {k: v.to(args.device) for k, v in base_inputs.items()}
                    base_pred, base_scores = _score_yes_no_from_first_step(
                        vqa_model, vqa_processor.tokenizer, base_inputs
                    )
                    base_margin = base_scores["yes"] - base_scores["no"]
                    vote_margin += base_margin
                    details.append({"view": "original+spatial", "pred": base_pred, "margin": base_margin})

                    # Mid Range layer with spatial
                    li = 2  # Mid Range
                    layer_np = layer_imgs[li]
                    layer_pil = Image.fromarray(layer_np.astype("uint8"))
                    layer_caption = _shorten(depth_captioner.captioner.get_caption(layer_np), max_words=12)
                    layer_spatial = depth_captioner.spatial_analyzer.analyze(
                        np.array(image), masks, max_relations_per_layer=4
                    )[li]

                    ctx_bits = [f"[{layer_names[li]}] {layer_caption}."]
                    if layer_spatial:
                        ctx_bits.append(f"[Sp] {_shorten(layer_spatial, 16)}.")
                    if vsr_spatial:
                        ctx_bits.append(f"[VSR] {_shorten(vsr_spatial, 20)}.")
                    small_ctx = " ".join(ctx_bits)

                    layer_prompt = f'{question} Context: {small_ctx}'
                    layer_inputs = vqa_processor(
                        images=layer_pil, text=layer_prompt,
                        return_tensors="pt", truncation=True, max_length=160
                    )
                    layer_inputs = {k: v.to(args.device) for k, v in layer_inputs.items()}
                    lp, ls = _score_yes_no_from_first_step(
                        vqa_model, vqa_processor.tokenizer, layer_inputs
                    )
                    m = ls["yes"] - ls["no"]
                    vote_margin += m * 0.5
                    details.append({"view": f"MidRange+spatial", "pred": lp, "margin": m})

                    pred_yn = "yes" if vote_margin >= 0 else "no"
                    raw_pred = json.dumps({
                        "strategy": "ldp_spatial_vote", "margin": vote_margin, "details": details
                    })

            except Exception as e:
                print(f"[{idx}] prediction error: {e}")
                pred_yn = "no"
                raw_pred = f"error: {e}"

            # Convert yes/no -> true/false for VSR
            pred_label = 1 if pred_yn == "yes" else 0
            gt_label = label

            y_true.append(gt_label)
            y_pred.append(pred_label)
            per_category[category]["y_true"].append(gt_label)
            per_category[category]["y_pred"].append(pred_label)

            entry = {
                "idx": idx,
                "caption": caption,
                "relation": relation,
                "category": category,
                "ground_truth": gt_label,
                "pred_label": pred_label,
                "pred_yn": pred_yn,
                "raw_pred": raw_pred,
                "mode": args.mode,
            }
            f_out.write(json.dumps(entry) + "\n")
            f_out.flush()

    # ── Results ──────────────────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n{'='*60}")
    print(f"  VSR Benchmark Results (BLIP) — Mode: {args.mode}")
    print(f"{'='*60}")
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
