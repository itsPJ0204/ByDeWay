"""
VSR (Visual Spatial Reasoning) Benchmark — ViLT
=================================================
Evaluates ViLT-VQA on the VSR dataset with three modes:
  1. baseline    — ViLT answers with the original image only
  2. ldp         — ViLT answers with depth-layer images + captions
  3. ldp_spatial — ViLT answers with depth-layer images + spatial analysis

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
from transformers import ViltProcessor, ViltForQuestionAnswering
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

def _compress_for_vilt(text: str, max_words: int = 6) -> str:
    """Aggressively shortens text to fit within ViLT's 40-token limit by removing articles and filler words."""
    if not text:
        return ""
    # Remove common filler words to save tokens
    text = re.sub(r'\b(the|a|an|is|are|of|in|on|at|to)\b', '', text, flags=re.IGNORECASE)
    # Remove extra spaces
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


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


def _predict_vilt(vqa_model, processor, image, prompt):
    """Run ViLT and return (answer_str, confidence_float)."""
    inputs = processor(images=image, text=prompt, return_tensors="pt", truncation=True, max_length=40).to(vqa_model.device)
    with torch.no_grad():
        outputs = vqa_model(**inputs)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        answer = vqa_model.config.id2label[idx]
        confidence = float(logits[0, idx].detach().cpu())
    return answer, confidence


def _vilt_yes_no_scores(vqa_model, processor, image, prompt):
    """
    Get yes/no scores from ViLT's classification head.
    ViLT has a fixed label set from VQAv2 training — 'yes' and 'no' are in it.
    """
    inputs = processor(images=image, text=prompt, return_tensors="pt", truncation=True, max_length=40).to(vqa_model.device)
    with torch.no_grad():
        outputs = vqa_model(**inputs)
        logits = outputs.logits[0]

    # Find yes/no label indices
    label2id = {v.lower(): k for k, v in vqa_model.config.id2label.items()}
    yes_id = label2id.get("yes", None)
    no_id = label2id.get("no", None)

    if yes_id is not None and no_id is not None:
        yes_score = float(logits[yes_id].detach().cpu())
        no_score = float(logits[no_id].detach().cpu())
        pred = "yes" if yes_score >= no_score else "no"
        return pred, {"yes": yes_score, "no": no_score}
    else:
        # Fallback: use top prediction
        idx = logits.argmax(-1).item()
        answer = vqa_model.config.id2label[idx]
        if answer.lower() in ("yes", "y", "true"):
            return "yes", {"answer": answer}
        else:
            return "no", {"answer": answer}


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


# ─── Argument Parsing ────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ViLT on VSR with LDP/Spatial.")
    parser.add_argument("--dry_run", action="store_true", help="Process 5 samples only.")
    parser.add_argument("--num_samples", type=int, default=0, help="Limit samples (0=all).")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL file (auto-generated from mode if not set).")
    parser.add_argument("--dataset", type=str, default="cambridgeltl/vsr_random",
                        help="HF dataset repo (cambridgeltl/vsr_random or cambridgeltl/vsr_zeroshot).")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--vqa_model", type=str, default="dandelin/vilt-b32-finetuned-vqa",
                        help="ViLT VQA checkpoint.")
    parser.add_argument("--depth_encoder", type=str, default="vits",
                        choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--yolo_model", type=str, default="yolov8l-worldv2.pt")
    parser.add_argument("--mode", type=str, default="ldp_spatial",
                        choices=["baseline", "ldp", "ldp_spatial"])
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


# ─── Main ────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    # Auto-generate output filename from mode if not explicitly provided
    if args.output is None:
        args.output = f"data/vsr_vilt_{args.mode}_predictions.jsonl"
    print(f"╔{'═'*56}╗")
    print(f"║  VSR Benchmark (ViLT) — mode={args.mode:<22} ║")
    print(f"╚{'═'*56}╝")
    print(f"  Output: {args.output}")

    # 1. Initialize depth + spatial pipeline
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

    # 2. Initialize ViLT VQA
    print(f"\n[2/3] Initializing ViLT VQA model ({args.vqa_model})...")
    vqa_processor = ViltProcessor.from_pretrained(args.vqa_model)
    vqa_model = ViltForQuestionAnswering.from_pretrained(args.vqa_model).to(args.device)
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
            caption = row.get("caption", "")
            label = int(row.get("label", 0))
            relation = row.get("relation", "")
            image = row.get("image", None)
            image_link = row.get("image_link", "")
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

            # Build question
            question = f'Is this true? "{caption}" Answer yes or no.'

            try:
                if args.mode == "baseline":
                    pred_yn, scores = _vilt_yes_no_scores(vqa_model, vqa_processor, image, question)
                    raw_pred = json.dumps({"pred": pred_yn, "scores": scores})

                elif args.mode == "ldp":
                    layer_imgs, masks = depth_captioner.depth_context.make_depth_context_img(
                        image, top_threshold=70, bottom_threshold=30, return_masks=True
                    )
                    layer_names = ["Closest", "Farthest", "Mid Range"]

                    vote_margin = 0.0
                    details = []

                    # Original image
                    base_pred, base_scores = _vilt_yes_no_scores(vqa_model, vqa_processor, image, question)
                    base_margin = base_scores.get("yes", 0) - base_scores.get("no", 0)
                    vote_margin += base_margin
                    details.append({"view": "original", "pred": base_pred, "margin": base_margin})

                    # Layer votes
                    for li, layer_np in enumerate(layer_imgs):
                        layer_pil = Image.fromarray(layer_np.astype("uint8"))
                        cap = _compress_for_vilt(depth_captioner.captioner.get_caption(layer_np), max_words=10)
                        prompt = f'Q: {question} Ctx: [{layer_names[li]}] {cap}. A:'

                        lp, ls = _vilt_yes_no_scores(vqa_model, vqa_processor, layer_pil, prompt)
                        m = ls.get("yes", 0) - ls.get("no", 0)
                        vote_margin += m * 0.5
                        details.append({"view": layer_names[li], "pred": lp, "margin": m})

                    pred_yn = "yes" if vote_margin >= 0 else "no"
                    raw_pred = json.dumps({"strategy": "ldp_vote", "margin": vote_margin, "details": details})

                else:  # ldp_spatial
                    layer_imgs, masks = depth_captioner.depth_context.make_depth_context_img(
                        image, top_threshold=70, bottom_threshold=30, return_masks=True
                    )
                    layer_names = ["Closest", "Farthest", "Mid Range"]

                    vote_margin = 0.0
                    details = []

                    # Original + spatial
                    spatial_bits = []
                    if vsr_spatial:
                        spatial_bits.append(f"Spatial: {_compress_for_vilt(vsr_spatial, 20)}")
                    spatial_ctx = ". ".join(spatial_bits) if spatial_bits else ""
                    enriched_q = f'Q: {question} Ctx: {spatial_ctx}. A:' if spatial_ctx else question

                    base_pred, base_scores = _vilt_yes_no_scores(vqa_model, vqa_processor, image, enriched_q)
                    base_margin = base_scores.get("yes", 0) - base_scores.get("no", 0)
                    vote_margin += base_margin
                    details.append({"view": "original+spatial", "pred": base_pred, "margin": base_margin})

                    # Mid Range layer + spatial
                    li = 2
                    layer_np = layer_imgs[li]
                    layer_pil = Image.fromarray(layer_np.astype("uint8"))
                    cap = _compress_for_vilt(depth_captioner.captioner.get_caption(layer_np), max_words=10)
                    layer_rel = depth_captioner.spatial_analyzer.analyze(
                        np.array(image), masks, max_relations_per_layer=4
                    )[li]

                    ctx_bits = [f"[{layer_names[li]}] {cap}."]
                    if layer_rel:
                        ctx_bits.append(f"[Sp] {_compress_for_vilt(layer_rel, 6)}.")
                    if vsr_spatial:
                        ctx_bits.append(f"[VSR] {_compress_for_vilt(vsr_spatial, 6)}.")
                    small_ctx = " ".join(ctx_bits)

                    prompt = f'Q: {question} Ctx: {small_ctx} A:'
                    lp, ls = _vilt_yes_no_scores(vqa_model, vqa_processor, layer_pil, prompt)
                    m = ls.get("yes", 0) - ls.get("no", 0)
                    vote_margin += m * 0.5
                    details.append({"view": "MidRange+spatial", "pred": lp, "margin": m})

                    pred_yn = "yes" if vote_margin >= 0 else "no"
                    raw_pred = json.dumps({
                        "strategy": "ldp_spatial_vote", "margin": vote_margin, "details": details
                    })

            except Exception as e:
                print(f"[{idx}] prediction error: {e}")
                pred_yn = "no"
                raw_pred = f"error: {e}"

            pred_label = 1 if pred_yn == "yes" else 0

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
    print(f"  VSR Benchmark Results (ViLT) — Mode: {args.mode}")
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
