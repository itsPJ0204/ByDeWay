"""
Detector Comparison: YOLOv8n vs YOLO-World on VSR samples
==========================================================
Demonstrates that YOLO-World with per-sample class targeting detects
the RELEVANT objects from VSR captions, while YOLOv8n often detects
irrelevant objects or misses the target objects entirely.

Runs on CPU — no VLM required.

Usage:
    python benchmarks/compare_detectors.py --num_samples 30
"""

import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
import requests
from io import BytesIO

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.depth_captioning.spatial_analysis import SpatialAnalyzer, RELATION_TO_CATEGORY


def parse_vsr_caption(caption, relation):
    """Extract subject and object names from a VSR caption."""
    text = caption.strip().rstrip(".")
    for marker in [f" is {relation} the ", f" {relation} the "]:
        if marker in text:
            left, right = text.split(marker, 1)
            subj = left[4:] if left.startswith("The ") else left
            return subj.strip(), right.strip()
    return None, None


def get_detections(analyzer, image_array):
    """Run object detection and return list of (label, confidence) tuples."""
    results = analyzer.model(image_array, verbose=False, stream=False)
    result = results[0]
    names = result.names
    detections = []
    if result.boxes:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = names[cls]
            detections.append((label, conf))
    return detections


def check_relevance(detections, target_subj, target_obj):
    """Check if any detection matches the target subject or object."""
    detected_labels = set(d[0].lower() for d in detections)
    subj_lower = target_subj.lower()
    obj_lower = target_obj.lower()

    subj_found = any(subj_lower in label or label in subj_lower for label in detected_labels)
    obj_found = any(obj_lower in label or label in obj_lower for label in detected_labels)

    return subj_found, obj_found


def main():
    parser = argparse.ArgumentParser(description="Compare YOLOv8n vs YOLO-World detections on VSR.")
    parser.add_argument("--num_samples", type=int, default=30, help="Number of samples to compare.")
    parser.add_argument("--dataset", type=str, default="cambridgeltl/vsr_random")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output", type=str, default="data/detector_comparison.jsonl")
    args = parser.parse_args()

    # Load both detectors
    print("=" * 70)
    print("  Detector Comparison: YOLOv8n vs YOLO-World (large v2)")
    print("=" * 70)

    print("\n[1/3] Loading YOLOv8n (baseline — COCO 80 classes)...")
    analyzer_yolov8n = SpatialAnalyzer(model_path="yolov8n.pt")

    print("\n[2/3] Loading YOLO-World Large v2 (open-vocabulary)...")
    analyzer_world = SpatialAnalyzer(model_path="yolov8l-worldv2.pt")

    print(f"\n[3/3] Loading dataset: {args.dataset} (split: {args.split})")
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    ds = load_dataset(args.dataset, data_files=data_files, split=args.split)
    ds = ds.select(range(min(args.num_samples, len(ds))))
    print(f"  Selected {len(ds)} samples\n")

    # Setup session
    session = requests.Session()
    retries = __import__('urllib3').util.retry.Retry(total=3, backoff_factor=0.5)
    adapter = requests.adapters.HTTPAdapter(max_retries=retries)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    # Tracking stats
    yolov8n_stats = {"subj_found": 0, "obj_found": 0, "both_found": 0, "total_detections": 0}
    world_stats = {"subj_found": 0, "obj_found": 0, "both_found": 0, "total_detections": 0}
    valid_samples = 0

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f_out:
        for idx, row in enumerate(tqdm(ds, desc="Comparing detectors")):
            caption = row.get("caption", "")
            relation = row.get("relation", "")
            image_link = row.get("image_link", "")
            label = int(row.get("label", 0))
            category = RELATION_TO_CATEGORY.get(relation, "unallocated")

            subj, obj = parse_vsr_caption(caption, relation)
            if not subj or not obj:
                continue

            # Load image
            try:
                resp = session.get(image_link, timeout=15)
                resp.raise_for_status()
                image = Image.open(BytesIO(resp.content))
                if image.mode != "RGB":
                    image = image.convert("RGB")
            except Exception:
                continue

            image_array = np.array(image)
            valid_samples += 1

            # --- YOLOv8n detection (fixed COCO classes) ---
            dets_v8n = get_detections(analyzer_yolov8n, image_array)
            v8n_subj, v8n_obj = check_relevance(dets_v8n, subj, obj)

            # --- YOLO-World detection (targeted classes) ---
            analyzer_world.set_classes([subj, obj])
            dets_world = get_detections(analyzer_world, image_array)
            w_subj, w_obj = check_relevance(dets_world, subj, obj)

            # Update stats
            yolov8n_stats["subj_found"] += int(v8n_subj)
            yolov8n_stats["obj_found"] += int(v8n_obj)
            yolov8n_stats["both_found"] += int(v8n_subj and v8n_obj)
            yolov8n_stats["total_detections"] += len(dets_v8n)

            world_stats["subj_found"] += int(w_subj)
            world_stats["obj_found"] += int(w_obj)
            world_stats["both_found"] += int(w_subj and w_obj)
            world_stats["total_detections"] += len(dets_world)

            # Print per-sample comparison
            v8n_labels = [f"{d[0]}({d[1]:.2f})" for d in dets_v8n]
            w_labels = [f"{d[0]}({d[1]:.2f})" for d in dets_world]

            status_v8n = "✓" if (v8n_subj and v8n_obj) else ("◐" if (v8n_subj or v8n_obj) else "✗")
            status_w = "✓" if (w_subj and w_obj) else ("◐" if (w_subj or w_obj) else "✗")

            print(f"\n{'─'*70}")
            print(f"  [{idx}] Caption: \"{caption}\"")
            print(f"       Target objects: [{subj}] and [{obj}]")
            print(f"  YOLOv8n  {status_v8n}  → {', '.join(v8n_labels[:8]) or '(none)'}")
            print(f"  YOLOWorld {status_w}  → {', '.join(w_labels[:8]) or '(none)'}")

            entry = {
                "idx": idx, "caption": caption, "relation": relation,
                "category": category, "label": label,
                "target_subj": subj, "target_obj": obj,
                "yolov8n_detections": [d[0] for d in dets_v8n],
                "yolov8n_subj_found": v8n_subj, "yolov8n_obj_found": v8n_obj,
                "world_detections": [d[0] for d in dets_world],
                "world_subj_found": w_subj, "world_obj_found": w_obj,
            }
            f_out.write(json.dumps(entry) + "\n")

    # ── Summary ──────────────────────────────────────────────────
    n = max(valid_samples, 1)
    print(f"\n\n{'='*70}")
    print(f"  DETECTOR COMPARISON RESULTS ({valid_samples} samples)")
    print(f"{'='*70}")
    print(f"")
    print(f"  {'Metric':<35} {'YOLOv8n':>10} {'YOLO-World':>12}")
    print(f"  {'─'*57}")
    print(f"  {'Subject detected':<35} {yolov8n_stats['subj_found']:>7}/{n}   {world_stats['subj_found']:>7}/{n}")
    print(f"  {'Object detected':<35} {yolov8n_stats['obj_found']:>7}/{n}   {world_stats['obj_found']:>7}/{n}")
    print(f"  {'BOTH detected (critical)':<35} {yolov8n_stats['both_found']:>7}/{n}   {world_stats['both_found']:>7}/{n}")
    print(f"  {'─'*57}")
    print(f"  {'Subject detection rate':<35} {yolov8n_stats['subj_found']/n:>9.1%}   {world_stats['subj_found']/n:>9.1%}")
    print(f"  {'Object detection rate':<35} {yolov8n_stats['obj_found']/n:>9.1%}   {world_stats['obj_found']/n:>9.1%}")
    print(f"  {'Both detected rate':<35} {yolov8n_stats['both_found']/n:>9.1%}   {world_stats['both_found']/n:>9.1%}")
    print(f"  {'Avg detections per image':<35} {yolov8n_stats['total_detections']/n:>9.1f}   {world_stats['total_detections']/n:>9.1f}")
    print(f"{'='*70}")
    print(f"  Results saved to {args.output}")

    # Improvement summary
    both_delta = world_stats['both_found'] - yolov8n_stats['both_found']
    print(f"\n  ► YOLO-World found BOTH target objects in {both_delta:+d} more samples than YOLOv8n")
    print(f"    ({world_stats['both_found']}/{n} vs {yolov8n_stats['both_found']}/{n})")


if __name__ == "__main__":
    main()
