import os
import sys
import json
import re
import torch
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import ViltProcessor, ViltForQuestionAnswering

# Suppress some noisy warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Add the project root to sys.path so we can import src modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.depth_captioning.depth_blip import DepthBlipCaptioner

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

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

def _normalize_yes_no(text: str) -> str:
    """
    Ensure the output is one of 'yes', 'no', or 'unknown'.
    """
    if not text:
        return "unknown"
    t = text.strip().lower()
    t = _NON_ALNUM_RE.sub(" ", t).strip()
    first = t.split(" ", 1)[0] if t else ""
    if first in ("yes", "y"):
        return "yes"
    if first in ("no", "n"):
        return "no"
    return "unknown"

def parse_pope_object(question):
    """Extract the target object name from a POPE question.

    POPE questions follow: 'Is there a {object} in the image?'

    Returns:
        str or None: The object name, or None if parsing fails.
    """
    import re as _re
    m = _re.search(r'Is there (?:a |an )?(.+?) in the image', question, _re.IGNORECASE)
    return m.group(1).strip() if m else None

def _compact_context(context: str, max_chars: int = 400) -> str:
    """
    Keep context length capped so the question is not truncated.
    """
    if not context:
        return ""
    c = " ".join(str(context).split())
    if len(c) <= max_chars:
        return c
    return c[: max_chars - 3].rstrip() + "..."

def _predict_vilt(vqa_model, processor, image, prompt):
    inputs = processor(images=image, text=prompt, return_tensors="pt", truncation=True, max_length=40).to(vqa_model.device)
    with torch.no_grad():
        outputs = vqa_model(**inputs)
        logits = outputs.logits
        # Check scores for yes and no specifically for voting
        yes_idx = vqa_model.config.label2id.get("yes")
        no_idx = vqa_model.config.label2id.get("no")
        
        if yes_idx is not None and no_idx is not None:
            yes_score = float(logits[0, yes_idx].detach().cpu())
            no_score = float(logits[0, no_idx].detach().cpu())
        else:
            yes_score, no_score = 0.0, 0.0
            
        idx = logits.argmax(-1).item()
        answer = vqa_model.config.id2label[idx]
        
    return answer, yes_score, no_score

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ViLT on POPE 150 with Spatial LDP")
    parser.add_argument("--dry_run", action="store_true", help="Run on only 5 samples for testing")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL file (auto-generated from vote_mode if not set)")
    parser.add_argument("--dataset", type=str, default="Rajarshi-Roy-research/lmms-lab-POPE", help="HF Dataset repository")
    parser.add_argument("--split", type=str, default="test_with_depth", help="Dataset split to use")
    parser.add_argument("--vqa_model", type=str, default="dandelin/vilt-b32-finetuned-vqa", help="ViLT VQA checkpoint")
    parser.add_argument("--depth_encoder", type=str, default="vits", choices=["vits","vitb","vitl","vitg"], help="Depth Anything V2 encoder size.")
    parser.add_argument("--yolo_model", type=str, default="yolov8l-worldv2.pt", help="YOLO model for spatial analysis.")
    parser.add_argument("--use_context", action="store_true", help="Include LDP+spatial context in the text prompt (baseline).")
    parser.add_argument("--layer_vote", action="store_true", help="Use layer-wise voting.")
    parser.add_argument("--orig_conf_threshold", type=float, default=2.0, help="Confidence threshold to trust original view.")
    parser.add_argument("--vote_mode", type=str, default="ldp_spatial", choices=["ldp", "ldp_spatial", "spatial"], help="Voting strategy.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for models (cpu/cuda)")
    return parser.parse_args()

def main():
    args = parse_args()
    # Auto-generate output filename from vote_mode if not explicitly provided
    if args.output is None:
        args.output = f"data/pope_vilt_{args.vote_mode}_predictions.jsonl"
    print(f"Starting POPE Benchmark for ViLT on device: {args.device}")
    print(f"Vote mode: {args.vote_mode} | Output: {args.output}")
    
    # 1. Initialize depth context creator or spatial analyzer
    depth_captioner = None
    spatial_analyzer = None
    if args.vote_mode == "spatial":
        print("\n[1/3] Initializing SpatialAnalyzer only (spatial mode)...")
        from src.depth_captioning.spatial_analysis import SpatialAnalyzer
        spatial_analyzer = SpatialAnalyzer(model_path=args.yolo_model)
    else:
        print("\n[1/3] Initializing DepthBlipCaptioner ...")
        depth_captioner = DepthBlipCaptioner(
            device=torch.device(args.device),
            encoder=args.depth_encoder,
            yolo_model_path=args.yolo_model,
        )
    
    # 2. Initialize ViLT VQA model
    print(f"\n[2/3] Initializing ViLT VQA model ({args.vqa_model})...")
    vqa_processor = ViltProcessor.from_pretrained(args.vqa_model)
    vqa_model = ViltForQuestionAnswering.from_pretrained(args.vqa_model).to(args.device)
    vqa_model.eval()

    # 3. Load Dataset
    print(f"\n[3/3] Loading dataset: {args.dataset} (split: {args.split})")
    try:
        ds = load_dataset(args.dataset, split=args.split)
    except Exception as e:
        print(f"Error loading dataset from HuggingFace: {e}")
        return
        
    num_samples = len(ds)
    if args.dry_run:
        num_samples = min(5, num_samples)
        print(f"\nDRY RUN enabled. Only processing {num_samples} samples.")
        ds = ds.select(range(num_samples))
    else:
        print(f"\nLoaded {num_samples} samples. Beginning evaluation...")
        
    y_true = []
    y_pred = []
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    with open(args.output, "w") as f_out:
        for idx, row in enumerate(tqdm(ds)):
            image = row['image']
            question = row['question']
            ground_truth = str(row['answer']).strip().lower()
            question_id = row.get('question_id', str(idx))
            
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # ---> Step A: Context Generation
            spatial_depth_caption = "N/A"
            vsr_spatial = ""
            try:
                # Set YOLO-World target class for this question's object
                target_obj = parse_pope_object(question)
                
                if args.vote_mode == "spatial" and spatial_analyzer is not None:
                    image_array = np.array(image)
                    is_present = spatial_analyzer.check_presence(image_array, target_obj)
                    if is_present:
                        vsr_spatial = f"Yes, there is a {target_obj} in the image."
                    else:
                        vsr_spatial = f"No, there is no {target_obj} in the image."
                    
                elif depth_captioner is not None:
                    image_array = np.array(image)
                    is_present = depth_captioner.spatial_analyzer.check_presence(image_array, target_obj)
                    if is_present:
                        vsr_spatial = f"Yes, there is a {target_obj} in the image."
                    else:
                        vsr_spatial = f"No, there is no {target_obj} in the image."
                    spatial_depth_caption = depth_captioner.get_caption_with_depth(image)
            except Exception as e:
                print(f"Error generating depth/spatial context for idx {idx}: {e}")
                spatial_depth_caption = "No depth context available."
                vsr_spatial = ""
            
            # ---> Step B/C: Predict Answer
            layer_details = []
            try:
                if args.vote_mode == "spatial":
                    # Spatial-only: Logit Ensembling instead of text prompting
                    raw_pred, y_score, n_score = _predict_vilt(vqa_model, vqa_processor, image, question)
                    m = y_score - n_score
                    
                    # YOLO mathematically boosts the logits
                    yolo_boost = 3.0 if is_present else -3.0
                    m += yolo_boost
                    
                    pred_answer = "yes" if m >= 0 else "no"
                    
                    raw_pred_json = json.dumps({
                        "strategy": "spatial_logit_ensemble", 
                        "yolo_present": is_present,
                        "original_margin": y_score - n_score,
                        "final_margin": m
                    })
                    raw_pred = raw_pred_json

                elif args.layer_vote:
                    layer_imgs_np, masks = depth_captioner.depth_context.make_depth_context_img(
                        image, top_threshold=70, bottom_threshold=30, return_masks=True
                    )
                    layer_names = ["Closest", "Farthest", "Mid Range"]

                    vote_margin = 0.0

                    # 1) Original image
                    base_pred, y_score, n_score = _predict_vilt(vqa_model, vqa_processor, image, question)
                    base_margin = y_score - n_score
                    layer_details.append({"view": "original", "pred": base_pred, "yes_score": y_score, "no_score": n_score, "margin": base_margin})

                    if abs(base_margin) >= float(args.orig_conf_threshold):
                        pred_answer = _normalize_yes_no(base_pred)
                        raw_pred = json.dumps({"strategy": "orig_only_confident", "orig_conf_threshold": args.orig_conf_threshold, "details": layer_details})
                    else:
                        vote_margin += base_margin

                        if args.vote_mode == "ldp":
                            for li, layer_np in enumerate(layer_imgs_np):
                                layer_pil = Image.fromarray(layer_np.astype("uint8"))
                                lp, ys, ns = _predict_vilt(vqa_model, vqa_processor, layer_pil, question)
                                m = ys - ns
                                vote_margin += m
                                layer_details.append({"view": layer_names[li], "pred": lp, "margin": m})

                            pred_answer = "yes" if vote_margin >= 0 else "no"
                            raw_pred = json.dumps({"strategy": "ldp_vote", "vote_margin": vote_margin, "details": layer_details})

                        else:
                            # LDP+Spatial: Iterate over all layers to gather 3 votes
                            for li, layer_np in enumerate(layer_imgs_np):
                                layer_pil = Image.fromarray(layer_np.astype("uint8"))
                                
                                caption = _compress_for_vilt(depth_captioner.captioner.get_caption(layer_np), max_words=6)

                                ctx_bits = []
                                if caption:
                                    ctx_bits.append(f"({layer_names[li][:3]} layer: {caption})")
                                small_ctx = " ".join(ctx_bits)

                                layer_prompt = f"{small_ctx} {question}" if small_ctx else question
                                lp, ys, ns = _predict_vilt(vqa_model, vqa_processor, layer_pil, layer_prompt)
                                
                                m = ys - ns
                                # Logit Ensembling: mathematically sway the vote
                                yolo_vote = 1.0 if is_present else -1.0
                                m += yolo_vote
                                
                                vote_margin += m
                                layer_details.append({"view": layer_names[li], "pred": lp, "margin": m, "caption": caption, "yolo_vote": yolo_vote})

                            pred_answer = "yes" if vote_margin >= 0 else "no"
                            raw_pred = json.dumps({"strategy": "ldp_spatial_vote", "vote_margin": vote_margin, "details": layer_details})

                else:
                    if args.use_context:
                        compact_context = _compact_context(spatial_depth_caption, max_chars=200)
                        prompt = f"Question: {question} Context: {compact_context}. Answer:"
                    else:
                        prompt = question

                    raw_pred, _, _ = _predict_vilt(vqa_model, vqa_processor, image, prompt)
                    pred_answer = _normalize_yes_no(raw_pred)
                    if pred_answer == "unknown":
                        pred_answer = "no"

            except Exception as e:
                print(f"Error generating answer for idx {idx}: {e}")
                raw_pred = ""
                pred_answer = "no"

            y_true.append(1 if ground_truth == "yes" else 0)
            y_pred.append(1 if pred_answer == "yes" else 0)
            
            log_entry = {
                "question_id": question_id,
                "question": question,
                "ground_truth": ground_truth,
                "pred_answer": pred_answer,
                "raw_pred": raw_pred,
                "spatial_depth_caption": spatial_depth_caption
            }
            f_out.write(json.dumps(log_entry) + "\n")
            f_out.flush()
            
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print("\n" + "="*50)
    print("      POPE Benchmark Results (ViLT)     ")
    print("="*50)
    print(f"Total Samples: {num_samples}")
    print(f"Accuracy:      {acc:.4f}")
    print(f"Precision:     {prec:.4f}")
    print(f"Recall:        {rec:.4f}")
    print(f"F1 Score:      {f1:.4f}")
    print("="*50)
    print(f"Detailed predictions saved to {args.output}")

if __name__ == "__main__":
    main()
