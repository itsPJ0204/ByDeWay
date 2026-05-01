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
from transformers import BlipProcessor, BlipForQuestionAnswering

# Suppress some noisy warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Add the project root to sys.path so we can import src modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.depth_captioning.depth_blip import DepthBlipCaptioner

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

def _shorten(text: str, max_words: int = 18) -> str:
    if not text:
        return ""
    words = str(text).strip().split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + " ..."

def _normalize_yes_no(text: str) -> str:
    """
    BLIP VQA sometimes answers with short free-form text (e.g. "yes.", "no", "not sure").
    We only accept a leading yes/no; anything else becomes 'unknown'.
    """
    if not text:
        return "unknown"
    t = text.strip().lower()
    # Keep only a simple token stream to avoid matching 'no' inside 'not'
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

def _compact_context(context: str, max_chars: int = 600) -> str:
    """
    Prevent BLIP tokenizer truncation from chopping off the question.
    Keep the question first and cap context length aggressively.
    """
    if not context:
        return ""
    c = " ".join(str(context).split())  # collapse whitespace/newlines
    if len(c) <= max_chars:
        return c
    return c[: max_chars - 3].rstrip() + "..."

def _score_yes_no_from_first_step(vqa_model, tokenizer, inputs):
    """
    BLIP-VQA is not instruction-tuned; free-form generation often drifts to nouns.
    For POPE we want a strict yes/no decision. We score only the first generated token:
    compare logits for token('yes') vs token('no') at decoding step 1.
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

    # scores[0] is logits for the first generated token
    step_logits = gen.scores[0][0]  # (vocab,)
    yes_score = float(step_logits[yes_id].detach().cpu())
    no_score = float(step_logits[no_id].detach().cpu())
    pred = "yes" if yes_score >= no_score else "no"
    return pred, {"yes": yes_score, "no": no_score}

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BLIP on POPE 150 with Spatial LDP")
    parser.add_argument("--dry_run", action="store_true", help="Run on only 5 samples for testing")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL file (auto-generated from vote_mode if not set)")
    parser.add_argument("--dataset", type=str, default="Rajarshi-Roy-research/lmms-lab-POPE", help="HF Dataset repository")
    parser.add_argument("--split", type=str, default="test_with_depth", help="Dataset split to use")
    parser.add_argument("--vqa_model", type=str, default="Salesforce/blip-vqa-base", help="BLIP VQA checkpoint (try 'Salesforce/blip-vqa-capfilt-large' for higher accuracy).")
    parser.add_argument("--depth_encoder", type=str, default="vits", choices=["vits","vitb","vitl","vitg"], help="Depth Anything V2 encoder size.")
    parser.add_argument("--yolo_model", type=str, default="yolov8l-worldv2.pt", help="YOLO model for spatial analysis (default: yolov8l-worldv2.pt).")
    parser.add_argument("--use_context", action="store_true", help="Include LDP+spatial context in the text prompt (for non-voting baseline).")
    parser.add_argument("--layer_vote", action="store_true", help="Use layer-wise BLIP voting (recommended for LDP+Spatial).")
    parser.add_argument("--orig_conf_threshold", type=float, default=2.0, help="If |original_margin| >= threshold, trust original view and skip layer voting.")
    parser.add_argument("--vote_mode", type=str, default="ldp_spatial", choices=["ldp", "ldp_spatial"], help="Voting strategy: 'ldp' uses only depth-layer images; 'ldp_spatial' adds small spatial text when needed.")
    # Defaul to CPU as requested by user with Intel Iris Xe (integrated graphics)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for models (cpu/cuda)")
    return parser.parse_args()

def main():
    args = parse_args()
    # Auto-generate output filename from vote_mode if not explicitly provided
    if args.output is None:
        args.output = f"data/pope_blip_{args.vote_mode}_predictions.jsonl"
    print(f"Starting POPE Benchmark for BLIP (LDP+Spatial) on device: {args.device}")
    print(f"Vote mode: {args.vote_mode} | Output: {args.output}")
    
    # 1. Initialize the depth captioner with spatial analysis
    print("\n[1/3] Initializing DepthBlipCaptioner (Depth Anything + YOLO + BLIP Captioning)...")
    depth_captioner = DepthBlipCaptioner(
        device=torch.device(args.device),
        encoder=args.depth_encoder,
        yolo_model_path=args.yolo_model,
    )
    
    # 2. Initialize the base BLIP VQA model that will answer the questions
    print("\n[2/3] Initializing base BLIP VQA model...")
    # Using base model because CPU evaluation is slow
    vqa_model_name = args.vqa_model
    vqa_processor = BlipProcessor.from_pretrained(vqa_model_name)
    vqa_model = BlipForQuestionAnswering.from_pretrained(vqa_model_name).to(args.device)
    vqa_model.eval()

    # 3. Load POPE dataset
    print(f"\n[3/3] Loading dataset: {args.dataset} (split: {args.split})")
    try:
        ds = load_dataset(args.dataset, split=args.split)
    except Exception as e:
        print(f"Error loading dataset from HuggingFace: {e}")
        print("Please ensure your internet connection works and HuggingFace is reachable.")
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
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    with open(args.output, "w") as f_out:
        for idx, row in enumerate(tqdm(ds)):
            image = row['image']
            question = row['question']
            ground_truth = str(row['answer']).strip().lower() # 'yes' or 'no'
            question_id = row.get('question_id', str(idx))
            
            # Ensure image is RGB
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # ---> Step A: Generate LDP + Spatial Caption
            vsr_spatial = ""
            try:
                target_obj = parse_pope_object(question)
                if depth_captioner is not None:
                    image_array = np.array(image)
                    is_present = depth_captioner.spatial_analyzer.check_presence(image_array, target_obj)
                    if is_present:
                        vsr_spatial = f"Yes, there is a {target_obj} in the image."
                    else:
                        vsr_spatial = f"No, there is no {target_obj} in the image."
                    spatial_depth_caption = depth_captioner.get_caption_with_depth(image)
                else:
                    spatial_depth_caption = ""
            except Exception as e:
                print(f"Error generating depth/spatial context for idx {idx}: {e}")
                spatial_depth_caption = "No depth context available."
                vsr_spatial = ""
            
            # ---> Step B/C: Predict Answer
            # Two modes:
            # - default: question-only BLIP
            # - --layer_vote: run BLIP on original + each depth layer (with small LDP+Spatial context) and aggregate
            try:
                if args.layer_vote:
                    # Build depth-layer images + masks
                    layer_imgs_np, masks = depth_captioner.depth_context.make_depth_context_img(
                        image, top_threshold=70, bottom_threshold=30, return_masks=True
                    )
                    layer_names = ["Closest", "Farthest", "Mid Range"]

                    # Vote accumulator: positive => yes, negative => no
                    vote_margin = 0.0
                    layer_details = []

                    # 1) Original image (question only)
                    base_inputs = vqa_processor(
                        images=image,
                        text=question,
                        return_tensors="pt",
                        truncation=True,
                        max_length=64,
                    ).to(args.device)
                    base_pred, base_scores = _score_yes_no_from_first_step(
                        vqa_model=vqa_model, tokenizer=vqa_processor.tokenizer, inputs=base_inputs
                    )
                    base_margin = base_scores["yes"] - base_scores["no"]
                    layer_details.append({"view": "original", "pred": base_pred, "scores": base_scores, "margin": base_margin})

                    # Confidence gate: BLIP's original view is usually strongest.
                    # If it's confident, don't let noisy layer context drag it down.
                    if abs(base_margin) >= float(args.orig_conf_threshold):
                        pred_answer = base_pred
                        raw_pred = json.dumps(
                            {"strategy": "orig_only_confident", "orig_conf_threshold": args.orig_conf_threshold, "details": layer_details}
                        )
                    else:
                        # Low-confidence case: apply the selected vote_mode.
                        vote_margin += base_margin

                        if args.vote_mode == "ldp":
                            # LDP-only: use depth-layer *images* only (no extra text).
                            for li, layer_np in enumerate(layer_imgs_np):
                                layer_pil = Image.fromarray(layer_np.astype("uint8"))
                                layer_inputs = vqa_processor(
                                    images=layer_pil,
                                    text=question,
                                    return_tensors="pt",
                                    truncation=True,
                                    max_length=64,
                                ).to(args.device)
                                lp, ls = _score_yes_no_from_first_step(
                                    vqa_model=vqa_model, tokenizer=vqa_processor.tokenizer, inputs=layer_inputs
                                )
                                m = ls["yes"] - ls["no"]
                                vote_margin += m
                                layer_details.append({"view": layer_names[li], "pred": lp, "scores": ls, "margin": m})

                            pred_answer = "yes" if vote_margin >= 0 else "no"
                            raw_pred = json.dumps(
                                {
                                    "strategy": "gated_orig_plus_all_layers",
                                    "vote_mode": "ldp",
                                    "orig_conf_threshold": args.orig_conf_threshold,
                                    "vote_margin": vote_margin,
                                    "details": layer_details,
                                }
                            )

                        else:
                            # LDP+Spatial: consult Mid Range with tiny caption + spatial if needed.
                            li = 2  # Mid Range
                            layer_np = layer_imgs_np[li]
                            layer_pil = Image.fromarray(layer_np.astype("uint8"))

                            caption = _shorten(depth_captioner.captioner.get_caption(layer_np), max_words=16)

                            ctx_bits = []
                            if caption:
                                ctx_bits.append(f"({layer_names[li]} layer: {caption})")
                            small_ctx = " ".join(ctx_bits)

                            layer_prompt = f"{small_ctx} {question}" if small_ctx else question
                            layer_inputs = vqa_processor(
                                images=layer_pil,
                                text=layer_prompt,
                                return_tensors="pt",
                                truncation=True,
                                max_length=96,
                            ).to(args.device)
                            lp, ls = _score_yes_no_from_first_step(
                                vqa_model=vqa_model, tokenizer=vqa_processor.tokenizer, inputs=layer_inputs
                            )
                            m = ls["yes"] - ls["no"]
                            
                            # Logit Ensembling: mathematically sway the vote
                            yolo_vote = 1.0 if is_present else -1.0
                            m += yolo_vote
                            
                            vote_margin += m
                            layer_details.append({"view": layer_names[li], "pred": lp, "scores": ls, "margin": m, "caption": caption, "yolo_vote": yolo_vote})

                            pred_answer = "yes" if vote_margin >= 0 else "no"
                            raw_pred = json.dumps(
                                {
                                    "strategy": "gated_orig_plus_mid",
                                    "vote_mode": "ldp_spatial",
                                    "orig_conf_threshold": args.orig_conf_threshold,
                                    "vote_margin": vote_margin,
                                    "details": layer_details,
                                }
                            )

                else:
                    # Non-voting baselines
                    if args.use_context:
                        compact_context = _compact_context(spatial_depth_caption, max_chars=600)
                        prompt = (
                            f"Question: {question}\n"
                            f"Context: {compact_context}\n"
                            f"Answer: yes or no."
                        )
                    else:
                        prompt = question

                    inputs = vqa_processor(
                        images=image,
                        text=prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=64,
                    ).to(args.device)
                    with torch.no_grad():
                        out = vqa_model.generate(**inputs, max_new_tokens=3, num_beams=5)
                    raw_pred = vqa_processor.decode(out[0], skip_special_tokens=True).strip()
                    pred_answer = _normalize_yes_no(raw_pred)
                    if pred_answer == "unknown":
                        pred_answer = "no"
            except Exception as e:
                print(f"Error generating answer for idx {idx}: {e}")
                raw_pred = ""
                pred_answer = "no"

            # Record truth & prediction
            y_true.append(1 if ground_truth == "yes" else 0)
            y_pred.append(1 if pred_answer == "yes" else 0)
            
            # Log to JSONL
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
            
    # Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print("\n" + "="*50)
    print("      POPE Benchmark Results (BLIP LDP+Spatial)     ")
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
