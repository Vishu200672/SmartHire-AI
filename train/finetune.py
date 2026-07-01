"""
finetune.py
-----------
Fine-tune sentence-transformers/all-MiniLM-L6-v2 on resume-JD pairs.

Why fine-tuning is needed:
  The base model assigns raw cosine 0.70 to a Senior ML Engineer vs an ML
  JD (gold score 0.97) and 0.65 to a Java developer vs the same JD (gold
  0.18). That narrow gap is too small for calibration alone to fix.
  Fine-tuning teaches domain-specific semantics:
    "ML Engineer" ≈ "Machine Learning Engineer"        (pull close)
    "Python PyTorch BERT" vs "Java Spring Boot REST"   (push apart)

Expected improvement:  Tier accuracy 65% → 80-90%

Usage (run from SmartHireAI folder):
    python train/finetune.py                   # 4 epochs (~10 min CPU)
    python train/finetune.py --epochs 6        # better accuracy
    python train/finetune.py --fast            # 2 epochs quick test
    python train/finetune.py --epochs 4 --lr 3e-5

Author: SmartHire AI
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("SmartHireAI.FineTune")


DEFAULT_CONFIG = {
    "base_model"    : "sentence-transformers/all-MiniLM-L6-v2",
    "output_dir"    : "models/smarthire-finetuned",
    "data_file"     : "train/training_data.json",
    "epochs"        : 4,
    "batch_size"    : 16,
    "warmup_steps"  : 50,
    "learning_rate" : 2e-5,
    "eval_split"    : 0.15,
    "max_seq_length": 256,
}

CALIBRATION_LOW  = 0.62
CALIBRATION_HIGH = 0.87
TIER_HR          = 70
TIER_REC         = 45
TIER_CON         = 20


def load_training_pairs(filepath: str) -> List[Dict]:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Training data not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid = []
    for item in data:
        if not all(k in item for k in ("resume", "jd", "score")):
            continue
        score = float(item["score"])
        if not 0.0 <= score <= 1.0:
            continue
        valid.append({
            "resume": str(item["resume"]).strip(),
            "jd"    : str(item["jd"]).strip(),
            "score" : score,
        })

    scores = [x["score"] for x in valid]
    high   = sum(1 for s in scores if s >= 0.85)
    mid    = sum(1 for s in scores if 0.35 <= s < 0.85)
    low    = sum(1 for s in scores if s < 0.35)
    logger.info(f"Loaded {len(valid)}/{len(data)} valid pairs  High:{high}  Mid:{mid}  Low:{low}")
    return valid


def split_data(pairs: List[Dict], eval_split: float = 0.15, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """Stratified train/eval split — each tier represented in eval."""
    import random
    random.seed(seed)

    high = [p for p in pairs if p["score"] >= 0.85]
    mid  = [p for p in pairs if 0.35 <= p["score"] < 0.85]
    low  = [p for p in pairs if p["score"] < 0.35]

    train, eval_set = [], []
    for bucket in [high, mid, low]:
        random.shuffle(bucket)
        n_eval = max(1, int(len(bucket) * eval_split))
        eval_set.extend(bucket[:n_eval])
        train.extend(bucket[n_eval:])

    random.shuffle(train)
    logger.info(f"Split  Train: {len(train)}  Eval: {len(eval_set)}")
    return train, eval_set


def _get_fit_kwargs(version_str: str) -> dict:
    """
    sentence-transformers renamed 'save_best_only' -> 'save_best_model'
    in v3.x. Detect which arg name to use so we work on both versions.
    """
    try:
        major = int(version_str.split(".")[0])
        return "save_best_model" if major >= 3 else "save_best_only"
    except Exception:
        return "save_best_model"   # default to newer name


def run_finetune(config: Dict) -> str:
    """Main fine-tuning function. Returns path to saved model."""
    try:
        import sentence_transformers as st_pkg
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
        from torch.utils.data import DataLoader
        import inspect
    except ImportError:
        logger.error("Run: pip install sentence-transformers")
        sys.exit(1)

    st_version = getattr(st_pkg, "__version__", "0.0.0")
    logger.info("=" * 60)
    logger.info("SmartHire AI — Fine-Tuning Pipeline")
    logger.info("=" * 60)
    logger.info(f"sentence-transformers : v{st_version}")
    logger.info(f"Base model            : {config['base_model']}")
    logger.info(f"Output dir            : {config['output_dir']}")
    logger.info(f"Epochs                : {config['epochs']}")
    logger.info(f"Batch size            : {config['batch_size']}")
    logger.info(f"Learning rate         : {config['learning_rate']}")
    logger.info(f"Max seq length        : {config['max_seq_length']}")
    logger.info("=" * 60)

    # Load model
    model = SentenceTransformer(config["base_model"])
    model.max_seq_length = config["max_seq_length"]
    logger.info(f"Embedding dim: {model.get_sentence_embedding_dimension()}")

    # Load & split data
    all_pairs               = load_training_pairs(config["data_file"])
    train_pairs, eval_pairs = split_data(all_pairs, config["eval_split"])

    train_examples = [
        InputExample(texts=[p["resume"], p["jd"]], label=float(p["score"]))
        for p in train_pairs
    ]

    train_loader = DataLoader(
        train_examples,
        shuffle    = True,
        batch_size = config["batch_size"],
    )

    loss_fn = losses.CosineSimilarityLoss(model=model)
    logger.info("Loss: CosineSimilarityLoss")

    evaluator = EmbeddingSimilarityEvaluator(
        sentences1 = [p["resume"] for p in eval_pairs],
        sentences2 = [p["jd"]     for p in eval_pairs],
        scores     = [p["score"]  for p in eval_pairs],
        name       = "resume-jd-eval",
    )

    output_path = Path(config["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)

    total_steps = len(train_loader) * config["epochs"]
    eval_steps  = max(10, total_steps // 8)

    logger.info(f"Total steps: {total_steps}  |  Warmup: {config['warmup_steps']}")
    logger.info("Starting training — progress bar will appear below:")
    logger.info("=" * 60)

    # ── Detect correct kwarg name for save_best ───────────────
    # sentence-transformers v2.x uses 'save_best_only'
    # sentence-transformers v3.x uses 'save_best_model'
    fit_sig    = inspect.signature(model.fit)
    save_kwarg = (
        "save_best_model"
        if "save_best_model" in fit_sig.parameters
        else "save_best_only"
    )
    logger.info(f"Using fit() kwarg: {save_kwarg}=True")

    fit_kwargs = {
        "train_objectives" : [(train_loader, loss_fn)],
        "evaluator"        : evaluator,
        "epochs"           : config["epochs"],
        "evaluation_steps" : eval_steps,
        "warmup_steps"     : config["warmup_steps"],
        "optimizer_params" : {"lr": config["learning_rate"]},
        "output_path"      : str(output_path),
        "show_progress_bar": True,
        save_kwarg         : True,
    }

    model.fit(**fit_kwargs)

    logger.info("=" * 60)
    logger.info(f"Training complete!  Model saved to: {output_path}")
    logger.info("=" * 60)
    return str(output_path)


def quick_eval(model_path: str, pairs: List[Dict]):
    """Tier accuracy comparison right after training."""
    try:
        from sentence_transformers import SentenceTransformer
        from scipy.stats import spearmanr
        import numpy as np
    except ImportError:
        return

    def calibrate(raw):
        return min(100.0, max(0.0,
            (raw - CALIBRATION_LOW) / (CALIBRATION_HIGH - CALIBRATION_LOW) * 100.0))

    def pred_tier(pct):
        if pct >= TIER_HR:    return "Highly Recommended"
        elif pct >= TIER_REC: return "Recommended"
        elif pct >= TIER_CON: return "Consider"
        else:                 return "Not Recommended"

    def gold_tier(g):
        if g >= 0.85:   return "Highly Recommended"
        elif g >= 0.65: return "Recommended"
        elif g >= 0.35: return "Consider"
        else:           return "Not Recommended"

    print(f"\n{'='*55}")
    print("POST-TRAINING ACCURACY COMPARISON")
    print(f"{'='*55}")

    for label, mp in [("Base model ", DEFAULT_CONFIG["base_model"]), ("Fine-tuned ", model_path)]:
        try:
            m = SentenceTransformer(mp)
            preds, golds, correct = [], [], 0
            for p in pairs:
                embs = m.encode([p["resume"], p["jd"]], normalize_embeddings=True)
                raw  = float(np.dot(embs[0], embs[1]))
                cal  = calibrate(raw)
                preds.append(raw)
                golds.append(p["score"])
                if pred_tier(cal) == gold_tier(p["score"]):
                    correct += 1
            acc = correct / len(pairs) * 100
            sr  = spearmanr(preds, golds)[0]
            print(f"  {label}: Tier acc = {acc:.1f}%  |  Spearman = {sr:.4f}")
        except Exception as e:
            print(f"  {label}: Could not evaluate — {e}")

    print(f"{'='*55}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune SmartHire AI embedding model")
    parser.add_argument("--base_model",  default=DEFAULT_CONFIG["base_model"])
    parser.add_argument("--output_dir",  default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--data",        default=DEFAULT_CONFIG["data_file"], dest="data_file")
    parser.add_argument("--epochs",      default=DEFAULT_CONFIG["epochs"],      type=int)
    parser.add_argument("--batch_size",  default=DEFAULT_CONFIG["batch_size"],  type=int)
    parser.add_argument("--lr",          default=DEFAULT_CONFIG["learning_rate"], type=float)
    parser.add_argument("--eval_split",  default=DEFAULT_CONFIG["eval_split"],  type=float)
    parser.add_argument("--max_seq_len", default=DEFAULT_CONFIG["max_seq_length"], type=int)
    parser.add_argument("--fast",        action="store_true",
                        help="Fast mode: 2 epochs, batch 32 (~2 min CPU)")
    parser.add_argument("--skip_eval",   action="store_true",
                        help="Skip post-training accuracy comparison")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.fast:
        logger.info("FAST MODE: 2 epochs, batch_size=32")
        args.epochs     = 2
        args.batch_size = 32

    config = {
        "base_model"    : args.base_model,
        "output_dir"    : args.output_dir,
        "data_file"     : args.data_file,
        "epochs"        : args.epochs,
        "batch_size"    : args.batch_size,
        "learning_rate" : args.lr,
        "eval_split"    : args.eval_split,
        "max_seq_length": args.max_seq_len,
        "warmup_steps"  : DEFAULT_CONFIG["warmup_steps"],
    }

    output_path = run_finetune(config)

    if not args.skip_eval:
        all_pairs = load_training_pairs(config["data_file"])
        quick_eval(output_path, all_pairs)

    logger.info(f"\n{'='*60}")
    logger.info("NEXT STEPS:")
    logger.info(f"  1. Model saved to: {output_path}")
    logger.info(f"  2. Close and reopen RUN_APP.bat  (auto-loads fine-tuned model)")
    logger.info(f"  3. Run: python train/evaluate.py --compare  (to see improvement)")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
