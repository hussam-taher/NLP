# imdb_train.py
# IMDb Sentiment Classification (Text Classification) - Course Project
#
# Key project customizations compared to the generic HF examples:
# 1) IMDb-only pipeline (load_dataset("imdb"))
# 2) Proper train/validation split for model selection
# 3) Early stopping based on validation F1
# 4) Extra analysis outputs: confusion matrix + misclassified examples
# 5) Quick experiment summary.csv for fast reporting

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

@dataclass
class RunConfig:
    model_name: str
    max_seq_length: int
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
    num_train_epochs: float
    weight_decay: float
    seed: int
    run_name: str
    output_root: str
    results_dir: str
    save_misclassified_k: int
    val_ratio: float
    early_stopping_patience: int
    early_stopping_threshold: float

    # OPTIONAL: allow overriding dropout if you want to show it explicitly in experiments
    dropout: float | None

def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="IMDb Sentiment Classification (Text Classification)")

    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=float, default=5.0)  # allow early stopping to kick in
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--run_name", type=str, default="imdb_run1")
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--save_misclassified_k", type=int, default=50)

    # NEW: validation + early stopping
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Fraction of train split used as validation")
    parser.add_argument("--early_stopping_patience", type=int, default=2, help="Stop if val metric doesn't improve")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0, help="Minimum improvement to count")

    # OPTIONAL: set dropout explicitly (good for showing you experimented)
    parser.add_argument("--dropout", type=float, default=None, help="Override model dropout (e.g., 0.1, 0.2)")

    args = parser.parse_args()

    return RunConfig(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        seed=args.seed,
        run_name=args.run_name,
        output_root=args.output_root,
        results_dir=args.results_dir,
        save_misclassified_k=args.save_misclassified_k,
        val_ratio=args.val_ratio,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        dropout=args.dropout,
    )


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def ensure_dirs(cfg: RunConfig) -> Tuple[Path, Path]:
    out_dir = Path(cfg.output_root) / cfg.run_name
    res_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, res_dir


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_confusion_matrix_csv(path: Path, cm: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["", "pred_0", "pred_1"])
        writer.writerow(["true_0", int(cm[0, 0]), int(cm[0, 1])])
        writer.writerow(["true_1", int(cm[1, 0]), int(cm[1, 1])])


def save_misclassified_csv(
    path: Path,
    texts: list[str],
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    k: int,
) -> None:
    wrong_idx = np.where(true_labels != pred_labels)[0][:k]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "true_label", "pred_label"])
        for i in wrong_idx:
            writer.writerow([texts[int(i)], int(true_labels[int(i)]), int(pred_labels[int(i)])])


def append_summary_csv(path: Path, row: Dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def override_dropout_if_requested(model, dropout: float | None) -> None:
    """
    Transformers models already include dropout by default (often ~0.1).
    If you want to explicitly experiment, this function tries to override
    common dropout config fields when they exist.
    """
    if dropout is None:
        return

    # BERT-like configs
    if hasattr(model.config, "hidden_dropout_prob"):
        model.config.hidden_dropout_prob = float(dropout)
    if hasattr(model.config, "attention_probs_dropout_prob"):
        model.config.attention_probs_dropout_prob = float(dropout)

    # DistilBERT configs
    if hasattr(model.config, "dropout"):
        model.config.dropout = float(dropout)
    if hasattr(model.config, "attention_dropout"):
        model.config.attention_dropout = float(dropout)


def main() -> None:
    cfg = parse_args()
    out_dir, res_dir = ensure_dirs(cfg)
    set_seed(cfg.seed)

    # 1) Load IMDb dataset
    raw = load_dataset("imdb")
    base_train = raw["train"]
    test_ds = raw["test"]

    # 2) Create a proper validation split from the training split
    split = base_train.train_test_split(test_size=cfg.val_ratio, seed=cfg.seed, shuffle=True)
    train_ds = split["train"]
    val_ds = split["test"]  # validation

    # 3) Tokenizer + tokenization
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.max_seq_length,
        )

    # Keep raw text for error analysis later (we'll use test_ds["text"])
    # For training/eval, we only need tokenized inputs + labels.
    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=[c for c in train_ds.column_names if c != "label"])
    val_tok = val_ds.map(tokenize_fn, batched=True, remove_columns=[c for c in val_ds.column_names if c != "label"])
    test_tok = test_ds.map(tokenize_fn, batched=True, remove_columns=[c for c in test_ds.column_names if c != "label"])

    # Trainer expects "labels"
    for ds_name in ["train_tok", "val_tok", "test_tok"]:
        ds = locals()[ds_name]
        if "label" in ds.column_names:
            locals()[ds_name] = ds.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 4) Model
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=2)
    override_dropout_if_requested(model, cfg.dropout)

    # 5) Training setup
    # We evaluate on VALIDATION during training, and keep TEST for the final report.
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        run_name=cfg.run_name,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        num_train_epochs=cfg.num_train_epochs,
        weight_decay=cfg.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_strategy="steps",
        logging_steps=50,
        report_to="none",
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg.early_stopping_patience,
                early_stopping_threshold=cfg.early_stopping_threshold,
            )
        ],
    )

    # 6) Train (with early stopping)
    start = time.time()
    train_result = trainer.train()
    train_time_sec = time.time() - start

    # Save model + tokenizer
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # 7) Evaluate on VALIDATION (for transparency)
    val_metrics = trainer.evaluate(eval_dataset=val_tok)

    # 8) Final evaluation on TEST (main result for your report)
    test_metrics = trainer.evaluate(eval_dataset=test_tok)

    clean_metrics = {
        "val": {
            "loss": float(val_metrics.get("eval_loss", np.nan)),
            "accuracy": float(val_metrics.get("eval_accuracy", np.nan)),
            "precision": float(val_metrics.get("eval_precision", np.nan)),
            "recall": float(val_metrics.get("eval_recall", np.nan)),
            "f1": float(val_metrics.get("eval_f1", np.nan)),
        },
        "test": {
            "loss": float(test_metrics.get("eval_loss", np.nan)),
            "accuracy": float(test_metrics.get("eval_accuracy", np.nan)),
            "precision": float(test_metrics.get("eval_precision", np.nan)),
            "recall": float(test_metrics.get("eval_recall", np.nan)),
            "f1": float(test_metrics.get("eval_f1", np.nan)),
        },
        "train_time_sec": float(train_time_sec),
        "config": {
            "model_name": cfg.model_name,
            "max_seq_length": cfg.max_seq_length,
            "train_batch_size": cfg.train_batch_size,
            "eval_batch_size": cfg.eval_batch_size,
            "learning_rate": cfg.learning_rate,
            "num_train_epochs": cfg.num_train_epochs,
            "weight_decay": cfg.weight_decay,
            "seed": cfg.seed,
            "val_ratio": cfg.val_ratio,
            "early_stopping_patience": cfg.early_stopping_patience,
            "early_stopping_threshold": cfg.early_stopping_threshold,
            "dropout": cfg.dropout,
            "run_name": cfg.run_name,
        },
    }

    save_json(out_dir / "metrics.json", clean_metrics)

    # 9) Results analysis on TEST: confusion matrix + misclassified examples
    preds_output = trainer.predict(test_tok)
    logits = preds_output.predictions
    y_true = preds_output.label_ids
    y_pred = np.argmax(logits, axis=-1)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    save_confusion_matrix_csv(out_dir / "confusion_matrix.csv", cm)

    test_texts = test_ds["text"]
    save_misclassified_csv(
        out_dir / "misclassified.csv",
        texts=test_texts,
        true_labels=y_true,
        pred_labels=y_pred,
        k=cfg.save_misclassified_k,
    )

    # 10) Quick summary for fast reporting (use TEST results)
    summary_row = {
        "run_name": cfg.run_name,
        "model": cfg.model_name,
        "max_seq_length": cfg.max_seq_length,
        "train_bs": cfg.train_batch_size,
        "eval_bs": cfg.eval_batch_size,
        "lr": cfg.learning_rate,
        "epochs_max": cfg.num_train_epochs,
        "val_ratio": cfg.val_ratio,
        "early_stop_patience": cfg.early_stopping_patience,
        "dropout": cfg.dropout,
        "val_f1": clean_metrics["val"]["f1"],
        "test_accuracy": clean_metrics["test"]["accuracy"],
        "test_f1": clean_metrics["test"]["f1"],
        "train_time_sec": clean_metrics["train_time_sec"],
        "output_dir": str(out_dir),
    }
    append_summary_csv(res_dir / "summary.csv", summary_row)

    print("\n=== DONE ===")
    print(f"Saved outputs to: {out_dir}")
    print(f"Summary updated: {res_dir / 'summary.csv'}")

if __name__ == "__main__":
    main()