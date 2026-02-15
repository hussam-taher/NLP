# IMDb Sentiment Classification Project Using Transformers

> This guide explains **what we did** step by step, how to reproduce the experiments, and where to find **the saved result files**.

> **Important Note (Code Adaptation):** This project was adapted from the Hugging Face Transformers example `run_glue.py`, then modified to fit IMDb sentiment classification, with validation splitting, early stopping, extra metrics, and results saving (JSON/CSV).

---

## 1) Project Overview
We apply **binary text classification** to IMDb movie reviews to predict sentiment:
- **0 = Negative**
- **1 = Positive**

Main script:
- `imdb_train.py`

Models compared (final report):
- `bert-base-uncased`
- `distilbert-base-uncased`

We also tested **dropout** as an experimental factor:
- default dropout (null)
- dropout = 0.2

---

## 2) Environment and Requirements

### 2.1 Running Environment
- Python 3.9+ (3.10/3.11 recommended)
- Use a virtual environment (`.venv`) to avoid dependency conflicts.

### 2.2 Main Libraries
- **transformers**  
  Load models/tokenizers, run training with `Trainer`, use `TrainingArguments`, enable `EarlyStoppingCallback`.

- **datasets**  
  Load IMDb using `load_dataset("imdb")`, and create a validation split.

- **torch (PyTorch)**  
  Training backend for forward/backward passes and optimization.

- **numpy**  
  Utility operations for arrays/predictions.

- **scikit-learn (sklearn.metrics)**  
  Compute Accuracy / Precision / Recall / F1 and Confusion Matrix.

- **argparse**  
  Configure experiments via command line (model, batch size, max length, dropout, etc.).

- **json / csv / pathlib.Path**  
  Save outputs to structured folders (metrics, confusion matrix, misclassified samples, summary).

---

## 3) Dataset: IMDb (Hugging Face)
We load IMDb via Hugging Face Datasets:

- `raw = load_dataset("imdb")`

This provides ready-made splits:
- `raw["train"]`
- `raw["test"]`

Dataset reference:
- https://huggingface.co/datasets/stanfordnlp/imdb

---

## 4) What happens inside the code? (Pipeline)

### 4.1 Train/Validation split
Because **Early Stopping** needs a validation set, we split the original train split into:
- Train
- Validation

Using:
- `train_test_split(test_size=val_ratio, seed=seed, shuffle=True)`

Default:
- `val_ratio = 0.1` (10% of train is used for validation)

### 4.2 Tokenization
Tokenizer:
- `AutoTokenizer.from_pretrained(model_name, use_fast=True)`

Tokenization settings:
- `truncation=True`
- `max_length=max_seq_length`

Then tokenization is applied to train/val/test with `map(...)`.

> **Important:** In our final experiments, we used **max_seq_length = 512** for fair comparison across models.

### 4.3 Dynamic Padding (Batch preparation)
We do **dynamic padding** during batching using:
- `DataCollatorWithPadding(tokenizer=tokenizer)`

This avoids static padding for every sample from the start and improves efficiency.

### 4.4 Model Loading
We load a pretrained Transformer for sequence classification:
- `AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)`

Because it is a binary sentiment task.

### 4.5 Training and Evaluation (Trainer)
Training is configured using `TrainingArguments` (learning rate, batch sizes, epochs, weight decay, etc.) then we build:
- `Trainer(...)`

Main steps:
- `trainer.train()` → training (with early stopping)
- `trainer.evaluate(val)` → validation evaluation during training
- `trainer.evaluate(test)` → final evaluation on test (main report results)

Early stopping:
- `EarlyStoppingCallback(...)`

The “best model” is selected based on:
- `metric_for_best_model="f1"`
- `load_best_model_at_end=True`

---

## 5) Concepts Demonstrated in Training

1) **Iteration (One Weight Update)**  
Each batch: forward → loss → backward → optimizer step (handled internally by `Trainer`).

2) **Batch Size**  
Controlled from CLI:
- `--train_batch_size`
- `--eval_batch_size`

3) **Early Stopping**  
Stops training if validation F1 does not improve for a specified patience:
- `--early_stopping_patience 2` (used in our runs)

4) **Dropout Experiment**  
Dropout exists by default in Transformer architectures.  
We tested an explicit override using:
- `--dropout 0.2`
Or we left it unset to keep default dropout (null).

---

## 6) Outputs (Where results are saved)
For each run, the script saves:
- `outputs/<run_name>/metrics.json`  
  Contains validation/test metrics (accuracy/precision/recall/f1) + runtime and configuration.

- `outputs/<run_name>/confusion_matrix.csv`  
  Confusion matrix values for the test split.

- `outputs/<run_name>/misclassified.csv`  
  Samples where prediction != true label (used for error analysis).

A combined comparison table is saved/appended to:
- `results/summary.csv`  
  (one row per run)

---

## 7) Final Experiments We Ran (4 Runs) + Test Results

We ran **4 configurations** (2 models × 2 dropout settings) using:
- `max_seq_length = 512`
- early stopping patience = 2
- epochs = 5
- learning rate = 2e-5
- weight decay = 0.01
- val_ratio = 0.1

### 7.1 Run A — DistilBERT (dropout = 0.2)
- Model: `distilbert-base-uncased`
- train_batch_size = 16
- eval_batch_size = 32
- dropout = 0.2

**Test Confusion Matrix**
- TN=11296, FP=1204
- FN=599,  TP=11901

**Test Metrics**
- Accuracy = **0.92788**
- Precision = **0.90813**
- Recall = **0.95208**
- F1 = **0.92958**

---

### 7.2 Run B — DistilBERT (dropout = null / default)
- Model: `distilbert-base-uncased`
- train_batch_size = 16
- eval_batch_size = 32
- dropout = default (null)

**Test Confusion Matrix**
- TN=11644, FP=856
- FN=868,  TP=11632

**Test Metrics**
- Accuracy = **0.93104**
- Precision = **0.93145**
- Recall = **0.93056**
- F1 = **0.93101**

---

### 7.3 Run C — BERT (dropout = 0.2)
- Model: `bert-base-uncased`
- train_batch_size = 8
- eval_batch_size = 16
- dropout = 0.2

**Test Confusion Matrix**
- TN=11617, FP=883
- FN=715,  TP=11785

**Test Metrics**
- Accuracy = **0.93608**
- Precision = **0.93030**
- Recall = **0.94280**
- F1 = **0.93651**

---

### 7.4 Run D — BERT (dropout = null / default)
- Model: `bert-base-uncased`
- train_batch_size = 8
- eval_batch_size = 16
- dropout = default (null)

**Test Confusion Matrix**
- TN=11617, FP=883
- FN=715,  TP=11785

**Test Metrics**
- Accuracy = **0.93608**
- Precision = **0.93030**
- Recall = **0.94280**
- F1 = **0.93651**

> Note: BERT results for dropout=0.2 and dropout=default came out identical in our saved outputs. This should be mentioned in the report as a point to verify whether dropout override applied as intended.

---

## 8) Reproducibility: Example Commands

### DistilBERT (dropout = null)
```bash
python imdb_train.py \
  --model_name distilbert-base-uncased \
  --max_seq_length 512 \
  --train_batch_size 16 \
  --eval_batch_size 32 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --num_train_epochs 5 \
  --val_ratio 0.1 \
  --early_stopping_patience 2 \
  --run_name distilbert_null
