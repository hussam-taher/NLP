# IMDb Sentiment Classification Project Using Transformers

> This guide is for you: It explains **what we did** step by step, how to run the experiments yourself, and where to quickly find **results files**.

> **Important Note (Code Adaptation):** This project was adapted from the Hugging Face Transformers example `run_glue.py`, then modified to be suitable for IMDb Sentiment Classification, with the addition of extra metrics and results saving.

<!-- This project was adapted from the Hugging Face Transformers example (run_glue.py), then modified to be suitable for IMDb Sentiment Classification,
with the addition of: Validation Split, Early Stopping, extra metrics, and results saving (JSON/CSV).

-->

---

## 1) Project Overview
We apply Text Classification to IMDb data (movie reviews) to determine if a review is:
- **0 = Negative**
- **1 = Positive**

The main script we used is:
- `imdb_train.py`

---

## 2) Environment and Requirements

### 2.1 Running Environment
- Python 3.9+ (3.10 or 3.11 recommended)
- It is recommended to run the project in a Virtual Environment (`.venv`)

### 2.2 Key Libraries and Why We Use Them
- **transformers**

For loading the model and tokenizer, managing training via `Trainer` and `TrainingArguments`, and enabling `EarlyStoppingCallback`.

- **datasets**

To load IMDb data directly using: `load_dataset("imdb")`, and also perform Train/Validation splitting.

- **torch (PyTorch)**

The actual training engine (backprop + weight update).

> Note: `Trainer` from transformers builds on top of PyTorch.

- **numpy**

For handling matrices and converting logits to predictions using `argmax`.

- **scikit-learn (sklearn.metrics)**

For calculating: Accuracy, Precision, Recall, F1, and Confusion Matrix.

- **argparse**

To change settings from the command line (model selection, batch size, text length, etc.).

- **json/csv/pathlib.Path**

To save results to JSON/CSV files and automatically create output folders.

---

## 3) Data: How do we get IMDb?

We use the `datasets` library and load the data with this line of code:

- `raw = load_dataset("imdb")`

This automatically:

1) Downloads the data (only the first time)

2) Stores it in a cache

3) Gives you ready-made splits:

- `raw["train"]`

- `raw["test"]`

> Important note: The dataset comes from Hugging Face Datasets (an external source), but the loading method is easy and direct via `load_dataset`.

---

## 4) What happens inside the code?

The script works in this sequence:

### 4.1 Splitting Train/Validation
Because **Early Stopping** requires a Validation set, we split `train` into:

- Train
- Validation

Using:

- `train_test_split(test_size=val_ratio, seed=seed, shuffle=True)`

The default is:

- `val_ratio = 0.1` (i.e., 10% of the train becomes validation)

### 4.2 Tokenization and Script Preparation
We load a tokenizer that matches the model:

- `AutoTokenizer.from_pretrained(model_name, use_fast=True)`

Then we apply:

- `truncation=True`

- `max_length=max_seq_length`

After that, we use `map(...)` to apply the transformation to:

- train/val/test

### 4.3 Padding (Preparing Batches)
Instead of applying static padding from the start, we used:
- `DataCollatorWithPadding(tokenizer=tokenizer)`

This performs dynamic padding for each batch during training and evaluation.

### 4.4 Loading the Model
We used:
- `AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)`

Because it's a binary task (0/1).

### 4.5 Training and Evaluation
We configure the training using `TrainingArguments` (such as learning rate / batch size / epochs...), then we create:
- `Trainer(...)`

Finally:
- `trainer.train()` → Starts training
- `trainer.evaluate(val)` → Evaluates validation during training
- `trainer.evaluate(test)` / `trainer.predict(test)` → Final evaluation on test + prediction extraction

---

## 5) These four concepts appear practically within training:

1) **Iteration (One Weight Update)**
Each batch goes through forward → loss → backward → weight update. This happens automatically within `Trainer`.

2) **Batch**

Controlled from the command line:

- `--train_batch_size`

- `--eval_batch_size`

3) **Early Stopping**

Implemented via:

- `EarlyStoppingCallback(...)`

Stops training if validation performance (as determined by F1) does not improve for a specified number of epochs.

4) **Dropout**

Inherent by default in BERT/DistilBERT architectures.

You can (optionally) test it via:

- `--dropout 0.2`

However, in our current testing, we left it at the default (dropout = null in config).

---

## 6) Experiments We Carried Out (Run1 and Run2) + Results
We conducted two basic experiments to compare a lightweight model versus a larger one:

### 6.1 Experiment 1: DistilBERT (run1_distilbert)
- Model: `distilbert-base-uncased`
- max_seq_length = 256
- train_batch_size = 16
- eval_batch_size = 32
- learning_rate = 2e-5
- weight_decay = 0.01
- val_ratio = 0.1
- early_stopping_patience = 2

**Validation Results**
- Accuracy = 0.9092
- F1 = 0.90996

**Test Results**
- Accuracy = 0.9132
- Precision = 0.90163
- Recall = 0.9276
- F1 = 0.91443
- Training time ≈ 2486 seconds

### 6.2 Second Experiment: BERT (run2_bert)
- Model: `bert-base-uncased`
- max_seq_length = 256
- train_batch_size = 8
- eval_batch_size = 16
- learning_rat
