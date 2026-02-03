# مشروع تصنيف النصوص (IMDb Sentiment Classification) باستخدام Transformers
<!-- تم تكييف هذا الكود من مثال Hugging Face Transformers (run_glue.py)، وتم تعديله لتصنيف المشاعر على بيانات IMDb، مع إضافة مقاييس تقييم إضافية وتسجيل النتائج. -->
هذا المشروع يطبّق **تصنيف النصوص (Text Classification)** على بيانات **IMDb** (مراجعات أفلام) لتحديد ما إذا كانت المراجعة **إيجابية** أو **سلبية** (تصنيف ثنائي).

السكربت الرئيسي في هذا المشروع هو:

- `imdb_train.py`

---

## 1) المتطلبات (Requirements)

### بيئة التشغيل
- Python 3.9+ (يفضّل 3.10 أو 3.11)
- يفضّل تشغيله داخل Virtual Environment (`.venv`)

### المكتبات المستخدمة (Libraries)
> هذه هي أهم المكتبات ولماذا استخدمناها داخل الكود:

- **transformers**  
  لتجهيز الموديل والتوكنيزر (Tokenizer) والتدريب باستخدام `Trainer` و`TrainingArguments`، وأيضًا إضافة `EarlyStoppingCallback`.

- **datasets**  
  لتحميل بيانات IMDb مباشرة باستخدام: `load_dataset("imdb")`، وكذلك عمل تقسيم Train/Validation.

- **numpy**  
  لتحويل logits إلى توقعات باستخدام `argmax` وللتعامل مع arrays.

- **scikit-learn (sklearn.metrics)**  
  لحساب مؤشرات الأداء: Accuracy و Precision و Recall و F1، بالإضافة إلى Confusion Matrix.

- **argparse**  
  لتشغيل السكربت بإعدادات مختلفة من سطر الأوامر (مثل اختيار موديل أو تغيير batch size).

- **json / csv / pathlib.Path**  
  لحفظ النتائج في ملفات (JSON/CSV) وإنشاء مجلدات الإخراج تلقائيًا.

---

## 2) كيف تعمل الداتا؟ وكيف يحملها الكود؟

### تحميل بيانات IMDb
الكود يستخدم:
- `raw = load_dataset("imdb")`

وهذا يقوم تلقائيًا بـ:
- تنزيل البيانات من Hugging Face Datasets (أول مرة فقط)
- تخزينها في cache محلي
- توفير splits جاهزة:
  - `raw["train"]`
  - `raw["test"]`

### إنشاء Validation Split
لأن Early Stopping يحتاج **Validation**، الكود يأخذ `train` ويقسمه إلى:
- Train
- Validation

باستخدام:
- `train_test_split(test_size=val_ratio, seed=seed, shuffle=True)`

> `val_ratio` افتراضيًا 0.1 (أي 10% من train تصبح validation).

---

## 3) شرح السكربت: ماذا يفعل خطوة بخطوة؟

> ملاحظة: الشرح هنا يتبع ترتيب الكود كما هو داخل `imdb_train_with_val_earlystop.py`.

### (A) التعليقات أعلى الملف
في بداية الملف توجد تعليقات تشرح أن السكربت:
- مخصص لـ IMDb فقط
- يضيف train/validation split
- يضيف Early Stopping
- يحفظ Confusion Matrix + أمثلة misclassified
- يكتب summary.csv للتجارب بسرعة

### (B) الاستيرادات Imports
يتم استيراد كل المكتبات المطلوبة:
- من بايثون: `argparse, csv, json, time, dataclass, Path`
- من خارج بايثون: `numpy, datasets, sklearn.metrics, transformers`

### (C) كلاس الإعدادات RunConfig
يتم تعريف `RunConfig` باستخدام `@dataclass` لتجميع كل إعدادات التشغيل في كائن واحد مثل:
- `model_name` اسم الموديل (DistilBERT افتراضيًا)
- `max_seq_length` طول النص بعد tokenization
- `batch sizes`, `learning rate`, `epochs`
- `val_ratio` نسبة الـ validation
- `early_stopping_patience` عدد epochs بدون تحسن قبل الإيقاف
- `dropout` (اختياري لتجارب dropout)

الفائدة: بدل ما تكون الإعدادات متفرقة، تصبح منظمة وسهلة التخزين ضمن النتائج.

### (D) parse_args() — قراءة إعدادات التشغيل من سطر الأوامر
الدالة:
- تنشئ `ArgumentParser`
- تضيف arguments مثل:
  - `--model_name`
  - `--max_seq_length`
  - `--train_batch_size`, `--eval_batch_size`
  - `--learning_rate`, `--num_train_epochs`
  - `--val_ratio`
  - `--early_stopping_patience`
  - `--dropout` (اختياري)
- ثم ترجع `RunConfig` يحتوي القيم كلها.

### (E) compute_metrics() — حساب المقاييس أثناء التقييم
الدالة تستقبل:
- `logits` (ناتج الموديل)
- `labels` (القيم الصحيحة)

ثم:
1) تحول logits إلى توقعات:
   - `preds = argmax(logits)`
2) تحسب:
   - Accuracy
   - Precision / Recall / F1 (للـ binary classification)
3) ترجع قاموس metrics ليستخدمه `Trainer`.

### (F) ensure_dirs() — إنشاء مجلدات الإخراج تلقائيًا
هذه الدالة تضمن وجود مجلدين:
- `outputs/<run_name>/`
- `results/`

إذا لم تكن موجودة، يتم إنشاؤها تلقائيًا.


### (G) دوال حفظ النتائج (JSON/CSV)
يوجد عدة دوال صغيرة للحفظ:
- `save_json(...)` لحفظ metrics في JSON
- `save_confusion_matrix_csv(...)` لحفظ confusion matrix
- `save_misclassified_csv(...)` لحفظ أمثلة تم تصنيفها خطأ
- `append_summary_csv(...)` لإضافة ملخص تجربة في `results/summary.csv`

### (H) override_dropout_if_requested() — (اختياري)
هذه الدالة:
- لا تعمل إلا إذا أعطيت `--dropout`
- تحاول تعديل قيم dropout في `model.config` حسب نوع الموديل (BERT أو DistilBERT)
- الهدف: السماح بتجربة dropout بشكل صريح لو تريد مقارنة.

> مهم: حتى لو لم تستخدم `--dropout`، فـ Transformers فيها dropout افتراضي داخل المعمارية.

---

## 4) main(): أين يبدأ العمل الحقيقي؟

### الخطوة 1: قراءة الإعدادات وإنشاء المجلدات
- قراءة config عبر `parse_args()`
- إنشاء المجلدات عبر `ensure_dirs()`
- تثبيت الـ seed عبر `set_seed(seed)`

### الخطوة 2: تحميل IMDb وتقسيم Train/Val
- `load_dataset("imdb")`
- `base_train = raw["train"]`
- `test_ds = raw["test"]`
- ثم تقسيم base_train إلى train/val باستخدام `train_test_split(...)`

### الخطوة 3: تحميل Tokenizer و Tokenization
- `AutoTokenizer.from_pretrained(model_name, use_fast=True)`

ثم تعريف `tokenize_fn` لتطبيق:
- truncation
- max_length

ثم تطبيقه على:
- train
- val
- test

باستخدام:
- `dataset.map(tokenize_fn, batched=True, remove_columns=...)`

> remove_columns يحذف الأعمدة غير المطلوبة بعد tokenization لتكون البيانات جاهزة لـ Trainer.

### الخطوة 4: تحويل عمود label إلى labels
`Trainer` يتوقع اسم العمود:
- `labels`

لكن IMDb يعطي:
- `label`

لذلك يتم إعادة التسمية:
- `rename_column("label", "labels")`

### الخطوة 5: إعداد DataCollator
- `DataCollatorWithPadding(tokenizer=tokenizer)`

وظيفته:
- عمل padding ديناميكي لكل batch بدل padding ثابت.

### الخطوة 6: تحميل الموديل
- `AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)`

`num_labels=2` لأن المهمة Binary (positive/negative).

ثم (اختياري):
- تعديل dropout إذا `--dropout` موجود.

### الخطوة 7: إعداد TrainingArguments
هنا نحدد أهم إعدادات التدريب:
- learning_rate
- batch sizes
- num_train_epochs (ويمكن أن يتوقف قبلها بسبب early stopping)
- evaluation_strategy="epoch"
- save_strategy="epoch"
- load_best_model_at_end=True
- metric_for_best_model="f1"
- save_total_limit=2
- logging_steps=50
- report_to="none" (لا يستخدم wandb)

**ملاحظة مهمّة:**  
التقييم أثناء التدريب يتم على **Validation**، وليس Test.  
والـ Test يبقى للتقييم النهائي.

### الخطوة 8: إنشاء Trainer مع EarlyStoppingCallback
يتم إنشاء:
- `Trainer(...)`

ويُمرر له:
- model
- args
- train_dataset
- eval_dataset (validation)
- tokenizer
- data_collator
- compute_metrics
- callbacks = [EarlyStoppingCallback(...)]

**EarlyStoppingCallback**
- يوقف التدريب إذا لم يتحسن val_f1 لمدة (patience) epochs.

### الخطوة 9: التدريب
- `trainer.train()`

ويتم قياس وقت التدريب بـ:
- `time.time()` قبل وبعد.

### الخطوة 10: الحفظ + التقييم
بعد التدريب:
- حفظ الموديل والتوكنيزر داخل `outputs/<run_name>/`

ثم:
- تقييم على Validation (للشفافية)
- تقييم نهائي على Test (هذه هي النتائج التي تكتبها في التقرير)

ويتم حفظ كل شيء في:
- `outputs/<run_name>/metrics.json`

### الخطوة 11: تحليل النتائج (Confusion Matrix + Misclassified)
بعد التقييم:
- `trainer.predict(test_tok)` لإخراج logits
- تحويلها إلى y_pred باستخدام argmax
- حساب confusion matrix وحفظها
- استخراج أمثلة misclassified وحفظها في CSV

### الخطوة 12: summary.csv للتجارب
في النهاية يتم إضافة سطر في:
- `results/summary.csv`

لكي تجمع نتائج كل التجارب بسهولة (run_name، model، lr، f1، الوقت…).

---

## 5) الملفات التي تُنتج تلقائيًا (Outputs)

بعد تشغيل السكربت، سيتم إنشاء:

داخل `outputs/<run_name>/`:
- `metrics.json`  (نتائج val + test + config)
- `confusion_matrix.csv`
- `misclassified.csv`
- ملفات الموديل (weights/config/tokenizer files)

داخل `results/`:
- `summary.csv` (ملخص كل Run)

---

## 6) طريقة التشغيل (Commands)

### تشغيل Baseline (dropout الافتراضي للموديل)
```powershell
.\.venv\Scripts\python.exe imdb_train.py --run_name imdb_base
```

### تجربة dropout=0.2 (اختياري)
```powershell
.\.venv\Scripts\python.exe imdb_train.py --run_name imdb_drop02 --dropout 0.2
```

### تغيير نموذج إلى BERT
```powershell
.\.venv\Scripts\python.exe imdb_train.py --run_name imdb_bert --model_name bert-base-uncased
```

---

## 7) ملاحظات مهمة للتقرير

- **Iteration**: يحدث تحديث للأوزان لكل batch داخل Trainer (forward → loss → backward → update).
- **Batch**: يتم التحكم به عبر `train_batch_size`.
- **Early Stopping**: مطبق باستخدام `EarlyStoppingCallback` وعلى validation F1.
- **Dropout**: موجود داخل Transformers افتراضيًا، ويمكن تجربته صراحة عبر `--dropout` إن رغبت.