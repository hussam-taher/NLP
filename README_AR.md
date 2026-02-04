# مشروع تصنيف النصوص (IMDb Sentiment Classification) باستخدام Transformers

> هذا الدليل موجّه لك كطالب/منفّذ للمشروع: يشرح **ماذا فعلنا** خطوة بخطوة، وكيف تشغّل التجارب بنفسك، وأين تجد **ملفات النتائج** بسرعة.
> **تنبيه مهم (اقتباس/تكييف الكود):** تم تكييف هذا المشروع من مثال Hugging Face Transformers `run_glue.py`، ثم عُدّل ليصبح مناسبًا لتصنيف المشاعر على IMDb، مع إضافة مقاييس إضافية وحفظ النتائج.

<!--
تم تكييف هذا المشروع من مثال Hugging Face Transformers (run_glue.py)، ثم عُدّل ليصبح مناسبًا لتصنيف المشاعر على IMDb،
مع إضافة: تقسيم Validation، الإيقاف المبكر Early Stopping، مقاييس إضافية، وحفظ النتائج (JSON/CSV).
-->

---

## 1) فكرة المشروع باختصار
نقوم بتطبيق **تصنيف النصوص (Text Classification)** على بيانات **IMDb** (مراجعات أفلام) لتحديد ما إذا كانت المراجعة:
- **0 = سلبية (Negative)**
- **1 = إيجابية (Positive)**

السكربت الرئيسي الذي نفّذنا به كل شيء هو:
- `imdb_train.py`

---

## 2) بيئة العمل والمتطلبات

### 2.1 بيئة التشغيل
- Python 3.9+ (يفضّل 3.10 أو 3.11)
- يفضّل تشغيل المشروع داخل Virtual Environment (`.venv`)

### 2.2 أهم المكتبات ولماذا نستخدمها
- **transformers**  
  لتحميل الموديل والـ Tokenizer، وإدارة التدريب عبر `Trainer` و`TrainingArguments`، وتفعيل `EarlyStoppingCallback`.

- **datasets**  
  لتحميل بيانات IMDb مباشرة باستخدام: `load_dataset("imdb")`، وكذلك عمل تقسيم Train/Validation.

- **torch (PyTorch)**  
  هو محرك التدريب الفعلي (backprop + تحديث الأوزان).  
  > ملاحظة: `Trainer` من transformers يبني فوق PyTorch.

- **numpy**  
  للتعامل مع المصفوفات وتحويل logits إلى توقعات بـ `argmax`.

- **scikit-learn (sklearn.metrics)**  
  لحساب: Accuracy و Precision و Recall و F1، وكذلك Confusion Matrix.

- **argparse**  
  لتغيير الإعدادات من سطر الأوامر (اختيار موديل، batch size، طول النص...).

- **json / csv / pathlib.Path**  
  لحفظ النتائج في ملفات (JSON/CSV) وإنشاء مجلدات الإخراج تلقائيًا.

---

## 3) البيانات: كيف نحصل على IMDb؟
نستخدم مكتبة `datasets` ونحمّل الداتا بهذا السطر داخل الكود:

- `raw = load_dataset("imdb")`

وهذا يقوم تلقائيًا بـ:
1) تنزيل البيانات (أول مرة فقط)
2) تخزينها في cache
3) إعطائك Splits جاهزة:
   - `raw["train"]`
   - `raw["test"]`

> ملاحظة مهمة: الـ dataset تأتي من Hugging Face Datasets (مصدر خارجي)، لكن طريقة التحميل سهلة ومباشرة عبر `load_dataset`.

---

## 4) ما الذي يحدث داخل الكود؟ (شرح عملي موجّه لك)
السكريبت يمشي بهذا التسلسل (هذا أهم ما تحتاجه لفهم التنفيذ):

### 4.1 تقسيم Train / Validation
لأن **Early Stopping** يحتاج مجموعة Validation، قمنا بتقسيم `train` إلى:
- Train
- Validation

باستخدام:
- `train_test_split(test_size=val_ratio, seed=seed, shuffle=True)`

والافتراضي:
- `val_ratio = 0.1` (أي 10% من train تصبح validation)

### 4.2 Tokenization وتجهيز النصوص
نحمّل Tokenizer مطابق للموديل:
- `AutoTokenizer.from_pretrained(model_name, use_fast=True)`

ثم نطبّق:
- `truncation=True`
- `max_length=max_seq_length`

بعدها نستخدم `map(...)` لتطبيق التحويل على:
- train / val / test

### 4.3 Padding (تجهيز الدُفعات Batches)
بدل ما نسوي padding ثابت من البداية، استخدمنا:
- `DataCollatorWithPadding(tokenizer=tokenizer)`

هذا يسوي padding ديناميكي لكل batch أثناء التدريب والتقييم.

### 4.4 تحميل الموديل (Model)
نستخدم:
- `AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)`

لأنها مهمة ثنائية (0/1).

### 4.5 التدريب (Training) والتقييم (Evaluation)
نضبط التدريب عبر `TrainingArguments` (مثل learning rate / batch size / epochs ...)، ثم ننشئ:
- `Trainer(...)`

وأخيرًا:
- `trainer.train()`  → يبدأ التدريب
- `trainer.evaluate(val)` → تقييم على validation أثناء التدريب
- `trainer.evaluate(test)` / `trainer.predict(test)` → تقييم نهائي على test + استخراج التوقعات

---

## 5) المفاهيم الأربعة المطلوبة: أين تظهر في مشروعنا؟
هذه المفاهيم الأربعة ظهرت بشكل عملي داخل التدريب:

1) **Iteration (تحديث واحد للوزن)**  
   كل Batch يمر بـ forward → loss → backward → تحديث الأوزان. هذا يحدث تلقائيًا داخل `Trainer`.

2) **Batch**  
   تتحكم فيها من سطر الأوامر:
   - `--train_batch_size`
   - `--eval_batch_size`

3) **Early Stopping**  
   مطبق عبر:
   - `EarlyStoppingCallback(...)`  
   ويوقف التدريب إذا لم يتحسن أداء validation (حسب F1) لعدد epochs محدد.

4) **Dropout**  
   موجود تلقائيًا داخل معماريات BERT/DistilBERT.  
   ويمكنك (اختياريًا) تجربته صراحة عبر:
   - `--dropout 0.2`  
   لكن في تجاربنا الحالية تركناه على الافتراضي (dropout = null في config).

---

## 6) التجارب التي نفّذناها (Run1 وRun2) + النتائج
قمنا بتجربتين أساسيتين لمقارنة نموذج خفيف مقابل نموذج أكبر:

### 6.1 التجربة الأولى: DistilBERT (run1_distilbert)
- الموديل: `distilbert-base-uncased`
- max_seq_length = 256
- train_batch_size = 16
- eval_batch_size = 32
- learning_rate = 2e-5
- weight_decay = 0.01
- val_ratio = 0.1
- early_stopping_patience = 2

**نتائج Validation**
- Accuracy = 0.9092
- F1 = 0.90996

**نتائج Test**
- Accuracy = 0.9132
- Precision = 0.90163
- Recall = 0.9276
- F1 = 0.91443
- زمن التدريب ≈ 2486 ثانية

### 6.2 التجربة الثانية: BERT (run2_bert)
- الموديل: `bert-base-uncased`
- max_seq_length = 256
- train_batch_size = 8
- eval_batch_size = 16
- learning_rate = 2e-5
- weight_decay = 0.01
- val_ratio = 0.1
- early_stopping_patience = 2

**نتائج Validation**
- Accuracy = 0.9112
- F1 = 0.91183

**نتائج Test**
- Accuracy = 0.91652
- Precision = 0.90967
- Recall = 0.92488
- F1 = 0.91721
- زمن التدريب ≈ 4187 ثانية

### 6.3 جدول مقارنة سريع (الأهم في التقرير)
| التجربة | الموديل | Test Accuracy | Test F1 | Train Time (sec) |
|---|---|---:|---:|---:|
| run1_distilbert | distilbert-base-uncased | 0.9132 | 0.9144 | 2486 |
| run2_bert | bert-base-uncased | 0.9165 | 0.9172 | 4187 |

**ملاحظة تفسيرية للقارئ:**  
BERT أعطى تحسنًا بسيطًا في الدقة وF1 مقارنة بـ DistilBERT، لكن زمن التدريب كان أعلى لأن النموذج أكبر.

---

## 7) Confusion Matrix (تحليل أخطاء) — مثال من DistilBERT
قمنا بحفظ مصفوفة الالتباس لتفسير الأخطاء. مثال (DistilBERT على test):

- true_0 → pred_0 = 11235  (سالب صحيح)
- true_0 → pred_1 = 1265   (False Positive)
- true_1 → pred_0 = 905    (False Negative)
- true_1 → pred_1 = 11595  (موجب صحيح)

**كيف تقرأها؟**
- النموذج جيد في اكتشاف الإيجابي (Recall مرتفع)، لكنه أحيانًا يصنّف بعض المراجعات السلبية كإيجابية (FP).

---

## 8) الملفات التي تُنتَج تلقائيًا وأين تجدها؟
بعد تشغيل أي تجربة، يتم إنشاء:

داخل `outputs/<run_name>/`:
- `metrics.json`  (نتائج val + test + config)
- `confusion_matrix.csv`
- `misclassified.csv` (اختياري: أمثلة أخطاء)
- ملفات الموديل والتوكنيزر

داخل `results/`:
- `summary.csv` (ملخص سريع لكل التجارب)

> أنت لا تحتاج لإنشاء هذه الملفات يدويًا: الكود ينشئها تلقائيًا.

---

## 9) طريقة التشغيل (Commands)

### 9.1 تشغيل DistilBERT
```powershell
.\.venv\Scripts\python.exe imdb_train.py ^
  --run_name run1_distilbert ^
  --model_name distilbert-base-uncased ^
  --max_seq_length 256 ^
  --train_batch_size 16 ^
  --eval_batch_size 32 ^
  --learning_rate 2e-5 ^
  --num_train_epochs 5 ^
  --val_ratio 0.1 ^
  --early_stopping_patience 2
```

### 9.2 تشغيل BERT
```powershell
.\.venv\Scripts\python.exe imdb_train.py ^
  --run_name run2_bert ^
  --model_name bert-base-uncased ^
  --max_seq_length 256 ^
  --train_batch_size 8 ^
  --eval_batch_size 16 ^
  --learning_rate 2e-5 ^
  --num_train_epochs 5 ^
  --val_ratio 0.1 ^
  --early_stopping_patience 2
```

### 9.3 اختبار سريع (Smoke Test) للتأكد أن كل شيء يعمل
```powershell
.\.venv\Scripts\python.exe imdb_train.py ^
  --run_name smoke_test ^
  --model_name distilbert-base-uncased ^
  --max_seq_length 128 ^
  --train_batch_size 8 ^
  --eval_batch_size 16 ^
  --num_train_epochs 1 ^
  --val_ratio 0.1 ^
  --early_stopping_patience 1
```