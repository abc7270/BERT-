import numpy as np
from transformers import BertForTokenClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

from model import NERDataset
from data_preprocessing import preprocess_data, load_data, label_map

# ---------------------------
# 1) Load raw data
# ---------------------------
texts, labels = load_data(
    "data/source_BIO_2014_cropus.txt",
    "data/target_BIO_2014_cropus.txt"
)

print("Total samples:", len(texts))

# ---------------------------
# 2) Split first (IMPORTANT)
# ---------------------------
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.1, random_state=42
)

print("Train samples:", len(train_texts))
print("Val samples:", len(val_texts))
print("验证集前5条标签示例：")
for i in range(5):
    print(val_labels[i][:200])


# ---------------------------
# 3) Preprocess (tokenize + align labels)
# ---------------------------
train_encodings, train_label_ids = preprocess_data(train_texts, train_labels, max_length=256)
val_encodings, val_label_ids = preprocess_data(val_texts, val_labels, max_length=256)

train_dataset = NERDataset(train_encodings, train_label_ids)
eval_dataset = NERDataset(val_encodings, val_label_ids)

# ---------------------------
# 4) Load model
# ---------------------------
model = BertForTokenClassification.from_pretrained(
    "hfl/chinese-bert-wwm",
    num_labels=len(label_map)
)

# id -> label
id2label = {v: k for k, v in label_map.items()}
label2id = {k: v for k, v in label_map.items()}

model.config.id2label = id2label
model.config.label2id = label2id

# ---------------------------
# 5) Metrics (NER evaluation using seqeval)
# ---------------------------
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = []
    true_predictions = []

    for pred, lab in zip(predictions, labels):
        temp_true_labels = []
        temp_true_preds = []
        for p_i, l_i in zip(pred, lab):
            if l_i != -100:  # ignore special tokens
                temp_true_labels.append(id2label[l_i])
                temp_true_preds.append(id2label[p_i])
        true_labels.append(temp_true_labels)
        true_predictions.append(temp_true_preds)

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }

# ---------------------------
# 6) Training arguments (GPU optimized)
# ---------------------------
training_args = TrainingArguments(
    output_dir="./output",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,

    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,

    num_train_epochs=1,        # ✅ 先跑 1 epoch
    weight_decay=0.01,

    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,

    save_total_limit=2,
    report_to="none"
)


# ---------------------------
# 7) Trainer
# ---------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# ---------------------------
# 8) Train
# ---------------------------
trainer.train()

# ---------------------------
# 9) Final evaluation report
# ---------------------------
preds = trainer.predict(eval_dataset)
predictions = np.argmax(preds.predictions, axis=2)
labels = preds.label_ids

true_labels = []
true_predictions = []

for pred, lab in zip(predictions, labels):
    temp_true_labels = []
    temp_true_preds = []
    for p_i, l_i in zip(pred, lab):
        if l_i != -100:
            temp_true_labels.append(id2label[l_i])
            temp_true_preds.append(id2label[p_i])
    true_labels.append(temp_true_labels)
    true_predictions.append(temp_true_preds)

print("\n==== Detailed Classification Report ====\n")

# 统计实体数量（非O标签）
entity_count = sum(
    1 for sent in true_labels for lab in sent if lab != "O"
)

if entity_count == 0:
    print("Warning: Evaluation set contains NO entity labels (all 'O').")
    print("Possible reasons:")
    print("1) 数据对齐时大量样本被跳过")
    print("2) max_length 截断导致实体被截掉")
    print("3) label_map 标签不匹配，导致实体变成 O")
else:
    print(classification_report(true_labels, true_predictions))
