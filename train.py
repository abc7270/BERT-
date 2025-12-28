from transformers import BertForTokenClassification, Trainer, TrainingArguments
from model import NERDataset
from data_preprocessing import preprocess_data, label_map, load_data

# 加载数据并进行预处理
train_texts, train_labels = load_data('data/source_BIO_2014_corpus.txt', 'data/target_BIO_2014_corpus.txt')
train_encodings, train_label_ids = preprocess_data(train_texts, train_labels)

# 创建数据集对象
train_dataset = NERDataset(train_encodings, train_label_ids)

# 加载模型
model = BertForTokenClassification.from_pretrained('hfl/chinese-bert-wwm', num_labels=len(label_map))

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./output',          # 输出目录
    evaluation_strategy="epoch",    # 每个 Epoch 后评估
    learning_rate=5e-5,
    per_device_train_batch_size=16, # 批量大小
    num_train_epochs=3,             # 训练 Epoch 数
)

# 设置 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,
)

# 开始训练
trainer.train()
