from transformers import BertForTokenClassification, BertTokenizer

# 加载模型和分词器
model = BertForTokenClassification.from_pretrained('./output')
tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm')

# 输入测试文本
test_text = "人民日报 1月1日 报道"
inputs = tokenizer(test_text, return_tensors="pt", is_split_into_words=True)

# 模型预测
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=-1)

# 获取 token 和预测标签
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
pred_labels = predictions[0].tolist()

# 打印预测结果
for token, label in zip(tokens, pred_labels):
    print(f"{token}: {label}")
