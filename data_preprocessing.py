from transformers import BertTokenizer

# 加载中文 BERT 分词器
tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm')


def load_data(text_file, label_file):
    """
    读取文本数据和标签数据
    """
    with open(text_file, 'r', encoding='utf-8') as f_text, open(label_file, 'r', encoding='utf-8') as f_label:
        texts = f_text.readlines()
        labels = f_label.readlines()
    return texts, labels


def preprocess_data(texts, labels):
    """
    将输入文本和标签进行tokenization，并对齐。
    """
    tokenized_inputs = tokenizer(texts, truncation=True, padding=True, is_split_into_words=True)

    # 处理标签
    label_ids = []
    for label in labels:
        label_ids.append([label_map[l.strip()] for l in label.split()])  # 对应标签处理
    return tokenized_inputs, label_ids


# 标签映射
label_map = {'O': 0, 'B-LOC': 1, 'I-LOC': 2, 'B-PER': 3, 'I-PER': 4, 'B-ORG': 5, 'I-ORG': 6}  # 根据数据调整标签映射

# 从文件加载数据
texts, labels = load_data('data/source_BIO_2014_corpus.txt', 'data/target_BIO_2014_corpus.txt')

# 数据预处理
tokenized_inputs, label_ids = preprocess_data(texts, labels)
