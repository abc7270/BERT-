from transformers import BertTokenizerFast

# 使用 Fast tokenizer 才能 word_ids() 对齐
tokenizer = BertTokenizerFast.from_pretrained("hfl/chinese-bert-wwm")

# 标签映射（必须覆盖你 target 里出现的所有标签）
label_map = {
    "O": 0,
    "B_LOC": 1, "I_LOC": 2,
    "B_PER": 3, "I_PER": 4,
    "B_ORG": 5, "I_ORG": 6,
    "B_T": 7, "I_T": 8
}


def load_data(text_file, label_file):
    """
    读取文本数据和标签数据（每行对应一条样本）
    """
    with open(text_file, "r", encoding="utf-8") as f_text, open(label_file, "r", encoding="utf-8") as f_label:
        texts = f_text.readlines()
        labels = f_label.readlines()
    return texts, labels


def preprocess_data(texts, labels, max_length=512):
    all_text_chars = []
    all_label_ids = []

    skip_count = 0

    for text_line, label_line in zip(texts, labels):
        text_line = text_line.strip()
        label_tokens = label_line.strip().split()

        # 如果标签为空，跳过
        if len(label_tokens) == 0:
            skip_count += 1
            continue

        # 关键：不再要求严格相等，而是取最短长度对齐
        min_len = min(len(text_line), len(label_tokens))

        if min_len == 0:
            skip_count += 1
            continue

        text_line = text_line[:min_len]
        label_tokens = label_tokens[:min_len]

        chars = list(text_line)
        all_text_chars.append(chars)

        ids = [label_map.get(tag, 0) for tag in label_tokens]
        all_label_ids.append(ids)

    print("跳过的样本数:", skip_count)
    print("保留的样本数:", len(all_text_chars))

    encodings = tokenizer(
        all_text_chars,
        is_split_into_words=True,
        truncation=True,
        padding=True,
        max_length=max_length
    )

    aligned_labels = []
    for i in range(len(all_label_ids)):
        word_ids = encodings.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(all_label_ids[i][word_idx])
            else:
                label_ids.append(all_label_ids[i][word_idx])
            previous_word_idx = word_idx
        aligned_labels.append(label_ids)

    return encodings, aligned_labels

