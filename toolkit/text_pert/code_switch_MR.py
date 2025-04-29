import json
import os
import random
import torch
import pandas as pd
from bert_score import score
from transformers import MarianMTModel, MarianTokenizer
import glob
# 定义翻译模型路径
model_paths = {
    'ar': 'Helsinki-NLP/opus-mt-en-ar',     # todo: download the model and put them into this location
    'ru': 'Helsinki-NLP/opus-mt-en-ru', 
    'de': 'Helsinki-NLP/opus-mt-en-de'   
}

# 加载翻译模型和tokenizer
def load_translation_models(model_paths, device):
    models = {}
    tokenizers = {}
    for lang, path in model_paths.items():
        model = MarianMTModel.from_pretrained(path).to(device) 
        tokenizer = MarianTokenizer.from_pretrained(path)
        models[lang] = model
        tokenizers[lang] = tokenizer
    return models, tokenizers

# 选择设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载所有语言对的翻译模型到指定设备
translation_models, translation_tokenizers = load_translation_models(model_paths, device)

def translate_text(text, model, tokenizer, device):
    batch = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)  # 将输入移动到设备上
    gen = model.generate(**batch)
    translated = tokenizer.batch_decode(gen, skip_special_tokens=True)
    return translated[0]

def process_paragraph_sample(text, translation_models, translation_tokenizers, probability=0.4, random_seed=42):
    random.seed(random_seed)

    words = text.split()  
    total_words = len(words)
    num_words_to_replace = int(total_words * probability)

    # 随机选择要替换的单词索引
    words_to_replace = random.sample(range(total_words), num_words_to_replace)

    # 定义要翻译的语言
    languages = ['ar', 'ru', 'de']  # 阿拉伯语、俄语、德语

    # 替换选中的单词或词组
    modified_words = words[:]
    for i in words_to_replace:
        if i <= total_words - 4:
            phrase_length = random.randint(4, min(6, total_words - i))
            phrase = ' '.join(words[i:i + phrase_length]).lower() 
        else:
            continue

        lang = random.choice(languages)
        model = translation_models[lang]
        tokenizer = translation_tokenizers[lang]
        translated_phrase = translate_text(phrase, model, tokenizer, device)
        
        # 替换原词组
        modified_words[i:i + phrase_length] = translated_phrase.split()

    modified_text = ' '.join(modified_words)

    return modified_text

# 新增的函数，用于批量处理BERT分数
def calculate_bert_score_batch(original_texts, modified_texts):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device: ", device)
    P, R, F1 = score(modified_texts, original_texts, lang="en", model_type="bert-base-uncased", device=device)
    return F1.cpu().numpy() 

def process_test_samples_batch(test_data, output_file, prob, batch_size=64, seed=42):
    # 设置随机种子
    random.seed(seed)

    total_bert_score = 0
    count = 0

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i + batch_size]
            
            texts = [data['text'] for data in batch if data['label'] == 1]

            if texts:  
                modified_texts = [process_paragraph_sample(text, translation_models, translation_tokenizers, probability=prob) for text in texts]

                bert_scores = calculate_bert_score_batch(texts, modified_texts)

                total_bert_score += bert_scores.sum()
                count += len(bert_scores) 

                modified_texts_iter = iter(modified_texts)
                bert_scores_iter = iter(bert_scores)

                for data in batch:
                    if data['label'] == 1:
                        modified_text = next(modified_texts_iter)
                        bert_score = next(bert_scores_iter)
                        modified_data = {
                            'text': modified_text,
                            'bert_score': float(bert_score),
                            'language': data.get('language', 'unknown'),
                            'label': data['label']
                        }
                    else:
                        modified_data = {
                            'text': data['text'],
                            'bert_score': None,
                            'language': data.get('language', 'unknown'),
                            'label': data['label']
                        }
                    # 将数据写入输出文件
                    f_out.write(json.dumps(modified_data, ensure_ascii=False) + '\n')

    # 计算所有进行过修改的 BERT 分数均值
    avg_bert_score = total_bert_score / count if count > 0 else 0

    return avg_bert_score


if __name__ == "__main__":
    file_path = r''             # todo: put your file here
    data = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)

    df_train_sampled = df.sample(n=10000, random_state=42)

    train_indices = df_train_sampled.index

    df_remaining = df.drop(index=train_indices)

    df_test_sampled = df_remaining.sample(n=2000, random_state=42)

    test_data = df_test_sampled.to_dict(orient='records')

    target_directory = './dataset/code-switch-MR'
    os.makedirs(target_directory, exist_ok=True)
    for probability_threshold in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
        output_file = os.path.join(target_directory, "processed.jsonl")
        avg_bert_score = process_test_samples_batch(test_data, output_file, prob=probability_threshold)
        final_output_file = os.path.join(target_directory, f"switch_bert{avg_bert_score:.2f}.jsonl")
        os.rename(output_file, final_output_file)
        print(f"Processed prob = {probability_threshold}")