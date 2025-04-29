import json
import os
import random
import torch
import pandas as pd
from bert_score import score

def load_translation_dict(file_path):
    translation_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for entry in data:
            original = entry['original'].strip().lower()  #
            translated = entry['translated'].strip()
            translation_dict[original] = translated
    return translation_dict

# 加载所有语言对的翻译字典
translation_dicts = {
    'ar': load_translation_dict(''),            # todo: put your dictionary here
    'ru': load_translation_dict(''),        # todo: put your dictionary here
    'de': load_translation_dict('')  # todo: put your dictionary here
}

def process_paragraph_sample(text, translation_dicts, probability=0.4, random_seed=42):
    random.seed(random_seed)

    words = text.split()  
    total_words = len(words)
    num_words_to_replace = int(total_words * probability)

    words_to_replace = random.sample(range(total_words), num_words_to_replace)

    languages = ['ar', 'ru', 'de']  

    modified_words = words[:]
    for i in words_to_replace:
        word = words[i].lower()  
        if word.isalpha():  
            lang = random.choice(languages)
            translated_word = translation_dicts[lang].get(word, word)  
            modified_words[i] = translated_word

    modified_text = ' '.join(modified_words)

    return modified_text

def calculate_bert_score_batch(original_texts, modified_texts):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device: ", device)
    P, R, F1 = score(modified_texts, original_texts, lang="en", model_type="bert-base-uncased", device=device)
    return F1.cpu().numpy()  

def process_test_samples_batch(test_data, output_file, prob, batch_size=64, seed=42):
    # 设置随机种子
    random.seed(seed)

    # 初始化BERT分数总和
    total_bert_score = 0
    count = 0

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i + batch_size]
            
            texts = [data['text'] for data in batch if data['label'] == 1]

            if texts:  
                modified_texts = [process_paragraph_sample(text, translation_dicts, probability=prob) for text in texts]
                
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
                    f_out.write(json.dumps(modified_data, ensure_ascii=False) + '\n')

            if count % 100 == 0:
                print("Processed: ", count)

    if count > 0:
        avg_bert_score = total_bert_score / count
    else:
        avg_bert_score = 0

    return avg_bert_score

if __name__ == "__main__":
    file_path = r''         # todo: put your file here
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
    target_directory = './dataset/code-switch-MF'
    os.makedirs(target_directory, exist_ok=True)
    for probability_threshold in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]:
        output_file = os.path.join(target_directory, "processed.jsonl")
        avg_bert_score = process_test_samples_batch(test_data, output_file, prob=probability_threshold)
        final_output_file = os.path.join(target_directory, f"switch_bert{avg_bert_score:.2f}.jsonl")
        os.rename(output_file, final_output_file)
        print(f"Processed prob = {probability_threshold}")

