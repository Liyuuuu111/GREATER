import os
import random
import json
import pandas as pd

def process_test_samples_batch(test_data, output_file, ratio, batch_size=64, seed=42):
    random.seed(seed)

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i + batch_size]

            texts = [data['text'] for data in batch if data['label'] == 1]

            if texts:  
                modified_texts = process_paragraph_samples(texts, ratio, test_data)

                modified_texts_iter = iter(modified_texts)

                for data in batch:
                    if data['label'] == 1:
                        modified_text = next(modified_texts_iter)
                        modified_data = {
                            'text': modified_text,
                            'language': data.get('language', 'unknown'),
                            'label': data['label']
                        }
                    else:
                        modified_data = {
                            'text': data['text'],
                            'language': data.get('language', 'unknown'),
                            'label': data['label']
                        }
                    f_out.write(json.dumps(modified_data, ensure_ascii=False) + '\n')

    return None

def process_paragraph_samples(texts, ratio, test_data):
    k = ratio * 100

    processed_texts = []
    for text in texts:
        processed_text, num_characters_removed = remove_percentage_of_text(text, k)

        candidate_texts = [data['text'] for data in test_data if data['label'] == 0]
        prefix_text_from_label_0 = find_suitable_text(candidate_texts, k, num_characters_removed)

        processed_text = prefix_text_from_label_0 + " " + processed_text
        processed_texts.append(processed_text)

    return processed_texts

import random

def remove_percentage_of_text(text, k):
    words = text.split()  
    total_words = len(words)
    num_words_to_remove = int(total_words * k / 100)

    removed_words = words[:num_words_to_remove]
    remaining_words = words[num_words_to_remove:]
    
    remaining_text = ' '.join(remaining_words)
    return remaining_text.strip(), len(removed_words)

def find_suitable_text(candidate_texts, target_length):
    for text in candidate_texts:
        words = text.split()
        if len(words) >= target_length:
            extracted_text = ' '.join(words[:target_length])
            if abs(len(extracted_text) - target_length) <= 5:  
                return extracted_text
    
    random_text = random.choice(candidate_texts)
    random_words = random_text.split()
    return ' '.join(random_words[:target_length])

def process_paragraph_samples(texts, ratio, test_data):

    k = ratio * 100

    processed_texts = []
    for text in texts:
        remaining_text, num_words_removed = remove_percentage_of_text(text, k)

        candidate_texts = [data['text'] for data in test_data if data['label'] == 0]
        suitable_text = find_suitable_text(candidate_texts, num_words_removed)

        processed_text = suitable_text + " " + remaining_text
        processed_texts.append(processed_text)

    return processed_texts

if __name__ == "__main__":
    file_path = r'/home/liyuanfan/test/dataset/english.jsonl'
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
    
    target_directory = './dataset/human_ob'
    os.makedirs(target_directory, exist_ok=True)
    
    for ratio in [0.02, 0.05, 0.1, 0.15, 0.2, 0.25]:
        output_file = os.path.join(target_directory, "processed.jsonl")
        avg_bert_score = process_test_samples_batch(test_data, output_file, ratio=ratio)
        final_output_file = os.path.join(target_directory, f"confusion_ratio{ratio:.2f}.jsonl")
        os.rename(output_file, final_output_file)
        print(f"Processed ratio= {ratio}")
