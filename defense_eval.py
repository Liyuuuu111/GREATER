import os
import json
import numpy as np
import pandas as pd
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import (
    XLMRobertaTokenizer, XLMRobertaForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    AlbertTokenizer, AlbertForSequenceClassification,
    DebertaTokenizer, DebertaForSequenceClassification,
    ElectraTokenizer, ElectraForSequenceClassification
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Calculate ASR (Attack Success Rate) for a model.")
    parser.add_argument("--original_file", type=str, required=True, help="Path to the original dataset file")
    parser.add_argument("--attacked_file", type=str, required=True, help="Path to the attacked dataset file")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory where the pre-trained models are located")
    return parser.parse_args()

# Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def predict(model, dataloader, texts):
    model.eval()
    predictions = []
    true_labels = []
    correct_classified_texts = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            correct_indices = (preds == labels).cpu().numpy().astype(bool)
            correct_classified_texts.extend(np.array(texts[i*batch['input_ids'].size(0):(i+1)*batch['input_ids'].size(0)])[correct_indices])

            predictions.append(logits.cpu().numpy())
            true_labels.append(labels.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    return predictions, true_labels, correct_classified_texts

def get_correct_indices(model, dataloader, target_label=1):
    model.eval()
    correct_indices = []
    current_index = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            preds_np = preds.cpu().numpy()
            labels_np = labels_batch.cpu().numpy()
            batch_size = input_ids.size(0)
            for i in range(batch_size):
                if labels_np[i] == target_label and preds_np[i] == target_label:
                    correct_indices.append(current_index + i)
            current_index += batch_size
    return correct_indices

# Main logic
if __name__ == "__main__":
    args = parse_args()

    # Load original data
    original_data = []
    with open(args.original_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                original_data.append(json.loads(line))
            except:
                continue
        
    df_original = pd.DataFrame(original_data)
    texts_original = df_original['text'].tolist()
    labels_original = df_original['label'].tolist()

    # Load attacked data
    attacked_data = []
    with open(args.attacked_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                attacked_data.append(json.loads(line))
            except:
                continue

    df_attacked = pd.DataFrame(attacked_data)
    texts_attacked = df_attacked['text'].tolist()
    labels_attacked = df_attacked['label'].tolist()

    print("Start calculating ASR...")
    asr_models = [
        "ALBERT_Base",
        "ALBERT_Large",
        "DeBERTa_Base",
        "DeBERTa_Large",
        "RoBERTa_Base",
        "RoBERTa_Large",
        "XLM-RoBERTa_Large"
    ]

    for asr_model in asr_models:
        print(f"\nCalculating ASR for model: {asr_model}")
        model_dir = os.path.join(args.model_dir, f"base_model_{asr_model}")
        if not os.path.exists(model_dir):
            print(f"Model directory {model_dir} does not exist, skipping.")
            continue

        checkpoints = [d for d in os.listdir(model_dir) if d.startswith('checkpoint')]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
            latest_checkpoint = checkpoints[-1]
            model_path = os.path.join(model_dir, latest_checkpoint)
        else:
            model_path = model_dir

        asr_model_lower = asr_model.lower()
        if "albert" in asr_model_lower:
            model_class = AlbertForSequenceClassification
            tokenizer_path = "albert-base-v2"
            tokenizer_class = AlbertTokenizer
        elif "roberta" in asr_model_lower and "xlm" not in asr_model_lower:
            if "large" in asr_model_lower:
                tokenizer_path = "roberta-large"
            else:
                tokenizer_path = "roberta-base"
            model_class = RobertaForSequenceClassification
            tokenizer_class = RobertaTokenizer
        elif "xlm" in asr_model_lower or "xlm-roberta" in asr_model_lower:
            tokenizer_path = "xlm-roberta-large"
            model_class = XLMRobertaForSequenceClassification
            tokenizer_class = XLMRobertaTokenizer
        elif "deberta" in asr_model_lower:
            if "large" in asr_model_lower:
                tokenizer_path = "deberta-large"
            else:
                tokenizer_path = "deberta-base"
            model_class = DebertaForSequenceClassification
            tokenizer_class = DebertaTokenizer
        elif "electra" in asr_model_lower:
            if "large" in asr_model_lower:
                tokenizer_path = "electra-large-discriminator"
            else:
                tokenizer_path = "electra-base-discriminator"
            model_class = ElectraForSequenceClassification
            tokenizer_class = ElectraTokenizer
        else:
            tokenizer_path = "xlm-roberta-base"
            model_class = XLMRobertaForSequenceClassification
            tokenizer_class = XLMRobertaTokenizer

        asr_model_instance = model_class.from_pretrained(model_path)
        asr_model_instance.to(device)
        asr_tokenizer = tokenizer_class.from_pretrained(tokenizer_path)

        original_dataset_asr = TextDataset(texts_original, labels_original, asr_tokenizer, max_len=256)
        original_dataloader_asr = DataLoader(original_dataset_asr, batch_size=256, shuffle=False)
        correct_indices = get_correct_indices(asr_model_instance, original_dataloader_asr, target_label=1)
        print(f"Model {asr_model} gives true predictions: {len(correct_indices)}")

        if len(correct_indices) == 0:
            print(f"Model {asr_model} can't get any true classification, skipping the calculation of ASR.")
            continue

        attacked_dataset_asr = TextDataset(texts_attacked, labels_attacked, asr_tokenizer, max_len=256)
        attacked_dataloader_asr = DataLoader(attacked_dataset_asr, batch_size=256, shuffle=False)
        asr_model_instance.eval()
        attacked_preds = []
        with torch.no_grad():
            for batch in attacked_dataloader_asr:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = asr_model_instance(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                attacked_preds.extend(preds.cpu().numpy().tolist())

        total = len(correct_indices)
        correct_attacked = sum(1 for idx in correct_indices if attacked_preds[idx] == 1)
        attacked_accuracy = correct_attacked / total
        ASR = 1 - attacked_accuracy
        print(f"Model {asr_model}'s ACC after attack: {attacked_accuracy:.4f}, ASR: {ASR:.4f}")

    print("\nFinish calculating ASR!")
