import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM, XLMRobertaTokenizer, XLMRobertaForSequenceClassification, AutoModelForSequenceClassification, AutoModel, AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel
import torch.nn.functional as F
import random
import numpy as np
from difflib import SequenceMatcher
import argparse
import os
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

parser = argparse.ArgumentParser(description='GREATER Query mode')
parser.add_argument('--zero-query', action='store_true', help='Zero-query mode')
parser.add_argument('--method', type=str, required=True, choices=['perturbation', 'mask'], help='Select the method of generating adversary example')
parser.add_argument('--model-path', type=str, required=True, help='Set the location of model')

args = parser.parse_args()

model_path = ""         # put your USE model path here
use_tokenizer = AutoTokenizer.from_pretrained(model_path)
use_model = AutoModel.from_pretrained(model_path).to(device)

model_name = ""
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForMaskedLM.from_pretrained(model_name).to(device)
model.eval()  

def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    return tokenizer, model
'''
classifier_tokenizer, classifier_model = load_model_and_tokenizer(args.model_path)
'''
classifier_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
classifier_model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device)

gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

generator_save_path = ''

importance_module = nn.Sequential(
    nn.Linear(model.config.hidden_size, 256),
    nn.ReLU(),
    nn.Linear(256, 1)
).to(device)

try:  
    importance_module_path = os.path.join(generator_save_path, 'importance_module.pt')
    importance_module.load_state_dict(torch.load(importance_module_path, map_location=device))
except:
    importance_module_path = os.path.join(generator_save_path, 'pytorch_model.bin')
    importance_module.load_state_dict(torch.load(importance_module_path, map_location=device))

query = 0  

def calculate_gpt2_ppl(text):
    inputs = gpt2_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = gpt2_model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        ppl = torch.exp(loss).item()
    return ppl

def calculate_use_similarity(original_text, adversarial_text):
    original_inputs = use_tokenizer(original_text, return_tensors='pt', padding=True, truncation=True).to(device)
    adversarial_inputs = use_tokenizer(adversarial_text, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        original_emb = use_model(**original_inputs).last_hidden_state.mean(dim=1)
        adversarial_emb = use_model(**adversarial_inputs).last_hidden_state.mean(dim=1)
    similarity = F.cosine_similarity(original_emb, adversarial_emb).item()
    return similarity

def query_classifier(text):
    global query
    query += 1
    inputs = classifier_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = classifier_model(**inputs)
    return torch.argmax(outputs.logits, dim=-1).item()

def clean_special_tokens(text):
    return text.replace("</s>", "").replace("<s>", "").strip()

def is_valid_token(token):
    clean_token = token.lstrip('Ġ').strip()
    if not clean_token or clean_token in ['<s>', '</s>', '<pad>', '<mask>', '<unk>', 'Ċ']:
        return False
    if all(char in string.punctuation for char in clean_token):
        return False
    if any(char.isalnum() for char in clean_token):
        return True
    return False

def get_important_token_indices(model, input_ids, num_important_tokens):
    input_ids = input_ids[:, :512]
    inputs_embeds = model.roberta.embeddings(input_ids=input_ids.to(device))
    inputs_embeds.retain_grad()
    outputs = model(inputs_embeds=inputs_embeds)
    target_label = input_ids.view(-1)
    loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), target_label)
    loss.backward()
    grad = inputs_embeds.grad[0]  

    token_grad_norms = torch.norm(grad, dim=-1)  # [seq_len]

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    valid_indices = []
    valid_grad_norms = []
    for idx, (token, norm) in enumerate(zip(tokens, token_grad_norms)):
        if 1:
            valid_indices.append(idx)
            valid_grad_norms.append(norm.item())

    if not valid_indices:
        return []

    num_tokens = min(num_important_tokens, len(valid_indices))
    valid_grad_norms = torch.tensor(valid_grad_norms)
    topk_indices_in_valid = torch.topk(valid_grad_norms, num_tokens).indices.tolist()
    important_token_indices = [valid_indices[i] for i in topk_indices_in_valid]
    return important_token_indices

'''
def get_important_token_indices(model, input_ids, num_important_tokens, seed=42):
    input_ids = input_ids[:, :512]

    if seed is not None:
        random.seed(seed)

    token_indices = list(range(input_ids.size(1)))

    num_tokens = min(num_important_tokens, len(token_indices))
    important_token_indices = random.sample(token_indices, num_tokens)

    return important_token_indices
'''

def generate_perturbed_embeddings_on_important_tokens(model, input_ids, important_token_indices, epsilon=0.3, xi=1e-2, ip=1):
    input_ids = input_ids[:, :512]
    inputs_embeds = model.roberta.embeddings(input_ids=input_ids.to(device))
    d = torch.rand_like(inputs_embeds).sub(0.5).to(device)
    d = xi * d / (torch.norm(d, p=2, dim=-1, keepdim=True) + 1e-8)
    d.requires_grad_()
    for _ in range(ip):
        perturbed_inputs_embeds = inputs_embeds.clone()
        perturbed_inputs_embeds[:, important_token_indices, :] += d[:, important_token_indices, :]
        outputs = model(inputs_embeds=perturbed_inputs_embeds)
        probs_perturbed = F.softmax(outputs.logits, dim=-1)
        kl_divergence = F.kl_div(probs_perturbed.log(), probs_perturbed, reduction='batchmean')
        kl_divergence.backward()
        if d.grad is not None:
            grad = d.grad.clone()
            grad_norm = torch.norm(grad[:, important_token_indices, :], p=2, dim=-1, keepdim=True)
            d_new = torch.zeros_like(d)
            d_new[:, important_token_indices, :] = grad[:, important_token_indices, :] / (grad_norm + 1e-8)
            d = d_new.detach()
            d.requires_grad_()
            model.zero_grad()
    r_adv = epsilon * d[:, important_token_indices, :] / (torch.norm(d[:, important_token_indices, :], p=2, dim=-1, keepdim=True) + 1e-8)
    perturbed_inputs_embeds = inputs_embeds.clone()
    perturbed_inputs_embeds[:, important_token_indices, :] += r_adv
    return perturbed_inputs_embeds

def calculate_perturbation_rate(original_text, adversarial_text):

    original_tokens = tokenizer.tokenize(original_text)[:512]
    adversarial_tokens = tokenizer.tokenize(adversarial_text)[:512]
    
    min_length = min(len(original_tokens), len(adversarial_tokens))
    original_tokens = original_tokens[:min_length]
    adversarial_tokens = adversarial_tokens[:min_length]
    
    if len(original_tokens) == 0:
        return 0.0

    matcher = SequenceMatcher(None, original_tokens, adversarial_tokens)
    changed_tokens = 0
    for opcode in matcher.get_opcodes():
        tag, i1, i2, j1, j2 = opcode
        if tag != 'equal':
            changed_tokens += max(i2 - i1, j2 - j1)
    perturbation_rate = changed_tokens / len(original_tokens)
    return perturbation_rate

import string
def is_readable(token):
    token = token.lstrip('Ġ').strip()
    if not token:
        return False
    if all(char in string.punctuation for char in token):
        return False
    if any(char.isalnum() for char in token):
        return True
    return False

def get_important_token_indices_with_importance_module(model, importance_module, input_ids, num_important_tokens):
    input_ids = input_ids[:, :512]
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

    outputs = model.roberta(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True
    )
    hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
    importance_scores = importance_module(hidden_states).squeeze(-1)  # (batch_size, seq_len)
    special_tokens_mask = tokenizer.get_special_tokens_mask(input_ids[0], already_has_special_tokens=True)
    special_tokens_mask = torch.tensor(special_tokens_mask).to(device)
    importance_scores = importance_scores.masked_fill(special_tokens_mask.bool().unsqueeze(0), float('-inf'))
    num_tokens = min(num_important_tokens, importance_scores.size(1))
    _, indices = torch.topk(importance_scores, num_tokens, dim=1)
    important_token_indices = indices[0].tolist()
    return important_token_indices

def generate_adversarial_text_perturbation(input_text, max_iterations=60, num_important_tokens_start=2, epsilon_start=0.8, epsilon_max=1.0, epsilon_step=0.05):
    input_ids = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512)['input_ids'].to(device)
    epsilon = epsilon_start
    best_adversarial_text = input_text
    global query
    query = 0
    original_class = query_classifier(input_text)
    num_important_tokens = num_important_tokens_start

    for step in range(max_iterations):
        important_token_indices = get_important_token_indices_with_importance_module(model, importance_module, input_ids, num_important_tokens)
        perturbed_inputs_embeds = generate_perturbed_embeddings_on_important_tokens(
            model, input_ids, important_token_indices, epsilon=epsilon, xi=1e-1)
        with torch.no_grad():
            outputs = model.lm_head(perturbed_inputs_embeds)
            adversarial_ids = input_ids.clone()
            for idx in important_token_indices:
                candidate_logits = outputs[0, idx]
                probs = F.softmax(candidate_logits, dim=-1)
                top_k_indices = torch.topk(probs, k=20).indices
                found = False
                for candidate_id in top_k_indices:
                    candidate_token = tokenizer.convert_ids_to_tokens([candidate_id.item()])[0]
                    if is_readable(candidate_token):
                        adversarial_ids[0, idx] = candidate_id.item()
                        found = True
                        break
                if not found:
                    adversarial_ids[0, idx] = input_ids[0, idx]
            adversarial_text = clean_special_tokens(tokenizer.decode(adversarial_ids[0], skip_special_tokens=True))
        adversarial_class = query_classifier(adversarial_text)
        if adversarial_class != original_class:
            unpruned_text = adversarial_text
            best_adversarial_text = adversarial_text
            break
        epsilon = min(epsilon + epsilon_step, epsilon_max)
        num_important_tokens = min(num_important_tokens + 1 + step, len(input_ids[0]))
    else:
        print("Attack Failed!")
        return best_adversarial_text, best_adversarial_text

    original_ids = input_ids[0].cpu().numpy()
    adversarial_ids_np = adversarial_ids[0].cpu().numpy()
    modified_indices = [i for i, (o_id, a_id) in enumerate(zip(original_ids, adversarial_ids_np)) if o_id != a_id]
    for idx in modified_indices:
        temp_ids = adversarial_ids.clone()
        temp_ids[0, idx] = input_ids[0, idx]  
        temp_text = clean_special_tokens(tokenizer.decode(temp_ids[0], skip_special_tokens=True))
        temp_class = query_classifier(temp_text)
        if temp_class != original_class:
            adversarial_ids = temp_ids
            best_adversarial_text = temp_text
        else:
            continue

    return best_adversarial_text, unpruned_text

def mask_synonym_substitution(text, important_word_indices):
    tokens = tokenizer.tokenize(text)
    for idx in important_word_indices:
        if idx < len(tokens):
            tokens[idx] = tokenizer.mask_token  

    masked_text = tokenizer.convert_tokens_to_string(tokens)
    
    inputs = tokenizer(masked_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    for idx in important_word_indices:
        if idx < logits.size(1):  
            predicted_token_id = torch.argmax(logits[0, idx]).item()
            tokens[idx] = tokenizer.convert_ids_to_tokens([predicted_token_id])[0]

    return tokenizer.convert_tokens_to_string(tokens)

def mask_synonym_substitution(text, important_word_indices):
    tokens = tokenizer.tokenize(text)
    
    important_word_indices = [idx for idx in important_word_indices if idx < len(tokens)]
    
    for idx in important_word_indices:
        tokens[idx] = tokenizer.mask_token  

    masked_text = tokenizer.convert_tokens_to_string(tokens)
    
    inputs = tokenizer(masked_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    for idx in important_word_indices:
        if idx < logits.size(1):  
            top_k = 5
            predicted_token_ids = torch.topk(logits[0, idx], top_k).indices.tolist()
            for token_id in predicted_token_ids:
                token = tokenizer.convert_ids_to_tokens([token_id])[0]
                if token != tokens[idx]:
                    tokens[idx] = token
                    break  
            else:
                tokens[idx] = tokens[idx]
    
    return tokenizer.convert_tokens_to_string(tokens)

def generate_adversarial_text_mask(input_text, max_iterations=50, num_important_words=2):
    input_ids = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512)['input_ids'].to(device)

    original_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    adversarial_tokens = original_tokens.copy()
    global query
    query = 0
    original_class = query_classifier(input_text)

    for step in range(max_iterations):
        important_word_indices = get_important_token_indices(model, input_ids, num_important_words)

        masked_adversarial_text = mask_synonym_substitution(tokenizer.convert_tokens_to_string(adversarial_tokens), important_word_indices)

        marked_adversarial_text = clean_special_tokens(masked_adversarial_text)

        adversarial_tokens = tokenizer.tokenize(marked_adversarial_text)
        input_ids = tokenizer(marked_adversarial_text, return_tensors='pt', truncation=True, max_length=512)['input_ids'].to(device)

        adversarial_class = query_classifier(marked_adversarial_text)

        if adversarial_class == 1 - original_class:
            unpruned_adversarial_text = marked_adversarial_text  
            break

        num_important_words = min(num_important_words + 1, len(original_tokens))
    else:
        print("Attack failed.")
        return input_text, input_text  

    # Pruning Start #
    adversarial_ids = tokenizer.convert_tokens_to_ids(adversarial_tokens)
    original_ids = tokenizer.convert_tokens_to_ids(original_tokens)

    modified_indices = [i for i, (o_id, a_id) in enumerate(zip(original_ids, adversarial_ids)) if o_id != a_id]

    for idx in modified_indices:
        if idx >= len(adversarial_tokens):
            continue  
        temp_tokens = adversarial_tokens.copy()
        temp_tokens[idx] = original_tokens[idx]  

        temp_text = tokenizer.convert_tokens_to_string(temp_tokens)
        temp_text = clean_special_tokens(temp_text)
        temp_class = query_classifier(temp_text)

        if temp_class == 1 - original_class:
            adversarial_tokens = temp_tokens
            marked_adversarial_text = temp_text
        else:
            continue

    perturbation_rate_after_pruning = calculate_perturbation_rate(input_text, marked_adversarial_text)
    print("Pert. after pruning: ", perturbation_rate_after_pruning)
    print("Final Query: ", query)

    if original_class != query_classifier(marked_adversarial_text):
        print("SUCCESSFUL!")
    else:
        print("FAILED!")

    return marked_adversarial_text, unpruned_adversarial_text  # 返回剪枝后的文本和剪枝前的文本

import os 
import json

class CustomDataset:
    def __init__(self, file_path):
        self.texts = []
        with open(file_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                self.texts.append(entry['text'])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

import pyphen
from textblob import TextBlob

def calculate_readability(text):
    dic = pyphen.Pyphen(lang='en')
    blob = TextBlob(text)
    sentences = len(blob.sentences)
    words = len(blob.words)
    syllables = sum([len(dic.inserted(word).split('-')) for word in blob.words])
    if sentences > 0 and words > 0:
        # Flesch-Kincaid Grade Level
        fk_grade = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59
    else:
        fk_grade = None  
    return fk_grade

def process_dataset(dataset, args, output_file_template):
    k = 0
    total_delta_ppl = 0
    total_use_similarity = 0
    total_delta_readability = 0
    readable_results_count = 0
    total_queries = 0
    successful_attacks = 0
    total_perturbation_rate = 0

    total_pre_prune_ppl = 0  
    total_pre_prune_use_similarity = 0  
    total_pre_prune_perturbation_rate = 0  
    total_pre_prune_delta_readability = 0  
    pre_prune_readable_results_count = 0  

    output_file = output_file_template.format(
        avg_delta_ppl=0, avg_use_similarity=0, avg_delta_readability=0, avg_queries=0, asr=0, avg_perturbation=0,
        avg_pre_prune_ppl=0, avg_pre_prune_use=0, avg_pre_prune_perturbation=0, avg_pre_prune_delta_readability=0)
    output_path = os.path.join("", output_file)

    file_exists = os.path.exists(output_path)
    file_mode = 'a' if file_exists else 'w'

    with open(output_path, file_mode, encoding='utf-8') as f:
        for text in dataset:
            if query_classifier(text) == 1:             # only attack MGT
                max_retries = 10
                retries = 0

                while retries < max_retries:
                    try:
                        if args.method == 'perturbation':
                            adversarial_text, unpruned_adversarial_text = generate_adversarial_text_perturbation(text, max_iterations=500)
                        elif args.method == 'mask':
                            adversarial_text, unpruned_adversarial_text = generate_adversarial_text_mask(text, max_iterations=500)

                        current_ppl = calculate_gpt2_ppl(adversarial_text)
                        original_ppl = calculate_gpt2_ppl(text)
                        delta_ppl = current_ppl - original_ppl

                        pre_prune_ppl = calculate_gpt2_ppl(unpruned_adversarial_text)
                        total_pre_prune_ppl += pre_prune_ppl

                        original_readability = calculate_readability(text)
                        adversarial_readability = calculate_readability(adversarial_text)
                        if original_readability is not None and original_readability != 0:
                            delta_readability = (adversarial_readability - original_readability) / original_readability * 100
                            total_delta_readability += delta_readability
                            readable_results_count += 1
                        else:
                            delta_readability = None

                        pre_prune_readability = calculate_readability(unpruned_adversarial_text)
                        if pre_prune_readability is not None and original_readability is not None and original_readability != 0:
                            pre_prune_delta_readability = (pre_prune_readability - original_readability) / original_readability * 100
                            total_pre_prune_delta_readability += pre_prune_delta_readability
                            pre_prune_readable_results_count += 1
                        else:
                            pre_prune_delta_readability = None

                        similarity_score = calculate_use_similarity(text, adversarial_text)
                        total_delta_ppl += delta_ppl
                        total_use_similarity += similarity_score

                        pre_prune_similarity_score = calculate_use_similarity(text, unpruned_adversarial_text)
                        total_pre_prune_use_similarity += pre_prune_similarity_score

                        perturbation_rate = calculate_perturbation_rate(text, adversarial_text)
                        total_perturbation_rate += perturbation_rate

                        pre_prune_perturbation_rate = calculate_perturbation_rate(text, unpruned_adversarial_text)
                        total_pre_prune_perturbation_rate += pre_prune_perturbation_rate

                        attack_success = query_classifier(adversarial_text) != query_classifier(text)
                        total_queries += query

                        if attack_success:
                            successful_attacks += 1

                        result = {
                            "text": adversarial_text,
                            "delta_ppl": delta_ppl,
                            "use_similarity": similarity_score,
                            "delta_readability_percent": delta_readability,
                            "queries": query,
                            "perturbation_rate": perturbation_rate,
                            "attack_success": attack_success,
                            'label': 1,
                            "pre_prune_ppl": pre_prune_ppl,
                            "pre_prune_use_similarity": pre_prune_similarity_score,
                            "pre_prune_perturbation_rate": pre_prune_perturbation_rate,
                            "pre_prune_delta_readability_percent": pre_prune_delta_readability
                        }

                        json.dump(result, f)
                        f.write('\n')

                        k += 1
                        print(f"Processing: {k}")
                        break

                    except Exception as e:
                        print(f"Error, repeat time: {retries + 1}, Error: {e}")
                        raise e     # or print(e)
                        retries += 1
            else:
                print("SKIPPED.")
                continue
        avg_delta_ppl = total_delta_ppl / k if k > 0 else 0
        avg_use_similarity = total_use_similarity / k if k > 0 else 0
        avg_delta_readability = total_delta_readability / readable_results_count if readable_results_count > 0 else 0
        avg_queries = total_queries / k if k > 0 else 0
        avg_perturbation_rate = total_perturbation_rate / k if k > 0 else 0
        asr = successful_attacks / k if k > 0 else 0

        avg_pre_prune_ppl = total_pre_prune_ppl / k if k > 0 else 0
        avg_pre_prune_use_similarity = total_pre_prune_use_similarity / k if k > 0 else 0
        avg_pre_prune_perturbation_rate = total_pre_prune_perturbation_rate / k if k > 0 else 0
        avg_pre_prune_delta_readability = total_pre_prune_delta_readability / pre_prune_readable_results_count if pre_prune_readable_results_count > 0 else 0

        final_output_file = output_file_template.format(
            avg_delta_ppl=avg_delta_ppl, avg_use_similarity=avg_use_similarity,
            avg_delta_readability=avg_delta_readability, avg_queries=avg_queries, asr=asr, avg_perturbation=avg_perturbation_rate,
            avg_pre_prune_ppl=avg_pre_prune_ppl, avg_pre_prune_use=avg_pre_prune_use_similarity, avg_pre_prune_perturbation=avg_pre_prune_perturbation_rate,
            avg_pre_prune_delta_readability=avg_pre_prune_delta_readability
        )
        final_output_path = os.path.join("", final_output_file)
        os.rename(output_path, final_output_path)

        print(f"Dataset has saved: {final_output_file}")

dataset_path = ''
dataset = CustomDataset(dataset_path)

output_file_template = f"{args.method}-avg_query-{{avg_queries:.4f}}-ASR-{{asr:.4f}}-avg_perturbation-{{avg_perturbation:.4f}}-avg_delta_ppl-{{avg_delta_ppl:.4f}}-use_similarity-{{avg_use_similarity:.4f}}-readability_change-{{avg_delta_readability:.4f}}-pre_prune_ppl-{{avg_pre_prune_ppl:.4f}}-pre_prune_use-{{avg_pre_prune_use:.4f}}-pre_pert-{{avg_pre_prune_perturbation:.4f}}-pre_readability-{{avg_pre_prune_delta_readability:.4f}}.jsonl"

process_dataset(dataset, args, output_file_template)
