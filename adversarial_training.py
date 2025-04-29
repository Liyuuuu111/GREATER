import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    XLMRobertaTokenizer, XLMRobertaForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    AlbertTokenizer, AlbertForSequenceClassification,
    DebertaTokenizer, DebertaForSequenceClassification,
    ElectraTokenizer, ElectraForSequenceClassification,
    RobertaForMaskedLM,
    GPT2Tokenizer, GPT2ForSequenceClassification
)
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import os
import string
from collections import Counter
from sklearn.utils import resample
from torch.optim.lr_scheduler import ExponentialLR

# model list for detector
model_names = [
    "xlm-roberta-base"
]
model_root_dir = ""         # Put your root dir here

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

generator_model_name = "roberta-large"

generator_tokenizer = RobertaTokenizer.from_pretrained(generator_model_name)
generator_model = RobertaForMaskedLM.from_pretrained(generator_model_name).to(device)
generator_model.eval()
for param in generator_model.parameters():
    param.requires_grad = False

def is_readable(token):
    token = token.lstrip('Ġ').strip()
    if not token:
        return False
    if all(char in string.punctuation for char in token):
        return False
    if any(char.isalnum() for char in token):
        return True
    return False

def clean_special_tokens(text):
    return text.replace("</s>", "").replace("<s>", "").strip()

def generate_perturbed_embeddings_on_selected_tokens(model, input_ids, selected_token_indices, epsilon=0.3, xi=1e-2, ip=1):
    input_ids = input_ids[:, :512]
    # roberta 模型主体
    inputs_embeds = model.roberta.embeddings(input_ids=input_ids.to(device))
    d = torch.rand_like(inputs_embeds).sub(0.5).to(device)
    d = xi * d / (torch.norm(d, p=2, dim=-1, keepdim=True) + 1e-8)
    d.requires_grad_()

    for _ in range(ip):
        perturbed_inputs_embeds = inputs_embeds.clone()
        perturbed_inputs_embeds[:, selected_token_indices, :] += d[:, selected_token_indices, :]
        outputs = model(inputs_embeds=perturbed_inputs_embeds)
        probs_perturbed = F.softmax(outputs.logits, dim=-1)
        kl_divergence = F.kl_div(probs_perturbed.log(), probs_perturbed, reduction='batchmean')
        kl_divergence.backward()
        if d.grad is not None:
            grad = d.grad.clone()
            grad_norm = torch.norm(grad[:, selected_token_indices, :], p=2, dim=-1, keepdim=True)
            d_new = torch.zeros_like(d)
            d_new[:, selected_token_indices, :] = grad[:, selected_token_indices, :] / (grad_norm + 1e-8)
            d = d_new.detach()
            d.requires_grad_()
            model.zero_grad()

    r_adv = epsilon * d[:, selected_token_indices, :] / (torch.norm(d[:, selected_token_indices, :], p=2, dim=-1, keepdim=True) + 1e-8)
    perturbed_inputs_embeds = inputs_embeds.clone()
    perturbed_inputs_embeds[:, selected_token_indices, :] += r_adv
    return perturbed_inputs_embeds

class Generator(nn.Module):
    def __init__(self, model_name):
        super(Generator, self).__init__()

        self.model = RobertaForMaskedLM.from_pretrained(model_name).to(device)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.importance_module = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(device)
        self.importance_module.train()

    def forward(self, input_text, max_query=100, epsilon=0.3):
        encoding = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        outputs = self.model.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = outputs.last_hidden_state  # (1, seq_len, hidden_size)

        importance_scores = self.importance_module(hidden_states).squeeze(-1)  # (1, seq_len)
        importance_scores = torch.sigmoid(importance_scores)

        special_tokens_mask = self.tokenizer.get_special_tokens_mask(input_ids[0], already_has_special_tokens=True)
        special_tokens_mask = torch.tensor(special_tokens_mask).unsqueeze(0).to(device)
        importance_scores = importance_scores.masked_fill(special_tokens_mask.bool(), 0.0)

        k = max(int(max_query), 1)          # At least 1 query
        _, selected_token_indices = torch.topk(importance_scores, k=k, dim=1)
        selected_token_indices = selected_token_indices[0]

        perturbed_inputs_embeds = generate_perturbed_embeddings_on_selected_tokens(
            self.model, input_ids, selected_token_indices, epsilon=epsilon
        )

        with torch.no_grad():
            outputs = self.model.lm_head(perturbed_inputs_embeds)
            adversarial_ids = input_ids.clone()
            for idx in selected_token_indices:
                candidate_logits = outputs[0, idx]
                probs = F.softmax(candidate_logits, dim=-1)
                top_k_indices = torch.topk(probs, k=20).indices
                found = False
                for candidate_id in top_k_indices:
                    candidate_token = self.tokenizer.convert_ids_to_tokens([candidate_id.item()])[0]
                    if is_readable(candidate_token):
                        adversarial_ids[0, idx] = candidate_id.item()
                        found = True
                        break
                if not found:
                    adversarial_ids[0, idx] = input_ids[0, idx]
        adversarial_text = clean_special_tokens(self.tokenizer.decode(adversarial_ids[0], skip_special_tokens=True))

        return [adversarial_text], importance_scores, selected_token_indices

class Discriminator(nn.Module):
    def __init__(self, model_path):
        super(Discriminator, self).__init__()
        model_name_lower = model_path.lower()
        
        if "albert" in model_name_lower:
            model_class = AlbertForSequenceClassification
            tokenizer_path = "albert-base-v2"
            tokenizer_class = AlbertTokenizer
            backbone_attr = "albert"
        elif "roberta" in model_name_lower:
            if "large" in model_name_lower:
                tokenizer_path = "roberta-large"
            else:
                tokenizer_path = "roberta-base"
            model_class = RobertaForSequenceClassification
            tokenizer_class = RobertaTokenizer
            backbone_attr = "roberta"
        elif "xlm-roberta" in model_name_lower:
            if "large" in model_name_lower:
                tokenizer_path = "xlm-roberta-large"
            else:
                tokenizer_path = "xlm-roberta-base"
            model_class = XLMRobertaForSequenceClassification
            tokenizer_class = XLMRobertaTokenizer
            backbone_attr = "roberta"
        elif "deberta" in model_name_lower:
            if "large" in model_name_lower:
                tokenizer_path = "deberta-large"
            else:
                tokenizer_path = "deberta-base"
            model_class = DebertaForSequenceClassification
            tokenizer_class = DebertaTokenizer
            backbone_attr = "deberta"
        elif "electra" in model_name_lower:
            if "large" in model_name_lower:
                tokenizer_path = "electra-large-discriminator"
            else:
                tokenizer_path = "electra-base-discriminator"
            model_class = ElectraForSequenceClassification
            tokenizer_class = ElectraTokenizer
            backbone_attr = "electra"
        else:
            raise ValueError("FAIL.")

        self.model = model_class.from_pretrained(model_path).to(device)
        self.tokenizer = tokenizer_class.from_pretrained(tokenizer_path)
        self.model.train()

        self.backbone = getattr(self.model, backbone_attr, None)
        if self.backbone is None:
            raise AttributeError(f"FAIL")

    def classify(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k,v in inputs.items()}
        # 自动混合精度
        scaler = torch.cuda.amp.GradScaler()
        with torch.cuda.amp.autocast():
            outputs = self.model(**inputs)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1).item()
        return preds, logits

file_path = r'dataset/english.jsonl'
data = []
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))
df = pd.DataFrame(data)

df_10k = df.sample(n=10000, random_state=42).copy()
df_10k['use_generator'] = False

df_10k_label_1 = df_10k[df_10k['label'] == 1]
df_10k_label_1_sample = df_10k_label_1.sample(n=1000, random_state=42)

df_10k.loc[df_10k_label_1_sample.index, 'use_generator'] = True

texts_all = df_10k['text'].tolist()
labels_all = df_10k['label'].tolist()
use_generator_flags = df_10k['use_generator'].tolist()

X_train, X_temp, y_train, y_temp, use_gen_train, use_gen_temp = train_test_split(
    texts_all, labels_all, use_generator_flags, test_size=0.01, random_state=42
)
X_test, X_val, y_test, y_val, use_gen_test, use_gen_val = train_test_split(
    X_temp, y_temp, use_gen_temp, test_size=0.01, random_state=42
)

class TextDataset(Dataset):
    def __init__(self, texts, labels, use_generator_flags, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.use_generator_flags = use_generator_flags
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        use_gen_flag = self.use_generator_flags[idx]

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
            'labels': torch.tensor(label, dtype=torch.long),
            'text': text,
            'use_generator': use_gen_flag
        }

train_dataset = TextDataset(X_train, y_train, use_gen_train, generator_tokenizer, max_len=512)
val_dataset   = TextDataset(X_val,   y_val,   use_gen_val,   generator_tokenizer, max_len=512)
test_dataset  = TextDataset(X_test,  y_test,  use_gen_test,  generator_tokenizer, max_len=512)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, logits, target):
        confidence = 1.0 - self.smoothing
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

loss_fn = LabelSmoothingLoss(smoothing=0.01)            # adjust the ls parameter here

# You can adjust the following parameter to get better performance
# This is the parameter of our paper, see Appendix A.1
original_lr = 1e-6
num_epochs = 6
lambda_importance = 20.0  
lambda_adv = 1.0          

k_percent_values= [0.12]

# load proxy model
# change the proxy model to Roberta-large to get the result of Table 1/2
proxy_model_path = "gpt2_classification_hc3/checkpoint-492"         # put your proxy model here
proxy_model = GPT2ForSequenceClassification.from_pretrained(proxy_model_path).to(device)
proxy_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
proxy_model.eval()  

for k_percent_value in k_percent_values:
    generator = Generator(generator_model_name).to(device)

    for model_name in model_names:
        discriminator_model_path = os.path.join(model_root_dir, model_name)

        discriminator = Discriminator(discriminator_model_path).to(device)

        generator_optimizer = torch.optim.Adam(generator.importance_module.parameters(), lr=1e-3)
        discriminator_optimizer = torch.optim.Adam(discriminator.model.parameters(), lr=original_lr)
        generator_scheduler = ExponentialLR(generator_optimizer, gamma=0.9)
        discriminator_scheduler = ExponentialLR(discriminator_optimizer, gamma=0.9)
        for epoch in range(num_epochs):
            k = 0  
            total_selected_tokens = 0
            total_attacks = 0
            print(f"Epoch {epoch+1}/{num_epochs}")

            for batch in train_loader:
                input_ids_batch = batch['input_ids'].to(device)
                labels_batch = batch['labels'].to(device)
                texts_batch = batch['text']
                use_gen_flags = batch['use_generator']
                batch_size = input_ids_batch.size(0)

                for i in range(batch_size):
                    text = texts_batch[i]
                    original_label = labels_batch[i].item()
                    use_gen = use_gen_flags[i]  

                    total_attacks += 1

                    if use_gen and original_label == 1:
                        for j in range(1, int(k_percent_value*len(text))):
                            adversarial_texts, importance_scores, selected_token_indices = generator(
                                text, max_query=j
                            )
                            adversarial_text = adversarial_texts[0]
                            adversarial_label, discriminator_logits_adv = discriminator.classify(adversarial_text)
                            attack_success = (adversarial_label != original_label)
                            if attack_success:
                                break
                        perturbation_rate = len(selected_token_indices) / len(generator.tokenizer.tokenize(text))
                        total_selected_tokens += len(selected_token_indices)
                    else:
                        adversarial_text = text
                        perturbation_rate = 0.0
                        selected_token_indices = torch.tensor([], dtype=torch.long, device=device)
                        importance_scores = torch.tensor([], device=device)

                    adversarial_label, discriminator_logits_adv = discriminator.classify(adversarial_text)
                    attack_success = (adversarial_label != original_label)
                    if attack_success:
                        k += 1

                    discriminator_inputs_adv = discriminator.tokenizer(
                        adversarial_text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).to(device)
                    discriminator_outputs_adv = discriminator.model(**discriminator_inputs_adv)
                    discriminator_logits_adv = discriminator_outputs_adv.logits

                    generator_target_label = torch.tensor([1 - original_label]).to(device)
                    generator_adv_loss = loss_fn(discriminator_logits_adv, generator_target_label)

                    if use_gen and original_label == 1:
                        proxy_inputs = proxy_tokenizer.encode_plus(
                            text,
                            add_special_tokens=True,
                            truncation=True,
                            max_length=512,
                            return_tensors='pt'
                        )
                        proxy_input_ids = proxy_inputs['input_ids'].to(device)
                        proxy_attention_mask = proxy_inputs['attention_mask'].to(device)
                        proxy_input_ids.requires_grad = False

                        embeddings = proxy_model.transformer.wte(proxy_input_ids)
                        embeddings.requires_grad_(True)
                        embeddings.retain_grad()

                        outputs = proxy_model(
                            inputs_embeds=embeddings,
                            attention_mask=proxy_attention_mask,
                            output_hidden_states=True,
                            return_dict=True
                        )
                        logits = outputs.logits
                        loss_fn_proxy = nn.CrossEntropyLoss()
                        loss_proxy = loss_fn_proxy(logits, torch.tensor([original_label]).to(device))

                        proxy_model.zero_grad()
                        loss_proxy.backward()

                        gradients = embeddings.grad  
                        mask = proxy_attention_mask.unsqueeze(-1)  
                        masked_gradients = gradients * mask  
                        proxy_importance_scores = masked_gradients.norm(dim=-1).squeeze(0)
                        proxy_importance_scores = (
                            (proxy_importance_scores - proxy_importance_scores.min()) /
                            (proxy_importance_scores.max() - proxy_importance_scores.min() + 1e-8)
                        )

                        importance_scores_normalized = importance_scores.squeeze(0)
                        if importance_scores_normalized.numel() > 0:
                            importance_scores_normalized = (
                                (importance_scores_normalized - importance_scores_normalized.min()) /
                                (importance_scores_normalized.max() - importance_scores_normalized.min() + 1e-8)
                            )
                        min_len = min(len(importance_scores_normalized), len(proxy_importance_scores)) if importance_scores_normalized.numel() > 0 else 0
                        if min_len > 0:
                            importance_scores_normalized = importance_scores_normalized[:min_len]
                            proxy_importance_scores = proxy_importance_scores[:min_len]
                            importance_loss_fn = nn.MSELoss()
                            importance_loss = importance_loss_fn(
                                importance_scores_normalized,
                                proxy_importance_scores.detach()
                            )
                        else:
                            importance_loss = torch.tensor(0.0, device=device)

                        generator_total_loss = lambda_adv * generator_adv_loss + lambda_importance * importance_loss

                        generator_optimizer.zero_grad()
                        generator_total_loss.backward()
                        generator_optimizer.step()
                    else:
                        generator_total_loss = torch.tensor(0.0, device=device)

                    for param in discriminator.model.parameters():
                        param.requires_grad = True

                    if original_label == 1:
                        for param_group in discriminator_optimizer.param_groups:
                            param_group['lr'] = 1e-5
                    else:
                        for param_group in discriminator_optimizer.param_groups:
                            param_group['lr'] = 1e-5

                    discriminator_inputs = discriminator.tokenizer(
                        [text, adversarial_text],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).to(device)
                    discriminator_labels = torch.tensor([original_label, original_label]).to(device)

                    discriminator_outputs = discriminator.model(**discriminator_inputs)
                    discriminator_logits = discriminator_outputs.logits
                    discriminator_loss = loss_fn(discriminator_logits, discriminator_labels)

                    discriminator_optimizer.zero_grad()
                    discriminator_loss.backward()
                    discriminator_optimizer.step()

                    print(f"Text {i+1}/{batch_size}, "
                          f"Adversary Loss: {generator_total_loss.item():.4f}, "
                          f"Detector Loss: {discriminator_loss.item():.4f}, "
                          f"Pert.: {perturbation_rate:.4f}")

            torch.cuda.empty_cache()
            generator_scheduler.step()
            discriminator_scheduler.step()
            print(f"Success Times: {k}/{total_attacks}")
            if total_selected_tokens > 0:
                print(f"Avg Pert.: {total_selected_tokens / total_attacks:.4f}")
            else:
                print("Avg Pert.: 0.0000")

        k_percent_str = str(k_percent_value).replace('.', '_')
        discriminator_save_path = os.path.join("model", f"trained_{model_name}_k_{k_percent_str}_gpt2_hc3")
        os.makedirs(discriminator_save_path, exist_ok=True)
        discriminator.model.save_pretrained(discriminator_save_path)
        discriminator.tokenizer.save_pretrained(discriminator_save_path)

    importance_module_save_path = os.path.join("model", f"importance_module_k_{k_percent_str}_gpt2_hc3")
    os.makedirs(importance_module_save_path, exist_ok=True)
    torch.save(generator.importance_module.state_dict(), os.path.join(importance_module_save_path, "pytorch_model.bin"))
    print(f"Important model (k_percent={k_percent_value}) saved successfully.")
