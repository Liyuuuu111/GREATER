import csv
import json
import numpy as np
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
from textattack.models.wrappers import PyTorchModelWrapper
import pandas as pd
from textattack.attack_recipes import PWWSRen2019, TextFoolerJin2019, BERTAttackLi2020
from textattack.attack_recipes.faster_genetic_algorithm_jia_2019 import FasterGeneticAlgorithmJia2019
from textattack.attack_recipes.checklist_ribeiro_2020 import CheckList2020
from textattack import Attacker, AttackArgs
from textattack.datasets import Dataset as TextAttackDataset
import os
import jieba
from torch.cuda.amp import autocast
from torch.nn import CrossEntropyLoss
from textattack.transformations import WordSwapMaskedLM
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.goal_functions import UntargetedClassification
from textattack import Attack
from textattack.attack_recipes.a2t_yoo_2021 import A2TYoo2021
import argparse

# Initialize jieba tokenizer
jieba.initialize()  # Force reloading the dictionary and disable cache

# -------------------------------
# Custom Dataset Class
# -------------------------------
class CustomTextAttackDataset(TextAttackDataset):
    def __init__(self, file_path, shuffle=True):
        self.data = pd.read_csv(file_path)
        self.texts = self.data['text'].tolist()
        self.labels = self.data['label'].tolist()
        self.shuffled = shuffle  # Add shuffled attribute
        self.label_names = ["human-written", "machine-written"]  # Set label names according to the task

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# -------------------------------
# Load model and tokenizer
# -------------------------------
def load_model_and_tokenizer(model_path, tokenizer_name='roberta-large'):
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    model = RobertaForSequenceClassification.from_pretrained(model_path)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

# -------------------------------
# Custom Model Wrapper Class
# -------------------------------
class CustomModelWrapper(PyTorchModelWrapper):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, text_input_list):
        # Ensure the input length does not exceed 512
        inputs = self.tokenizer(
            text_input_list, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            # Use autocast for mixed-precision inference
            with torch.amp.autocast('cuda'):  # Enable mixed-precision inference
                outputs = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logits = outputs.logits
        return logits

    def _tokenize(self, inputs):
        # Use tokenizer to encode and extract input_ids
        tokenized_inputs = [
            self.tokenizer(x, return_tensors='pt', padding=True, truncation=True, max_length=512)
            for x in inputs
        ]
        # Extract input_ids for each input and convert to tokens
        tokens = [
            self.tokenizer.convert_ids_to_tokens(tokenized_input['input_ids'][0].tolist()) 
            for tokenized_input in tokenized_inputs
        ]
        return tokens

    def get_grad(self, text_input, loss_fn=CrossEntropyLoss()):
        """
        Get the gradient of the loss function with respect to input tokens.

        Args:
            text_input (str): Input text
            loss_fn (torch.nn.Module): Loss function, default is CrossEntropyLoss
        Returns:
            Dictionary with input_ids and corresponding gradients (numpy array)
        """
        if not hasattr(self.model, "get_input_embeddings"):
            raise AttributeError(
                f"{type(self.model)} must implement `get_input_embeddings` method."
            )
        if not isinstance(loss_fn, torch.nn.Module):
            raise ValueError("Loss function must be an instance of torch.nn.Module.")

        self.model.train()

        # Get embedding layer and save original gradient state
        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        # Define a hook function to capture gradients
        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        # Register the hook
        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        # Clear gradients
        self.model.zero_grad()
        model_device = next(self.model.parameters()).device

        # Tokenize the input
        inputs = self.tokenizer(
            text_input, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        input_ids = inputs['input_ids'].to(model_device)

        # Forward pass
        predictions = self.model(input_ids=input_ids)
        # Compute loss (using the predicted class as the label)
        output = predictions.logits.argmax(dim=1)
        loss = loss_fn(predictions.logits, output)
        loss.backward()

        # Get gradient of the embedding layer
        if emb_grads[0].shape[1] == 1:
            grad = torch.transpose(emb_grads[0], 0, 1)[0].cpu().numpy()
        else:
            grad = emb_grads[0][0].cpu().numpy()

        # Restore the original state of the embedding layer and remove the hook
        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        return {"ids": input_ids[0].tolist(), "gradient": grad}

# -------------------------------
# Command-line arguments
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="TextAttack Model Attack Script")
    parser.add_argument('--dataset', type=str, required=True, help="Path to the dataset CSV file")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the pre-trained model checkpoint")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save output files")
    parser.add_argument('--query_budgets', type=int, nargs='+', default=[1500], help="List of query budgets to run attacks")
    return parser.parse_args()

# -------------------------------
# Build BERT-based Attack
# -------------------------------
def build_bert_attack(model_wrapper, max_candidates=5):
    transformation = WordSwapMaskedLM(
        method="bae",  # BERT-based word replacement
        max_candidates=max_candidates
    )
    constraints = [RepeatModification(), StopwordModification()]
    goal_function = UntargetedClassification(model_wrapper)
    search_method = GreedyWordSwapWIR()
    return Attack(goal_function, constraints, transformation, search_method)

# -------------------------------
# Define attack methods
# -------------------------------
def define_attacks(model_wrapper):
    attacks = {
        "PWWSRen2019": PWWSRen2019.build(model_wrapper),
        "TextFooler": TextFoolerJin2019.build(model_wrapper),
        "BERTAttack": build_bert_attack(model_wrapper, max_candidates=5),
        "A2T": A2TYoo2021.build(model_wrapper, mlm=False),  # Using A2T attack without MLM
    }
    return attacks

# -------------------------------
# Main Loop: Run Attacks
# -------------------------------
def run_attacks(dataset, model_wrapper, attacks, query_budgets, output_dir):
    for attack_name, attack in attacks.items():
        for q in query_budgets:
            print(f"Running {attack_name} attack with query budget {q}...")
            csv_output_file = os.path.join(output_dir, f"{attack_name}_attacked_samples_{q}.csv")
            jsonl_output_file = os.path.join(output_dir, f"{attack_name}_attacked_samples_{q}.jsonl")

            # Setup attack parameters
            attack_args = AttackArgs(
                num_examples=500,               # Number of examples to attack (adjust as needed)
                log_to_csv=csv_output_file,     # Save results to CSV format
                checkpoint_dir="checkpoints",   # Directory to save checkpoints
                shuffle=False,                  # Do not shuffle the dataset
                query_budget=q,                 # Current query budget
                num_workers_per_device=64       # Adjust number of workers per device
            )

            # Run attack using TextAttack's Attacker class
            attacker = Attacker(attack, dataset, attack_args)
            attacker.attack_dataset()

            # Convert CSV results to JSONL format
            with open(csv_output_file, newline='', encoding='utf-8') as csvfile:
                csv_reader = csv.DictReader(csvfile)
                with open(jsonl_output_file, mode='w', encoding='utf-8') as jsonlfile:
                    for row in csv_reader:
                        json.dump(row, jsonlfile, ensure_ascii=False)
                        jsonlfile.write("\n")
            
            print(f"{attack_name} attack (query budget {q}) completed. Results saved to '{jsonl_output_file}'\n")

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    args = parse_args()

    # Load dataset
    custom_dataset = CustomTextAttackDataset(args.dataset)

    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(args.model_path)

    # Wrap the model
    model_wrapper = CustomModelWrapper(model, tokenizer)

    # Define attacks
    attacks = define_attacks(model_wrapper)

    # Run attacks
    run_attacks(custom_dataset, model_wrapper, attacks, args.query_budgets, args.output_dir)
