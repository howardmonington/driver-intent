# example run command: python3 fine-tune-bert.py --model "bert-base-uncased"

import argparse
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import DatasetDict, Dataset, load_from_disk
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import wandb 


parser = argparse.ArgumentParser(description='Fine-tune BERT model')
parser.add_argument('--model', type=str, default='bert-base-uncased', help='Model type (e.g., "bert-base-uncased", "bert-large-uncased")')
args = parser.parse_args()


wandb.init(project="driver-intent-classification", name="bert-run")

path_to_retrieve = "../tokenized_dataset"
dataset_dict = load_from_disk(path_to_retrieve)

model_name = args.model  
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=5)

def compute_metrics(p):
    logits, labels = p.predictions, p.label_ids
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    wandb.log({"accuracy": acc})  
    return {"accuracy": acc}

training_args = TrainingArguments(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    output_dir='/results',
    num_train_epochs=1,
    evaluation_strategy="steps",
    save_steps=10,
    save_total_limit=2,
    remove_unused_columns=False,
    run_name='run_name',
    logging_dir='/logs',
    logging_steps=10,
    load_best_model_at_end=True,
    report_to='wandb', 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["test"],
    compute_metrics=compute_metrics, 
)

trainer.train()