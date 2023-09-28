import pandas as pd
import torch
import argparse
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import DatasetDict, Dataset, load_from_disk
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, GenerationConfig

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder
from transformers import BitsAndBytesConfig
import bitsandbytes as bnb
from accelerate import Accelerator
from peft import prepare_model_for_kbit_training, LoraConfig, TaskType, get_peft_model
from transformers import TrainingArguments, AutoConfig, \
    AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, DataCollatorWithPadding
from peft import (
    PeftConfig,
    PeftModel,
)
import wandb 
from transformers import EarlyStoppingCallback, IntervalStrategy


parser = argparse.ArgumentParser(description='Fine-tune BERT model with QLoRA')
parser.add_argument('--model', type=str, default='bert-large-uncased', help='Model type (e.g., "bert-base-uncased", "bert-large-uncased")')
args = parser.parse_args()

wandb.init(project="driver-intent-classification", name="qlora-run")


path_to_retrieve = "../tokenized_dataset"
dataset_dict = load_from_disk(path_to_retrieve)

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
model_id = args.model 
num_labels=5

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=5,quantization_config=bnb_config, device_map={"":0})

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

modules = find_all_linear_names(model)

config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.1,
    bias="all",
    task_type=TaskType.SEQ_CLS,
    target_modules = modules
)

peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=12, lora_alpha=32, lora_dropout=0.1)

model = get_peft_model(model, peft_config)

model.print_trainable_parameters()

def compute_metrics(p):
    logits, labels = p.predictions, p.label_ids
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    wandb.log({"accuracy": acc})  
    return {"accuracy": acc}

early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2)

training_args = TrainingArguments(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    output_dir='/results',
    num_train_epochs=8,
    evaluation_strategy="steps",
    save_steps=10,
    save_total_limit=2,
    remove_unused_columns=False,
    run_name='run_name',
    logging_dir='/logs',
    logging_steps=10,
    load_best_model_at_end=True,
    learning_rate=5e-4,
    optim="paged_adamw_8bit",
    report_to='wandb', 
    metric_for_best_model="accuracy", 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["test"],
    compute_metrics=compute_metrics, 
    callbacks=[early_stopping_callback],
)

trainer.train()
