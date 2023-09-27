import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import DatasetDict, Dataset
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(r'../driver-intent-classification-dataset.csv', encoding='ISO-8859-1')

# I want to remove the intent to turn on the high beams since I think this is a more critical function
# and I want to validate that this works on non-critical functions first
df = df[df['Intent'] != 'turn on high beams']

le = LabelEncoder()
df['labels'] = le.fit_transform(df['Intent'])

train_df, val_df = train_test_split(df, stratify=df['labels'], test_size=0.2, random_state=42)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['Text'], padding='max_length', truncation=True, max_length=512)

train_df_reset = train_df.reset_index(drop=True)
val_df_reset = val_df.reset_index(drop=True)

# Now create Dataset objects
train_dataset = Dataset.from_pandas(train_df_reset)
val_dataset = Dataset.from_pandas(val_df_reset)

# Apply tokenization function to the datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

# Remove the original Text and Intent columns
tokenized_train_dataset = tokenized_train_dataset.remove_columns(['Text', 'Intent'])
tokenized_val_dataset = tokenized_val_dataset.remove_columns(['Text', 'Intent'])

dataset_dict = DatasetDict(train=tokenized_train_dataset, test=tokenized_val_dataset)

path_to_save = "../tokenized_dataset"
dataset_dict.save_to_disk(path_to_save)