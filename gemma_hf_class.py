import torch
import pandas as pd
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)

# import bitsandbytes as bnb

import evaluate
import numpy as np

import random

from huggingface_hub import login

from utils import load_data, TextDataset


with open("hf_token.txt", "r") as f:
    hf_token = f.read().strip()
login(hf_token)


# Load the dataset
df = load_data()
train_dataset = TextDataset(df, tokenizer=None, include_narration=False, split="train")
ds = Dataset.from_dict({
    "query": train_dataset.queries,
    "label": train_dataset.labels,
})

# load tokenizer
model_id  = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
print(f' Vocab size of the model {model_id}: {len(tokenizer.get_vocab())}')
#Vocab size of the model google/gemma-2b-it: 256000
print(f" max model input length: {tokenizer.model_max_length}")
#max model input length: 2048

# preprocess dataset
preprocess_function = lambda examples: tokenizer(examples["query"], truncation=True)
tokenized_ds = ds.map(preprocess_function, batched=True)
labels = set(ds['label'])
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# defining eval metrics
metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)  # Convert probabilities to predicted labels
    return metric.compute(predictions=predictions, references=labels)

# quant config
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,  # Enables 4-bit quantization
#     bnb_4bit_use_double_quant=True,  # Use double quantization for potentially higher accuracy (optional)
#     bnb_4bit_quant_type="nf4",  # Quantization type (specifics depend on hardware and library)
#     bnb_4bit_compute_dtype=torch.bfloat16  # Compute dtype for improved efficiency (optional)
# )
bnb_config = None

# load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
    quantization_config=bnb_config,
    # device_map={"": 0}
)
model.gradient_checkpointing_enable()  # Enable gradient checkpointing for memory-efficient training

lora_config = LoraConfig(
    r=4,  # Reduction factor (lower r means more parameters in the adapter)
    # lora_alpha=8,  # Dimensionality of the adapter projection
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,  # Dropout rate for the adapter
    bias="none",  # Bias configuration for the adapter
    task_type="SEQ_CLS"  # Task type (sequence classification in this case)
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="epoch_weights",  # Output directory for checkpoints
    learning_rate=2e-5,  # Learning rate for the optimizer
    per_device_train_batch_size=1,  # Batch size per device
    per_device_eval_batch_size=1,  # Batch size per device for evaluation 
    num_train_epochs=5,  # Number of training epochs
    weight_decay=0.01,  # Weight decay for regularization
    evaluation_strategy='epoch',  # Evaluate after each epoch
    save_strategy="epoch",  # Save model checkpoints after each epoch
    load_best_model_at_end=True,  # Load the best model based on the chosen metric
    push_to_hub=False,  # Disable pushing the model to the Hugging Face Hub 
    report_to="none",  # Disable logging to Weight&Bias
    metric_for_best_model='eval_loss'  # Metric for selecting the best model 
)
early_stop = EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=.0)
trainer = Trainer(
    model=model,  # The LoRA-adapted model
    args=training_args,  # Training arguments
    train_dataset=tokenized_imdb["train"],  # Training dataset
    eval_dataset=tokenized_imdb["test"],  # Evaluation dataset
    tokenizer=tokenizer,  # Tokenizer for processing text
    data_collator=data_collator,  # Data collator for preparing batches
    compute_metrics=compute_metrics,  # Function to calculate evaluation metrics
    callbacks=[early_stop]  # Optional early stopping callback
)

trainer.train()
