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
import os
import evaluate
import numpy as np
import random
from huggingface_hub import login
import wandb

from utils import load_data, TextDataset


model_id  = "google/gemma-2b-it"
max_seqlen = 1024
lora_rank = 4
# lora_rank = 32
# batch_size = 8
batch_size = 64
# batch_size = 32
n_epochs = 3
n_epochs = 20
# use_narrations = True
use_narrations = False
min_correct = 4.0
min_sensible = 4.0


with open("hf_token.txt", "r") as f:
    hf_token = f.read().strip()
login(hf_token)


# Load the dataset
df = load_data(min_correct=min_correct, min_sensible=min_sensible)
train_dataset = TextDataset(df, tokenizer=None, include_narration=use_narrations, split="train")
val_dataset = TextDataset(df, tokenizer=None, include_narration=use_narrations, split="val")
test_dataset = TextDataset(df, tokenizer=None, include_narration=use_narrations, split="test")
ds_train = Dataset.from_dict({
    "query": [f"{q}\n{n}" for q, n in zip(train_dataset.queries, train_dataset.narrations)] if use_narrations else train_dataset.queries,
    "label": train_dataset.labels,
})
ds_val = Dataset.from_dict({
    "query": [f"{q}\n{n}" for q, n in zip(val_dataset.queries, val_dataset.narrations)] if use_narrations else val_dataset.queries,
    "label": val_dataset.labels,
})
ds_test = Dataset.from_dict({
    "query": [f"{q}\n{n}" for q, n in zip(test_dataset.queries, test_dataset.narrations)] if use_narrations else test_dataset.queries,
    "label": test_dataset.labels,
})

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.model_max_length = max_seqlen
print(f' Vocab size of the model {model_id}: {len(tokenizer.get_vocab())}')
print(f" max model input length: {tokenizer.model_max_length}")

# preprocess dataset
preprocess_function = lambda examples: tokenizer(examples["query"], truncation=True)
tokenized_ds_train = ds_train.map(preprocess_function, batched=True)
tokenized_ds_val = ds_val.map(preprocess_function, batched=True)
tokenized_ds_test = ds_test.map(preprocess_function, batched=True)
labels = set(ds_train['label'])
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# defining eval metrics
# metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])
metric = evaluate.combine(["accuracy"])

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
    r=lora_rank,
    # lora_alpha=8,  # Dimensionality of the adapter projection
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,  # Dropout rate for the adapter
    bias="none",  # Bias configuration for the adapter
    task_type="SEQ_CLS"  # Task type (sequence classification in this case)
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# os.environ["WANDB_PROJECT"] = "parse-ego4d"
# os.environ["WANDB_ENTITY"] = "rug-minds"
wandb.init(
    project="parse-ego4d",
    entity="rug-minds",
    config={
        "model_name": model_id,
        "use_narrations": use_narrations,
        "lora_rank": lora_rank,
        "max_seqlen": max_seqlen,
        "batch_size": batch_size,
        "min_correct": min_correct,
        "min_sensible": min_sensible,
    }
)

from transformers import TrainerCallback
from copy import deepcopy

class CustomCallback(TrainerCallback):
    def __init__(self, trainer, test_ds) -> None:
        super().__init__()
        self._trainer = trainer
        self._test_ds = test_ds

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._test_ds, metric_key_prefix="test_")
            return control_copy

training_args = TrainingArguments(
    output_dir="epoch_weights",  # Output directory for checkpoints
    learning_rate=2e-5,  # Learning rate for the optimizer
    per_device_train_batch_size=batch_size,  # Batch size per device
    per_device_eval_batch_size=batch_size,  # Batch size per device for evaluation 
    num_train_epochs=n_epochs,  # Number of training epochs
    weight_decay=0.01,  # Weight decay for regularization
    evaluation_strategy='epoch',  # Evaluate after each epoch
    save_strategy="epoch",  # Save model checkpoints after each epoch
    load_best_model_at_end=True,  # Load the best model based on the chosen metric
    push_to_hub=False,  # Disable pushing the model to the Hugging Face Hub 
    report_to="wandb",  # Disable logging to Weight&Bias
    metric_for_best_model='eval_loss',  # Metric for selecting the best model 
    logging_steps=100,
    logging_strategy='steps',
)
early_stop = EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=.0)
trainer = Trainer(
    model=model,  # The LoRA-adapted model
    args=training_args,  # Training arguments
    train_dataset=tokenized_ds_train,  # Training dataset
    eval_dataset=tokenized_ds_val,  # Evaluation dataset
    tokenizer=tokenizer,  # Tokenizer for processing text
    data_collator=data_collator,  # Data collator for preparing batches
    compute_metrics=compute_metrics,  # Function to calculate evaluation metrics
    # callbacks=[early_stop]  # Optional early stopping callback
)
trainer.add_callback(CustomCallback(trainer, tokenized_ds_test)) 
trainer.train()