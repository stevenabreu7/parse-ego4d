import argparse
import numpy as np
import pandas as pd
import json
import os
import keras
import keras_nlp
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModel
from transformers import AutoTokenizer
from tqdm import tqdm
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

from utils import load_data, TextDataset, str2bool

model_name = "gemma2_2b_en"
max_seqlen = 256
lora_rank = 4

kaggle_userdata = json.load(open('kaggle.json'))
os.environ["KAGGLE_USERNAME"] = kaggle_userdata['username']
os.environ["KAGGLE_KEY"] = kaggle_userdata['key']
os.environ["KERAS_BACKEND"] = "jax"  # Or "torch" or "tensorflow".
# Avoid memory fragmentation on JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"

df = load_data()
train_dataset = TextDataset(df, tokenizer=None, include_narration=False, split="train")
train_data = []
for i in range(len(train_dataset)):
  query, label = train_dataset[i]
  train_data.append(f"Instruction:\n{query}\n\nResponse:\n{label}")

gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(model_name)
print(gemma_lm.summary())

# Enable LoRA for the model and set the LoRA rank to 4.
gemma_lm.backbone.enable_lora(rank=lora_rank)
print(gemma_lm.summary())

# Limit the input sequence length to 256 (to control memory usage).
gemma_lm.preprocessor.sequence_length = max_seqlen
# Use AdamW (a common optimizer for transformer models).
optimizer = keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=0.01,
)
# Exclude layernorm and bias terms from decay.
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

wandb.init(
    project="parse-ego4d",
    entity="rug-minds",
    config={
        "model_name": model_name,
        "max_seqlen": max_seqlen,
        "lora_rank": lora_rank,
    }
)

gemma_lm.fit(
  train_data, 
  epochs=1, 
  batch_size=1, 
  callbacks=[
    WandbMetricsLogger(log_freq=100),
    WandbModelCheckpoint("models"),
  ],
)

