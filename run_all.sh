#!/bin/bash

# layer_sizes=("" "256" "512" "1024" "512,512" "1024,1024")
# model_names=("avsolatorio/GIST-small-Embedding-v0")
layer_sizes=("" "512" "512,512" "1024,1024")
model_names=("avsolatorio/GIST-small-Embedding-v0" "Alibaba-NLP/gte-base-en-v1.5" "Alibaba-NLP/gte-large-en-v1.5", "google/mobilebert-uncased")

for layer_size in "${layer_sizes[@]}"; do
    for model_name in "${model_names[@]}"; do
        sbatch run_experiment.sh --layer_sizes="$layer_size" --model_name="$model_name"
        sbatch run_experiment.sh --layer_sizes="$layer_size" --model_name="$model_name" --use_narrations
    done
done
