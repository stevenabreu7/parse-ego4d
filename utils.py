import argparse
import numpy as np
import pandas as pd
import json
import torch
from torch.utils.data import Dataset


def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')


class TextDataset(Dataset):
    def __init__(self, df, tokenizer, include_narration=True, split=None):
        self.include_narration = include_narration

        if split == "train":
            df = df[df["split"] == "train"]
        elif split == "val":
            df = df[df["split"] == "val"]
        elif split == "test":
            df = df[df["split"] == "test"]

        if self.include_narration:
            self.narrations = df.Narrations.tolist()
        self.queries = df.Query.tolist()
        self.labels = df.Label.tolist()
        self.tokenizer = tokenizer
        self.max_length_query = df.Query.apply(lambda x: len(x)).max().item()
        self.max_length_narr = df.Narrations.apply(lambda x: len(x)).max().item()
        self.n_classes = len(df.Label.unique())
        print("Max query length:", self.max_length_query)
        print("Max narration length:", self.max_length_narr)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        label = self.labels[idx]

        if self.tokenizer is None:
            if self.include_narration:
                return query, self.narrations[idx], label
            else:
                return query, label

        # Tokenize the text
        query_encoding = self.tokenizer.encode_plus(
            query,
            add_special_tokens=True,
            max_length=self.max_length_query,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        sample = {
            'query_input_ids': query_encoding['input_ids'].flatten(),
            'query_attention_mask': query_encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

        if self.include_narration:
            narr = self.narrations[idx]
            narr_encoding = self.tokenizer.encode_plus(
                narr,
                add_special_tokens=True,
                max_length=self.max_length_narr,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            sample['narr_input_ids'] = narr_encoding['input_ids'].flatten()
            sample['narr_attention_mask'] = narr_encoding['attention_mask'].flatten()

        return sample


def load_data(folder="", min_sensible=0.0, min_correct=0.0, min_implicit=0.0):
    with open(folder + "narration_compressed.json", "r") as f:
        narrations = json.load(f)
    list(narrations.keys())[:3]

    csv_file = folder + 'annotations.csv'
    df = pd.read_csv(csv_file)
    n_orig = df.shape[0]
    df = df.dropna(axis=0, subset=["Action", "Query"])
    print(n_orig - df.shape[0], "NaNs dropped.", df.shape[0], "left")
    df.head(1)

    df["Narrations"] = df["PARSE-Ego4D ID"].apply(
        lambda x: "\n".join(narrations["-".join(x.split("-")[:-3]).lower()]["narration_pass_1"][1][
            :(int(x.split("-")[-3]) + int(x.split("-")[-1]) * int(x.split("-")[-2]))
        ]))
    df["Summary"] = df["PARSE-Ego4D ID"].apply(lambda x: narrations["-".join(x.split("-")[:-3]).lower()]["narration_pass_1"][0])
    ua = list(df.Action.unique())
    df["Label"] = df.Action.apply(lambda x: ua.index(x)).astype(int)

    # Filter out sensible actions
    df = df[df["SENSIBLE_mean"] >= min_sensible]
    df = df[df["CORRECT_mean"] >= min_correct]
    df = df[df["IMPLICIT_mean"] >= min_implicit]
    print(f"after filtering, {df.shape[0]} samples are left.")

    return df