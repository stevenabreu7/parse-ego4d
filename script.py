import argparse
import numpy as np
import pandas as pd
import json
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModel
from transformers import AutoTokenizer
from tqdm import tqdm


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


def load_data(folder=""):
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

    return df


def create_custom_model(input_size, output_size, layer_sizes):
    layers = []
    prev_size = input_size
    for size in layer_sizes:
        layers.append(nn.Linear(prev_size, size))
        layers.append(nn.ReLU())
        prev_size = size
    layers.append(nn.Linear(prev_size, output_size))
    layers.append(nn.Softmax(dim=1))
    model = nn.Sequential(*layers)
    return model


def train_model(
    df: pd.DataFrame,
    use_narrations: bool,
    model_name: str, layer_sizes: list[int],
    num_epochs: int, batch_size: int,
    embedding_max_length: int = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = TextDataset(df, tokenizer, include_narration=True, split="train")
    val_dataset = TextDataset(df, tokenizer, include_narration=True, split="val")
    test_dataset = TextDataset(df, tokenizer, include_narration=True, split="test")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if 'gte-base' in model_name:
         embedding_max_length = 8192//4

    emb_model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model = create_custom_model(
        embedding_sizes[model_name] * (2 if use_narrations else 1), 
        train_dataset.n_classes,
        layer_sizes,
    ).to(device)

    wandb.init(
        project="parse-ego4d",
        entity="rug-minds",
        config={
            "model_name": model_name,
            "layer_sizes": layer_sizes,
            "use_narrations": use_narrations,
            "max_length": embedding_max_length,
        }
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Initialize variables for early stopping
    best_loss = float('inf')
    patience = 5
    counter = 0

    # Training loop
    for epoch in range(num_epochs):
        step = epoch * len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}")
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        it = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Training (Epoch {epoch+1})")
        for idx, batch in it:
            step = epoch * len(train_dataloader) + idx
            query_input_ids = batch['query_input_ids'][:, :].to(device)
            query_attention_mask = batch['query_attention_mask'][:, :].to(device)
            if use_narrations:
                narr_input_ids = batch['narr_input_ids'][:, :embedding_max_length].to(device)
                narr_attention_mask = batch['narr_attention_mask'][:, :embedding_max_length].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            # print(f"LOGGING query_input_ids.shape={query_input_ids.shape}, query_attention_mask.shape={query_attention_mask.shape}")
            # print(f"LOGGING narr_input_ids.shape={narr_input_ids.shape}, narr_attention_mask.shape={narr_attention_mask.shape}")
            outputs = emb_model(input_ids=query_input_ids, attention_mask=query_attention_mask)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            if use_narrations:
                narr_outputs = emb_model(input_ids=narr_input_ids, attention_mask=narr_attention_mask)
                narr_embeddings = narr_outputs.last_hidden_state.mean(dim=1)
                embeddings = torch.cat([embeddings, narr_embeddings], dim=1)
            logits = model(embeddings)

            # Compute loss
            loss = loss_fn(logits, labels)
            running_loss += loss.detach().cpu().item()
            running_correct += (logits.argmax(dim=1) == labels).sum().detach().cpu().item()
            running_total += labels.shape[0]
            wandb.log({"batch_loss": loss.item()}, step=step)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = running_loss / len(train_dataloader)
        epoch_accuracy = running_correct / running_total
        print(f" Loss: {epoch_loss}, Accuracy: {epoch_accuracy}")
        wandb.log({"train_loss": epoch_loss, "train_accuracy": epoch_accuracy}, step=step)

        # Validation
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_dataloader:
                query_input_ids = batch['query_input_ids'][:, :].to(device)
                query_attention_mask = batch['query_attention_mask'][:, :].to(device)
                if use_narrations:
                    narr_input_ids = batch['narr_input_ids'][:, :embedding_max_length].to(device)
                    narr_attention_mask = batch['narr_attention_mask'][:, :embedding_max_length].to(device)
                labels = batch['label'].to(device)
                # Forward pass
                outputs = emb_model(input_ids=query_input_ids, attention_mask=query_attention_mask)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                if use_narrations:
                    narr_outputs = emb_model(input_ids=narr_input_ids, attention_mask=narr_attention_mask)
                    narr_embeddings = narr_outputs.last_hidden_state.mean(dim=1)
                    embeddings = torch.cat([embeddings, narr_embeddings], dim=1)
                logits = model(embeddings)
                # Compute loss
                val_loss += loss_fn(logits, labels).detach().cpu().item()
                # Compute accuracy
                val_correct += (logits.argmax(dim=1) == labels).detach().sum().cpu().item()
                val_total += labels.shape[0]

        val_loss /= len(val_dataloader)
        val_acc = val_correct / val_total
        print(f" Validation loss: {val_loss}, accuracy: {val_acc}")
        wandb.log({"val_loss": val_loss, "val_accuracy": val_acc}, step=step)

        if epoch % 5 == 0:
            # Testing
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch in test_dataloader:
                    query_input_ids = batch['query_input_ids'][:, :].to(device)
                    query_attention_mask = batch['query_attention_mask'][:, :].to(device)
                    if use_narrations:
                        narr_input_ids = batch['narr_input_ids'][:, :embedding_max_length].to(device)
                        narr_attention_mask = batch['narr_attention_mask'][:, :embedding_max_length].to(device)
                    labels = batch['label'].to(device)
                    # Forward pass
                    outputs = emb_model(input_ids=query_input_ids, attention_mask=query_attention_mask)
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                    if use_narrations:
                        narr_outputs = emb_model(input_ids=narr_input_ids, attention_mask=narr_attention_mask)
                        narr_embeddings = narr_outputs.last_hidden_state.mean(dim=1)
                        embeddings = torch.cat([embeddings, narr_embeddings], dim=1)
                    logits = model(embeddings)
                    # Compute loss
                    val_loss += loss_fn(logits, labels).detach().cpu().item()
                    # Compute accuracy
                    val_correct += (logits.argmax(dim=1) == labels).detach().sum().cpu().item()
                    val_total += labels.shape[0]

            val_loss /= len(test_dataloader)
            val_acc = val_correct / val_total
            print(f" Test loss: {val_loss}, accuracy: {val_acc}")
            wandb.log({"test_loss": val_loss, "test_accuracy": val_acc}, step=step)

        # Check for early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            wandb.log({"stop_counter": counter, "best_val_loss": best_loss}, step=step)
        else:
            counter += 1
            wandb.log({"stop_counter": counter, "best_val_loss": best_loss}, step=step)
            if counter >= patience:
                print("Early stopping triggered.")
                break


def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("--model_name", type=str, help="Name of the model")
    parser.add_argument("--use_narrations", type=str2bool, default=False, help="Flag to indicate whether to use narrations")
    parser.add_argument("--layer_sizes", type=str, help="List of layer sizes")
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs (default: 100)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    model_name = args.model_name
    use_narrations = args.use_narrations
    layer_sizes = [] if args.layer_sizes == "" else [int(size) for size in args.layer_sizes.split(",")]
    N_EPOCHS = args.n_epochs
    BATCH_SIZE = args.batch_size
    print(f"Running with layer_sizes={layer_sizes}, use_narrations={use_narrations}, model_name={model_name}, n_epochs={N_EPOCHS}, batch_size={BATCH_SIZE}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    embedding_sizes = {
        "avsolatorio/GIST-small-Embedding-v0": 384,
        "Alibaba-NLP/gte-base-en-v1.5": 768,
        "Alibaba-NLP/gte-large-en-v1.5": 1024,
        "google/mobilebert-uncased": 512,
        # "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        # "distilbert-base-uncased",
    }
    embedding_max_tokens = {
        "avsolatorio/GIST-small-Embedding-v0": 512,
        "Alibaba-NLP/gte-base-en-v1.5": 8192,
        "Alibaba-NLP/gte-large-en-v1.5": 8192,
        "google/mobilebert-uncased": 512,
        # "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        # "distilbert-base-uncased",
    }
    model_names = list(embedding_sizes.keys())

    if model_name not in model_names:
        raise ValueError("Invalid model_name")

    print('Loading model:', model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    print("Loading data...")
    # folder = "./drive/MyDrive/stevenabreu@/Ego4D stevenabreu@/Ego4D data/External/colab/"
    # folder = "/content/drive/MyDrive/stevenabreu@/Ego4D stevenabreu@/Ego4D data/External/colab/"
    folder = ""
    df = load_data(folder)

    print(f"Training {model_name} with layer sizes {layer_sizes} and use_narrations={use_narrations}")
    train_model(
        df,
        model_name=model_name,
        use_narrations=use_narrations,
        layer_sizes=layer_sizes,
        num_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        embedding_max_length=embedding_max_tokens[model_name],
    )
    # try:
    # except Exception as e:
    #     print('\n')
    #     print("ERROR training model:", e)
    #     print('\n')


# if __name__ == "__main__":
#     N_EPOCHS = 100
#     BATCH_SIZE = 64
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     embedding_sizes = {
#         "avsolatorio/GIST-small-Embedding-v0": 384,
#         "Alibaba-NLP/gte-base-en-v1.5": 768,
#         "Alibaba-NLP/gte-large-en-v1.5": 1024,
#         "google/mobilebert-uncased": 128,
#         # "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
#         # "distilbert-base-uncased",
#     }
#     model_names = list(embedding_sizes.keys())

#     for model_name in model_names:
#         print('Loading model:', model_name)
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

#     print("Loading data...")
#     # folder = "./drive/MyDrive/stevenabreu@/Ego4D stevenabreu@/Ego4D data/External/colab/"
#     # folder = "/content/drive/MyDrive/stevenabreu@/Ego4D stevenabreu@/Ego4D data/External/colab/"
#     folder = ""
#     df = load_data(folder)

#     list_layer_sizes = [
#         [256],
#         [512],
#         [1024],
#         [512, 512],
#         [1024, 1024],
#     ]

#     print()
#     print("Total number of training configs:", len(model_names) * len(list_layer_sizes) * 2)
#     print()
#     for use_narrations in [False, True]:
#         for model_name in model_names:
#             for layer_sizes in list_layer_sizes:
#                 print(f"Training {model_name} with layer sizes {layer_sizes} and use_narrations={use_narrations}")
#                 try:
#                     train_model(
#                         df,
#                         model_name=model_name,
#                         use_narrations=use_narrations,
#                         layer_sizes=layer_sizes,
#                         num_epochs=N_EPOCHS,
#                         batch_size=BATCH_SIZE,
#                     )
#                 except Exception as e:
#                     print('\n')
#                     print("ERROR training model:", e)
#                     print('\n')
