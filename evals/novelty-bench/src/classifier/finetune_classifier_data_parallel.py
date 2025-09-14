import numpy as np
import argparse
import json
import math
import os
import sys
from collections import deque
from contextlib import nullcontext

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

parser = argparse.ArgumentParser()

OUTPUT_DIR = "/checkpoint/ram/tianjian/math-classifier/qwen3-4b-emb-finetuned"
TRAIN_FILE = "/checkpoint/ram/tianjian/math_cots/dapo_14k_data/nm_train_o3.jsonl" # this is the data for math
VAL_FILE = "/checkpoint/ram/tianjian/math_cots/dapo_14k_data/nm_val_o3.jsonl"
LABEL_COLUMN = "similar"
WARMUP_STEPS = 10
TRAIN_STEPS = 80
PRETRAINED_MODEL = "Qwen/Qwen3-Embedding-4B"

# Multi-GPU settings
NUM_GPUS = torch.cuda.device_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE_PER_GPU = 1
BATCH_SIZE = BATCH_SIZE_PER_GPU * NUM_GPUS if NUM_GPUS > 0 else 1
VAL_BATCH_SIZE = BATCH_SIZE
MAX_LR = 2e-5
MIN_LR = 2e-6
MAX_LEN = 2048

# Adjust gradient accumulation to account for multiple GPUs
GRAD_ACC_STEPS = 64 // NUM_GPUS if NUM_GPUS > 0 else 64

AUTOCAST = (
    torch.autocast(
        DEVICE,
        dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32),
    )
    if DEVICE == "cuda"
    else nullcontext()
)


def get_lr(t: int) -> float:
    assert MAX_LR >= MIN_LR >= 0.0
    assert TRAIN_STEPS >= WARMUP_STEPS >= 0

    if t <= WARMUP_STEPS:
        return (t / WARMUP_STEPS) * MAX_LR

    elif t >= TRAIN_STEPS:
        return MIN_LR

    return (MAX_LR - MIN_LR) / 2 * math.cos(
        (t - WARMUP_STEPS) * math.pi / (TRAIN_STEPS - WARMUP_STEPS)
    ) + (MIN_LR + MAX_LR) / 2


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def hyperparameters():
    return {
        key: value
        for key, value in globals().items()
        if key
        in [
            "TRAIN_FILE",
            "VAL_FILE",
            "LABEL_COLUMN",
            "TRAIN_STEPS",
            "WARMUP_STEPS",
            "PRETRAINED_MODEL",
            "BATCH_SIZE",
            "BATCH_SIZE_PER_GPU",
            "NUM_GPUS",
            "MAX_LR",
            "MIN_LR",
            "MAX_LEN",
            "GRAD_ACC_STEPS",
        ]
    }


def to_device(data):
    return {k: v.to(DEVICE) for k, v in data.items()}


class DictDataset(Dataset):
    def __init__(self, data: list[dict]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> dict:
        return self.data[i]


def get_dataloader(tokenizer, df, train):
    data = []
    for _, row in tqdm(df.iterrows(), "preping data..."):
        prompt = row["prompt"]
        generation_0 = row["generation_0"]
        generation_1 = row["generation_1"]
        # if tokenizer does not have cls token and sep token
        # use 151644 as cls and 151645 as sep
        if tokenizer.cls_token_id is None:
            tokenizer.cls_token_id = 151644
        if tokenizer.sep_token_id is None:
            tokenizer.sep_token_id = 151645

        input_ids = [tokenizer.cls_token_id] 
        for s in [generation_0, generation_1]:
            input_ids.extend(
                tokenizer.encode(
                    s,
                    truncation=True,
                    max_length=MAX_LEN,
                    add_special_tokens=False,
                )
            )
            input_ids.append(tokenizer.sep_token_id)
            prompt_len = input_ids.index(tokenizer.sep_token_id) + 1
        token_type_ids = [0] * prompt_len + [1] * (
            len(input_ids) - prompt_len
        )

        if 'Qwen' in PRETRAINED_MODEL:
            data.append(
                {
                    "input_ids": torch.LongTensor(input_ids),
                    "labels": row[LABEL_COLUMN],
                }
            )
        else:
            data.append(
                {
                    "input_ids": torch.LongTensor(input_ids),
                    "token_type_ids": torch.LongTensor(token_type_ids),
                    "labels": row[LABEL_COLUMN],
                }
            )

        if train:
            input_ids = [tokenizer.cls_token_id]
            for s in [generation_1, generation_0]:
                input_ids.extend(
                    tokenizer.encode(
                        s,
                        truncation=True,
                        max_length=MAX_LEN,
                        add_special_tokens=False,
                    )
                )
                input_ids.append(tokenizer.sep_token_id)
                prompt_len = input_ids.index(tokenizer.sep_token_id) + 1
            token_type_ids = [0] * prompt_len + [1] * (
                len(input_ids) - prompt_len
            )
            
            if 'Qwen' in PRETRAINED_MODEL:
                data.append(
                    {
                        "input_ids": torch.LongTensor(input_ids),
                        "labels": row[LABEL_COLUMN],
                    }
                )
            else:
                data.append(
                    {
                        "input_ids": torch.LongTensor(input_ids),
                        "token_type_ids": torch.LongTensor(token_type_ids),
                        "labels": row[LABEL_COLUMN],
                    }
                )

    dataset = DictDataset(data)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE if train else VAL_BATCH_SIZE,
        shuffle=train,
        pin_memory=True,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )


def get_train_iter(dl):
    while True:
        for batch in dl:
            yield to_device(batch)


def main():
    # Print GPU information
    print(f"Number of GPUs available: {NUM_GPUS}")
    if NUM_GPUS > 0:
        for i in range(NUM_GPUS):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")
    
    print(f"Using batch size of {BATCH_SIZE_PER_GPU} per GPU, total batch size: {BATCH_SIZE}")
    print(f"Gradient accumulation steps: {GRAD_ACC_STEPS}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "hyperparams.json"), "w") as f:
        json.dump(hyperparameters(), f)

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_data = pd.read_json(TRAIN_FILE, lines=True)
    val_data = pd.read_json(VAL_FILE, lines=True)
    train_dl = get_dataloader(tokenizer, train_data, True)
    val_dl = get_dataloader(tokenizer, val_data, False)
    train_iter = get_train_iter(train_dl)

    # Create the model on the primary device
    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL, num_labels=2, trust_remote_code=True).to(DEVICE)
    
    # Wrap model with DataParallel if multiple GPUs are available
    if NUM_GPUS > 1:
        model = torch.nn.DataParallel(model)
    
    if model.module.config.pad_token_id is None if isinstance(model, torch.nn.DataParallel) else model.config.pad_token_id is None:
        if isinstance(model, torch.nn.DataParallel):
            model.module.config.pad_token_id = tokenizer.pad_token_id
        else:
            model.config.pad_token_id = tokenizer.pad_token_id
            
    # Calculate class weights
    class_counts = train_data[LABEL_COLUMN].value_counts().to_dict()
    total_count = sum(class_counts.values())
    weights = [total_count / class_counts[i] for i in range(len(class_counts))]
    class_weights = torch.FloatTensor(weights).to(DEVICE)
    
    opt = torch.optim.AdamW(model.parameters(), lr=0.0, fused=DEVICE == "cuda")
    train_losses = deque(maxlen=16)
    train_accs = deque(maxlen=16)
    
    with tqdm(total=TRAIN_STEPS) as pbar:
        for i in range(TRAIN_STEPS):
            lr = get_lr(i)
            for g in opt.param_groups:
                g["lr"] = lr
            for _ in range(GRAD_ACC_STEPS):
                batch = next(train_iter)
                with AUTOCAST:
                    outputs = model(**batch)
                    logits = outputs.logits
                    loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
                    loss = loss_fct(logits, batch["labels"])
                    loss = loss / GRAD_ACC_STEPS  # Normalize the loss based on accumulation steps
                    preds = outputs["logits"].argmax(-1)
                
                train_accs.append((batch["labels"] == preds).to(torch.float32).mean().item())
                train_losses.append(loss.item() * GRAD_ACC_STEPS)  # Store the unnormalized loss for reporting
                loss.backward()
            opt.step()
            opt.zero_grad()
            pbar.update(1)
            pbar.set_postfix(
                {
                    "Running train loss": f"{np.mean(train_losses).item()}",
                    "Running train acc": f"{np.mean(train_accs).item()}"
                }
            )
            if i != 0 and i % 10 == 0:
                # Save the model state (handle DataParallel wrapper if present)
                if isinstance(model, torch.nn.DataParallel):
                    torch.save(model.module.state_dict(), os.path.join(OUTPUT_DIR, f"model-{i}.pt"))
                else:
                    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"model-{i}.pt"))

    # Save the final model
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), os.path.join(OUTPUT_DIR, "model.pt"))
    else:
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.pt"))

    for step in range(TRAIN_STEPS):
        if step % 10 == 0 and step != 0:
            # Evaluate the model on the validation set
            model_path = os.path.join(OUTPUT_DIR, f"model-{step}.pt")
            
            # Load the model for evaluation (handle with or without DataParallel)
            if isinstance(model, torch.nn.DataParallel):
                model.module.load_state_dict(torch.load(model_path))
            else:
                model_state = torch.load(model_path)
                model.load_state_dict(model_state)
                
            model.eval()
            labels = val_data[LABEL_COLUMN].tolist()
            preds = []
            with torch.inference_mode():
                for batch in tqdm(val_dl, total=math.ceil(len(val_data) / VAL_BATCH_SIZE)):
                    batch = to_device(batch)
                    with AUTOCAST:
                        outputs = model(**batch)
                    preds_batch = outputs["logits"].argmax(-1)
                    preds.extend(preds_batch.flatten().tolist())
            assert len(labels) == len(preds)

            val_data[LABEL_COLUMN + "-pred"] = preds

            val_eval = {}
            val_eval["precision"] = precision_score(labels, preds)
            val_eval["recall"] = recall_score(labels, preds)
            val_eval["f1"] = f1_score(labels, preds)
            val_eval["accuracy"] = accuracy_score(labels, preds)
            print(f"Step {step} - Validation results:")
            print(json.dumps(val_eval, indent=2))
            
    # Eval final checkpoint
    model_path = os.path.join(OUTPUT_DIR, "model.pt")
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path))
        
    model.eval()
    labels = val_data[LABEL_COLUMN].tolist()
    preds = []
    with torch.inference_mode():
        for batch in tqdm(val_dl, total=math.ceil(len(val_data) / VAL_BATCH_SIZE)):
            batch = to_device(batch)
            with AUTOCAST:
                outputs = model(**batch)
            preds_batch = outputs["logits"].argmax(-1)
            preds.extend(preds_batch.flatten().tolist())
    assert len(labels) == len(preds), f"Labels and predictions length mismatch: {len(labels)} != {len(preds)}"
    val_data[LABEL_COLUMN + "-pred"] = preds
    val_eval = {}
    val_eval["precision"] = precision_score(labels, preds)
    val_eval["recall"] = recall_score(labels, preds)
    val_eval["f1"] = f1_score(labels, preds)
    val_eval["accuracy"] = accuracy_score(labels, preds)
    print("Final Validation results:")
    print(json.dumps(val_eval, indent=2))


if __name__ == "__main__":
    main()