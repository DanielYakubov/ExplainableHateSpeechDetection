import ast
import logging

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_scheduler,
                          BatchEncoding, BertTokenizer)

from data.helpers.make_dataset import HateSpanDataset

logging.basicConfig(level=logging.INFO)

TOKENIZER = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def preprocess_dataset_for_span_classification(
        dataset: pd.DataFrame,
        tokenizer: BertTokenizer = TOKENIZER) -> tuple[BatchEncoding, list[int]]:
    """Preprocesses the span dataset. Spans that have a_s > 0 and have a post level label in
    ["hatespeech", "offensive"] are kept as is, a_s < -1 means the span is mixed,
    everything else is converted to normal labels.

    Args:
        dataset: The dataset dataframe, it must contain the columns ["span", "post_hs_label"]
        tokenizer: The initialized BERT tokenizer

    Returns:
        encodings: A tokenized spans
        out_labels: The list of classification labels as ints
    """
    l2n = {
        "mixed": 0,
        "non-toxic": 1,
        "toxic": 2,
    }
    texts = dataset["span"].to_list()
    labels = dataset["post_hs_label"]
    out_labels = [l2n[l] for l in labels]
    encodings = tokenizer(texts,
                          truncation=True,
                          padding="max_length",
                          max_length=128)
    return encodings, out_labels


def preprocess_dataset_for_e2e_classification(
        dataset: pd.DataFrame,
        tokenizer: BertTokenizer = TOKENIZER) -> tuple[BatchEncoding, list[int]]:
    """Preprocesses the dataset for end-to-end classification.

    Args:
         dataset: The dataset dataframe, it must contain the columns ["span", "post_hs_label"]
         tokenizer: The initialized BERT tokenizer

    Returns:
        encodings: A container of embeddings
        out_labels: The list of classification labels as ints
    """
    texts = [" ".join(ast.literal_eval(pt)) for pt in dataset["post_tokens"]]
    labels = [0 if d == "non-toxic" else 1 for d in dataset["hs_label"]]
    encodings = tokenizer(texts,
                          truncation=True,
                          padding="max_length",
                          max_length=128)
    return encodings, labels


def setup_training(
    classifier: torch.nn.Module, lr: float, num_warmup_steps: int,
    num_epochs: int, dataloader: DataLoader
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR, int]:
    """Set up the optimizer, learning rate scheduler, and compute total training steps.

    Args:
        classifier: The model to fine-tune.
        lr: Learning rate for the optimizer.
        num_warmup_steps: Number of warmup steps for the scheduler.
        num_epochs: Total number of training epochs.
        dataloader: The training dataloader (used to determine steps per epoch).

    Returns:
        optimizer: AdamW optimizer for model parameters.
        scheduler: Linear learning rate scheduler with warmup.
        num_training_steps: Total number of training steps (epochs × batches).
    """
    optimizer = AdamW(classifier.parameters(), lr=lr)
    num_training_steps = num_epochs * len(dataloader)
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler, num_training_steps


def fine_tune(train_encodings: BatchEncoding,
              train_labels: list[int],
              model_name: str = "distilbert-base-uncased",
              num_labels: int = 3,
              lr: float = 5e-5,
              num_epochs: int = 3,
              num_warmup_steps: int = 500,
              save_path="models/span_classifier_model.pth") -> None:
    """Fine-tune a pretrained transformer for sequence classification.

    Args:
        train_encodings: Tokenized input data.
        train_labels: Labels aligned with the encodings.
        model_name: Hugging Face model to start from. Default is "distilbert-base-uncased".
        num_labels: Number of target classes. Default is 3.
        lr: Learning rate. Default is 5e-5.
        num_epochs: Number of training epochs. Default is 3.
        num_warmup_steps: Scheduler warmup steps. Default is 500.
        save_path: Where to save the model after each epoch. Default is "models/span_classifier_model.pth".

    Returns:
        None. Saves the trained model to disk.
    """
    train_dataset = HateSpanDataset(train_encodings, train_labels)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=64)

    classifier = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels)
    optimizer, lr_scheduler, num_training_steps = setup_training(
        classifier, lr, num_warmup_steps, num_epochs, train_dataloader)
    device = (torch.device("cuda")
              if torch.cuda.is_available() else torch.device("cpu"))
    classifier.to(device)

    classifier.train()
    for epoch in tqdm(range(num_epochs)):
        for batch in train_dataloader:
            batch = to_device(batch, device)
            outputs = classifier(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        # TODO add a validation loop, I can't believe I didn't add one here...
        # saving the model per epoch
        torch.save(classifier, save_path)


if __name__ == "__main__":
    # Span Level Classification
    dataset = pd.read_csv("data/datasets/span_annotation_train.tsv",
                          delimiter="\t")
    train_encodings, train_labels = preprocess_dataset_for_span_classification(
        dataset, tokenizer=TOKENIZER)
    fine_tune(train_encodings, train_labels)

    # Full Text Classification
    dataset = pd.read_csv("data/datasets/preprocessed_data_train.tsv",
                          delimiter="\t")
    train_encodings, train_labels = preprocess_dataset_for_e2e_classification(
        dataset, tokenizer=TOKENIZER)
    fine_tune(train_encodings,
              train_labels,
              num_labels=2,
              save_path="models/e2e_classifier_model.pth")