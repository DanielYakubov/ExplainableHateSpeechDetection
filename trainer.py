import logging
from typing import List, Tuple

import evaluate
import numpy as np
import pandas as pd
import sklearn
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          BertModel, Trainer, TrainingArguments, get_scheduler)

from data.helpers.make_dataset import HateSpanDataset

logging.basicConfig(level=logging.INFO)

TOKENIZER = AutoTokenizer.from_pretrained('distilbert-base-uncased')
MODEL = BertModel.from_pretrained('bert-base-uncased',
                                 output_hidden_states=True)


def get_embeddings(span: str) -> object:
    """
    Calculates the embedding for a span. The embedding is calculated by taking the average of
    the second to last layer of a BERT model for each token in the span
    :param span: str span
    :return: an embedding for the span
    """
    span = "[CLS]" + span + "[SEP]"
    tokenized_span = list(TOKENIZER(span))
    indexed_tokens = TOKENIZER.convert_tokens_to_ids(tokenized_span)
    segment_ids = [1] * len(indexed_tokens)
    tok_tensor = torch.tensor([indexed_tokens])
    seg_tensor = torch.tensor([segment_ids])
    with torch.no_grad():
        outputs = MODEL(tok_tensor, seg_tensor)    
        hidden_states = outputs[2]
    token_vecs = hidden_states[-2][0]
    # calculate average of second to last
    sentence_embedding = torch.mean(token_vecs, dim=0)
    return sentence_embedding


def preprocess_dataset_for_classification(dataset_filepath: str) -> Tuple[List[object], List[int]]:
    """
    preprocesses the span dataset. Spans that have a_s > 0 and have a post level label in
    ["hatespeech", "offensive"] are kept as is, everything else is converted to normal labels.
    :param dataset_filepath: a file path to a dataset containing spans, the columns must include
    ["span", "post_hs_label", "span_label"]
    :return: a tuple containing a list of embeddings and a list of labels for the output
    """
    l2n = {
        "normal": 0,
        "offensive": 1,
        "hatespeech": 2
    }

    data = pd.read_csv(dataset_filepath, delimiter='\t')
    texts = data["span"].to_list()
    labels = data["post_hs_label"]
    a_s = data["span_label"]

    out_labels = []
    for l, a in zip(labels, a_s):
        l = l2n[l]
        # if a = 0 the span is "normal" i.e. not a rationale for hate/offensive label, even if the whole text is hate/offensive
        if a == 0:
            l = 0
        out_labels.append(l)
    encodings = TOKENIZER(texts, truncation=True, padding=True, max_length=30) # don't expect any spans to be very long
    return encodings, out_labels


def preprocess_dataset_for_att_classification(dataset_filepath):
    """
    preprocesses the span dataset for agreeableness classification
    :param dataset_filepath: a file path to a dataset containing spans, the columns must include
    ["span", "span_label"]
    :return: a tuple containing a list of embeddings and a list of labels for the output
    """
    data = pd.read_csv(dataset_filepath, delimiter='\t')
    texts = data["span"].to_list()
    labels = data["span_label"]
    encodings = TOKENIZER(texts, truncation=True, padding=True, max_length=30) # don't expect any spans to be very long
    return encodings, labels


if __name__ == "__main__":
    # Classification of span label
    # train_encodings, train_labels = preprocess_dataset_for_classification('data/datasets/span_annotation_train.tsv')
    # val_encodings, val_labels = preprocess_dataset_for_classification('data/datasets/span_annotation_val.tsv')
    # train_dataset = HateSpanDataset(train_encodings, train_labels)
    # val_dataset = HateSpanDataset(val_encodings, val_labels)

    # train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=12)
    # test_dataloader = DataLoader(val_dataset, batch_size=12)

    # classifier = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', 
    #                                                             num_labels=3)

    # optimizer = AdamW(classifier.parameters(), lr=5e-5)

    # num_epochs = 3
    # num_training_steps = num_epochs * len(train_dataloader)
    # lr_scheduler = get_scheduler(
    #     name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    # )
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # classifier.to(device)

    # # training loop
    # progress_bar = tqdm(range(num_training_steps))
    # classifier.train()
    # for epoch in range(num_epochs):
    #     for batch in train_dataloader:
    #         batch = {k: v.to(device) for k, v in batch.items()}
    #         outputs = classifier(**batch)
    #         loss = outputs.loss
    #         loss.backward()

    #         optimizer.step()
    #         lr_scheduler.step()
    #         optimizer.zero_grad()
    #         progress_bar.update(1)
    #     # saving the model per epoch
    #     # torch.save(classifier, 'models/span_classifier_model.pth')
    
    # # evaluate
    # metric = evaluate.load("f1")
    # classifier.eval()
    # for batch in test_dataloader:
    #     batch = {k: v.to(device) for k, v in batch.items()}
    #     with torch.no_grad():
    #         outputs = classifier(**batch)
    #     logits = outputs.logits
    #     predictions = torch.argmax(logits, dim=-1)
    #     metric.add_batch(predictions=predictions, references=batch["labels"])
    # res = metric.compute(average='micro')
    # print(res)



    # Classification of span label strength
    # train_encodings, train_labels = preprocess_dataset_for_att_classification('data/datasets/span_annotation_train.tsv')
    val_encodings, val_labels = preprocess_dataset_for_att_classification('data/datasets/span_annotation_val.tsv')
    # train_dataset = HateSpanDataset(train_encodings, train_labels)
    val_dataset = HateSpanDataset(val_encodings, val_labels)

    # train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=5)
    test_dataloader = DataLoader(val_dataset, batch_size=5)

    # classifier = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', 
    #                                                             num_labels=6)

    # optimizer = AdamW(classifier.parameters(), lr=5e-5)

    # num_epochs = 3
    # num_training_steps = num_epochs * len(train_dataloader)
    # lr_scheduler = get_scheduler(
    #     name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    # )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # classifier.to(device)

    # # training loop
    # progress_bar = tqdm(range(num_training_steps))
    # classifier.train()
    # for epoch in range(num_epochs):
    #     for batch in train_dataloader:
    #         batch = {k: v.to(device) for k, v in batch.items()}
    #         outputs = classifier(**batch)
    #         loss = outputs.loss
    #         loss.backward()

    #         optimizer.step()
    #         lr_scheduler.step()
    #         optimizer.zero_grad()
    #         progress_bar.update(1)
    #     # saving the model per epoch
    #     torch.save(classifier, 'models/span_intensity_cls_model.pth')
    classifier = torch.load('models/span_intensity_cls_model.pth')
    classifier.eval()
    
    # evaluate
    metric = evaluate.load("f1")
    classifier.eval()
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = classifier(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    res = metric.compute(average='micro')
    print(res)
